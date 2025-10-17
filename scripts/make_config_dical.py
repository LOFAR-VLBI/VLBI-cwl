#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Jurjen de Jong, James Petley, Leah Morabito"

from argparse import ArgumentParser
from collections.abc import Sequence
import os
import ret

from astropy.table import Table
from astropy.coordinates import SkyCoord
import casacore.tables as ct
import numpy as np
import pandas as pd
import tables
from submods.source_selection.selfcal_selection import parse_source_from_h5


def make_config(best_solint, smoothness, imagecat, inputmodel, ms):
    """
    Make config for facetselfcal

    Args:
        best_solint: Optimal solint determined within this script
        smoothness: Optimal smoothness constraint determined within this script
        imagecat: Image catalogue used to decide whether phaseup and bandpass correction needed
        inputmodel: Input skymodel to be added to config file
        ms: MeasurementSet
    """
    # Get the source name
    filename = parse_source_from_h5(os.path.basename(ms))

    # Set defaults
    configdict = {}
    configdict['imsize'] = 1600
    configdict['pixelscale'] = 0.075
    configdict['robust'] = -0.5
    configdict['uvmin'] = 40000
    configdict['maskthreshold'] = [7.0]
    configdict['soltype_list'] = ['scalarphasediff','scalarphase','scalarphase','scalarcomplexgain']
    configdict['soltypecycles_list'] = [0, 0, 0, 2]
    configdict['solint_list'] = [4, 1, 4, "1h"]
    configdict['nchan_list'] = [1, 1, 1, 1]
    configdict['smoothnessconstraint_list'] = [40.0 , 2.0, 40.0, 40.0]
    configdict['antennaconstraint_list'] = ['alldutch', None, None, None]
    configdict['resetsols_list'] = [None, 'alldutch', None, None]
    configdict['docircular'] = 'True'
    configdict['forwidefield'] = 'True'
    configdict['stop'] = 10
    configdict['phaseupstations'] = "'core'"
    configdict['paralleldeconvolution'] = 1024
    configdict['parallelgridding'] = 6
    configdict['channelsout'] = 12
    configdict['fitspectralpol'] = 5
    configdict['update_multiscale'] = 'True'


    # Check number of components in VLASS model
    with open( inputmodel, 'r' ) as f:
        lines = f.readlines()
    if len(lines) > 3:
        # there is more than one component (after the two header lines)
        # start scalarcomplex gain at cycle 7 and then add on another 5 cycles
        configdict['soltypecycles_list'] = [0, 0, 0, 7]
        configdict['stop'] = 15

    # Get time array
    with ct.table(ms, readonly=True, ack=False) as t:
        time = np.unique(t.getcol('TIME'))
    deltime = np.abs(time[1] - time[0])

    # Decide if a bandpass correction is needed
    if imagecat is not None:
        bandpass, phaseup = process_catalog(imagecat, ms)
    else: 
        bandpass = False
        phaseup = True

    print(f"bandpass correction: {bandpass} ---- phaseup: {phaseup}")

    if bandpass:
        configdict['soltype_list'].append('scalarcomplexgain')
        configdict['solint_list'].append("9h")
        configdict['soltypecycles_list'].append(configdict['stop'] - 2)
        configdict['nchan_list'].append(1)
        configdict['smoothnessconstraint_list'].append(0)
        configdict['antennaconstraint_list'].append(None)
        configdict['resetsols_list'].append(None)

    if not phaseup:
        del configdict['phaseupstations']
        reset_idx = [ i for i, val in enumerate(configdict['resetsols_list']) if val == 'alldutch' ][0] + 1
        ## insert another scalarphase
        configdict['soltype_list'].insert(reset_idx,'scalarphase')
        configdict['solint_list'].insert(reset_idx,1)
        configdict['soltypecycles_list'].insert(reset_idx,0)
        configdict['nchan_list'].insert(reset_idx,1)
        configdict['smoothnessconstraint_list'].insert(reset_idx,40.0)
        configdict['antennaconstraint_list'].insert(reset_idx,None)
        configdict['resetsols_list'].insert(reset_idx,'core')
        configdict['robust'] = -1.5

    ## Adjust solution intervals based on best_solint
    best_solint = best_solint * 60 # convert to seconds
    for i in range(len(configdict['soltype_list'])):
        if configdict['soltype_list'] == 'scalarphase':
            configdict['solint_list'][i] = str(int(np.ceil(max(best_solint,deltime))))+'s'
            

    ## Update smoothness constraints based on ionospheric conditions
    configdict = update_smoothness( smoothness, configdict )

    ## average to smallest solution interval if that is larger than data resolution
    phase_inds = [i for i, val in enumerate(configdict['soltype_list']) if val == 'scalarphase'] # Getting indexes for scalarphase
    phase_solint = configdict['solint_list'][phase_inds[0]]
    avgstep = int(np.ceil(max(phase_solint, deltime))) // int(deltime) # Converting to seconds
    if avgstep > 1:
        configdict['avgtimestep'] = avgstep

    configfile = write_config( filename, configdict )
    print("CREATED: " + configfile)

def write_config(filename,configdict):
    """
    Write the dictionary of config parameters to a text file
    """
    ## write out the config file
    with open(filename + ".config.txt", "w") as f:
        keys = configdict.keys()
        for key in keys:
            ## string, int, float, or array
            value = configdict[key]
            match value:
              case str() | int() | float():
                f.write(f'{key} = {value}\n')
              case Sequence():
                ss = []
                for aa in value:
                    match aa:
                      case None | int() | float():
                        # The casting to string is needed for the case of None
                        ss.append(str(aa))
                      case str():
                        ss.append("'" + aa + "'")
                      case _:
                        raise ValueError(f"Element of unexpected type found in {value}")
                    f.write(f'{key} = [{','.join(ss)}]\n')
              case _:
                raise ValueError(f"Value of unexpected type found in {configdict}")
    return filename + ".config.txt"


def get_best_solint(ms, phasediff_output):
    """
    Get best solution interval

    Args:
        ms: MeasurementSet.
        phasediff_output: Path to the Phase-diff CSV output.

    Returns:
        solint: Solution interval in minutes.
    """

    phasediff = pd.read_csv(phasediff_output)
    sourceid = parse_source_from_h5(ms.split("/")[-1])

    for col in ['Source_id', 'source']:  # Handling possible column variations (versions)
        if col in phasediff.columns:
            return phasediff[phasediff[col].apply(parse_source_from_h5) == sourceid]['best_solint'].min()

    raise ValueError("Expected column 'Source_id' or 'source' not found in phasediff_output.")

def process_catalog(imagecat, ms):
    '''
    Search through image_catalogue.csv for two purposes. 
    1. Is calibrator bright enough for final bandpass solve
    2. Is there a nearby source that requires core phaseup

    Args:
        ms: input measurement set
        imagecat: image_catalogue.csv from plot_field.py

    Returns:
        bandpass: bool 
        phaseup: bool
    '''
    im_t = Table.read(imagecat)

    bandpass = False # Default option
    phaseup = True # Default option
    
    with ct.table(f"{ms}/FIELD", readonly=True, ack=False) as field_table:
        phase_dir = field_table.getcol('PHASE_DIR')[0, 0]  # shape: (n_fields, 1, 2)
        ra_rad, dec_rad = phase_dir
        ra_deg = np.degrees(ra_rad)
        dec_deg = np.degrees(dec_rad)

    calibrator_coord = SkyCoord(ra = ra_deg, dec = dec_deg, unit = 'deg')
    image_coords = SkyCoord(ra = im_t['RA'], dec = im_t['DEC'], unit = 'deg')

    # Calculate separation for all sources
    separations = calibrator_coord.separation(image_coords)
    im_t['separation_arcsec'] = separations.arcsec

    # Sort table by separation
    im_t.sort('separation_arcsec')

    # Calibrator should be closest source
    delay_cal = im_t[0]
    
    # Get time
    with ct.table(ms, readonly=True, ack=False) as t:
        time = np.unique(t.getcol('TIME'))
        full_time = np.abs(time[-1] - time[0])

   
    im_t = im_t[1:]

    # 8 hours requires 0.5Jy
    scaling = full_time/(8 * 60 * 60)
    min_flux = 500/np.sqrt(scaling) # This is minimum flux for bandpass solve
    if delay_cal["Total_flux"] > min_flux:
        bandpass = True


    # Some sort of logic for flux weighted - we are imaging a 2arcmin box
    # Anything 2-10 arcmins and bright could be a problem 

    #Filter catalogue to only those bright enough to be a problem
    im_t = im_t[im_t['Total_flux'] > delay_cal['Total_flux']*0.25]

    # Search within 2 arcmin
    small_search = im_t[im_t['separation_arcsec'] < 2*60]

    # Search within 10 arcmin
    large_search = im_t[im_t['separation_arcsec'] < 10*60]
    large_search = large_search[large_search['Total_flux'] > delay_cal['Total_flux']]

    if(len(small_search) == 0) and (len(large_search) == 0):
        phaseup = False 
    

    print('Minimum flux for bandpass: ', min_flux)
    print('Delay cal flux: ', delay_cal['Total_flux'])
    return bandpass, phaseup

def make_utf8(inp):
    """
    Convert input to utf8 instead of bytes

    :param inp: string input
    """
    try:
        inp = inp.decode('utf8')
        return inp
    except (UnicodeDecodeError, AttributeError):
        return inp

def get_scalarphase( ms ):
    outh5 = 'test_phases_'+ms+'.h5'
    with open( 'test_dp3_phasesolve.parset', 'w') as f:
        f.write('msin={:s}\n'.format(ms))
        f.write('msin.datacolumn=DATA\n')
        f.write('msout=.\n')
        f.write('msin.weightcolumn=WEIGHT_SPECTRUM\n')
        f.write('msout.storagemanager=dysco\n')
        f.write('msout.storagemanager.weightbitrate=16\n')
        f.write('steps=[ddecal]\n')
        f.write('ddecal.type=ddecal\n')
        f.write('ddecal.sourcedb=skymodel.txt\n')
        f.write('ddecal.mode=scalarphase\n')
        f.write('ddecal.datause=single\n')
        f.write('ddecal.solveralgorithm=directioniterative\n')
        f.write('ddecal.maxiter=100\n')
        f.write('ddecal.propagatesolutions=True\n')
        f.write('ddecal.solint=1\n')
        f.write('ddecal.nchan=1\n')
        f.write('ddecal.h5parm={:s}\n'.format(outh5))
        f.write('ddecal.uvlambdamin=40000\n')
        f.write('ddecal.smoothnessconstraint=2000000.0\n')
        f.write('ddecal.smoothnessreffrequency=0.0\n')
        f.write('ddecal.smoothnessspectralexponent=-1.0\n')
        f.write('ddecal.smoothnessrefdistance=0.0\n')
        f.write('ddecal.tolerance=0.0001\n')
    os.system( 'DP3 test_dp3_phasesolve.parset') 
    return outh5

def get_smoothing( h5 ):
    with tables.open_file(h5) as H:
        phase_table = H.root.sol000.phase000
        freqs = H.root.sol000.phase000.freq[:]
        axes = make_utf8(phase_table.val.attrs["AXES"]).split(',')
        total_bw = (freqs.max() - freqs.min())*1e-6
        phase_sols = phase_table.val[:] * phase_table.weight[:]
        if 'pol' in axes:
            phase_sols = np.take(phase_sols, [0], axis=axes.index('pol'))

    ref_phase = np.take(phase_sols, [0], axis=axes.index('ant'))
    phase_sols -= ref_phase

    phase_freq_diff = ( np.diff(phase_sols, axis=axes.index('freq')) - np.pi ) % (np.pi*2) - np.pi
    freqsum = np.nansum(phase_freq_diff/(2*np.pi), axis=axes.index('freq') )
    abssum = np.abs(freqsum)
    wrap_count = np.max(abssum)

    freq_per_wrap = total_bw / wrap_count
    smoothness = round( freq_per_wrap / 3., 1) # Sampling 3 times per frequency wrap
    return smoothness

def update_smoothness( smoothness, cfgdict ):
    '''
    smoothness constraint determined by ionosphere for international stations (when reset = 'alldutch')
    this should be set to number of wraps divided by 3. This has been found to be a good sampling 
    for smoothness
    '''
    scalarphase_idx = [ i for i,val in enumerate(cfgdict['soltype_list']) if val == 'scalarphase' ]
    for i in scalarphase_idx:
        reset = cfgdict['resetsols_list'][i]
        if reset == 'alldutch':
            cfgdict['smoothnessconstraint_list'][i] = smoothness
    return cfgdict

def parse_args():
    """
    Command line argument parser

    Returns: parsed arguments
    """

    parser = ArgumentParser(description='Make parameter configuration file for facetselfcal.')
    parser.add_argument('--ms', type=str, help='MeasurementSet')
    parser.add_argument('--imagecat', type=str, help='Image catalogue CSV file')
    parser.add_argument('--inputmodel', type=str, help='Input skymodel')
    parser.add_argument('--phasediff_output', type=str, help='Phasediff CSV output')
    parser.add_argument('--scalarphase-h5', type=str, help='h5 with scalarphase solutions for ionospheric conditions')
    return parser.parse_args()


def main():
    """
    Main function
    """

    args = parse_args()

    best_solint = get_best_solint(args.ms, args.phasediff_output)
    smoothness = get_smoothing(args.scalarphase_h5)
    make_config(best_solint, smoothness, args.imagecat, args.inputmodel, args.ms)

if __name__ == "__main__":
    main()
