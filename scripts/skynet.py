#!/usr/bin/env python3
import numpy as np
import re
import os
import glob

from astropy.coordinates import SkyCoord
from astropy.table import Table
import bdsf

def write_skymodel (model, outname):

    print(f'writing the skymodel for: {model}')
    with open(outname, 'w') as skymodel:
        skymodel.write( "# (Name, Type, Patch, Ra, Dec, I, Q, U, V, MajorAxis, MinorAxis, Orientation, ReferenceFrequency='144e+06', SpectralIndex='[]', LogarithmicSI) = format\n" )
        skymodel.write(', , P0, 00:00:00, +00.00.00\n')
        for i in range(len(model)):
            sra = model[i][3]
            sdec = model[i][4]
            # the angles RA and DEC should be sexigesimal coordinates, which for
            # RA is in hours, minutes, seconds (format "XXhYYmZZs") and for
            # DEC is in degrees, minutes, seconds (format "XXdYYmZZs").
            # These should be formatted as strings. If, instead, the angles are
            # given in decimal degrees (floats), a conversion to the previous format is applied.
            if not isinstance((sra, sdec), (str, str)):
                s = SkyCoord(sra,sdec,unit='degree')
                s = s.to_string(style='hmsdms')
                sra = s.split()[0]
                sdec = s.split()[1]
            sra = sra.replace('h',':').replace('m',':').replace('s','')
            sdec = sdec.replace('d','.').replace('m','.').replace('s','')
            model[i][3] = sra
            model[i][4] = sdec
            number_elements = len(model[i])
            ss_to_write = ",".join(
                str(model[i][j]) for j in range(number_elements)
            )
            skymodel.write( '{:s}\n'.format(ss_to_write) )

def model_from_image( modelImage, smodel, opt_coords, astroSearchRadius=3.0 ):
    img = bdsf.process_image(modelImage, mean_map='zero', rms_map=True, rms_box = (100,10))
    sources = img.sources
    maxval = 0.
    for src in sources:
        maxval = np.max( (maxval, src.total_flux) )
    img = bdsf.process_image(modelImage, mean_map='zero', rms_map=True, rms_box = (100,10), advanced_opts=True, blank_limit=0.01*maxval)
    sources = img.sources
    # Scale model flux density to the provided value.
    tot_flux = 0.
    for src in sources:
        tot_flux = tot_flux + src.total_flux
    flux_scaling = smodel/tot_flux
    ## get positional corrections if available
    delta_ra = 0.
    delta_dec = 0.
    if opt_coords is not None:
        seps = []
        for src in sources:
            src_coords = SkyCoord( src.posn_sky_centroid[0], src.posn_sky_centroid[1], unit='deg' )
            sep = src_coords.separation(opt_coords).to("arcsec").value
            seps.append(sep)
        minsep_idx = np.argmin(seps)
        minsep_src = sources[minsep_idx]
        if seps[minsep_idx] < astroSearchRadius:
            delta_ra = opt_coords.ra.value - minsep_src.posn_sky_centroid[0]
            delta_dec = opt_coords.dec.value - minsep_src.posn_sky_centroid[1]
    tmp = []
    i = 0
    for src in sources:
        ra = src.posn_sky_centroid[0] + delta_ra
        dec = src.posn_sky_centroid[1] + delta_dec
        tflux = src.total_flux * flux_scaling
        dcmaj = src.deconv_size_sky[0]*3600.
        dcmin = src.deconv_size_sky[1]*3600.
        dcpa = src.deconv_size_sky[2]
        component = [ 'ME{:s}'.format(str(i)),'GAUSSIAN','P0',ra,dec,tflux,0.0, 0.0, 0.0, dcmaj, dcmin, dcpa, '144e+06', "[-0.7]", 'true' ]
        tmp.append(component)
        i = i + 1
    sky_model = np.array(tmp)
    return sky_model

################## skynet ##############################

def main (MS, delayCalFile, modelImage='', astroSearchRadius=3.0, skip_vlass=False):

    ## make sure the parameters are the correct format
    # MS is assumed to be of the form:
    # /path/to/MS/{observation_id}_*
    # where observation_id is either the LBCS observation id
    # or the ILTJ name of the source

    # {observation_id} should start with either 'S', 'L', 'I'
    # followed by a number of digits
    if not re.search(r'\/([SLI]\d+)\_', MS):
        ValueError(f"{MS} does not contain a valid observation ID")
    MS = MS.rstrip('/')
    tmp = MS.split('/')[-1]
    MS_src = tmp.split('_')[0]

    t = Table.read( delayCalFile, format='csv' )

    # Check if the skymodel uses a LBCS-format catalogue,
    # which has coordinates in columns 'RA' and 'DEC',
    # or if it uses a LoTSS catalogue, which has its
    # coordinates in columns called 'RA_LOTSS' and 'DEC_LOTSS'
    if (('RA' in t.colnames) and ('DEC' in t.colnames)):
        ra_col = 'RA'
        de_col = 'DEC'
    else:
        ra_col = 'RA_LOTSS'
        de_col = 'DEC_LOTSS'
    src_ids = t['Source_id']

    src_names = []
    for src_id in src_ids:
        if isinstance(src_id, str):
            # Check if the name comes from the LoTSS catalogue
            if src_id.startswith('I'):
                val = str(src_id)
            # Check if the name is the gaussian ID
            elif MS_src.startswith('S'):
                val = 'S'+str(src_id)
            # In this case the name is the LBCS observation ID
            # and starts with an 'L'
            else:
                val = str(src_id)
                if MS_src.startswith('S'):
                    val = 'S'+str(src_id)
                else:
                    val = str(src_id)
        else:
            val = src_id
        src_names.append(val)
    src_idx = [ i for i, val in enumerate(src_names) if MS_src == val ][0]

    # get the coordinate values and the flux from the skymodel
    # and convert the flux from mJy to Jy.
    ra = t[ra_col].data[src_idx]
    dec = t[de_col].data[src_idx]
    smodel = t['Total_flux'].data[src_idx]*1.0e-3

    ## gaia information if available - else use panstarrs if it is available
    if "gaia_id" in t.keys() and t['gaia_id'].data[src_idx] != "--":
        opt_coords = SkyCoord( t['gaia_RA'].data[src_idx], t['gaia_DEC'].data[src_idx], unit='deg' )
    elif "ps_id" in t.keys() and t['ps_id'].data[src_idx] != "--":
        opt_coords = SkyCoord( t['ps_RA'].data[src_idx], t['ps_DEC'].data[src_idx], unit='deg' )
    else:
        opt_coords = None

    ## spectral index information
    if {"alpha_1", "alpha_2"}.issubset(t.keys()) and t['alpha_1'].data[src_idx] != "--":
        a_1 = t['alpha_1'].data[src_idx]
        a_2 = t['alpha_2'].data[src_idx]
    else:
        a_1, a_2 = None, None


    if os.path.isfile(modelImage):
        ## a model image is specified, use it
        print('Using user-specified model {:s}'.format(modelImage))
        sky_model = model_from_image( modelImage, smodel, opt_coords, astroSearchRadius=astroSearchRadius )
    else:
        if not skip_vlass:
            ## search for a vlass image
            lbcs_id = t[src_idx]['Observation']
            vlass_file = glob.glob( os.path.join( os.path.dirname(delayCalFile), '{:s}_vlass.fits'.format(lbcs_id) ) )
            if len(vlass_file) > 0:
                sky_model = model_from_image( vlass_file[0], smodel, opt_coords, astroSearchRadius=astroSearchRadius )
            else:
                print('VLASS image not found, generating a point source model')
                if opt_coords is not None:
                    sky_model = np.array( [ ['ME0','GAUSSIAN','P0',opt_coords.ra.value,opt_coords.dec.value,smodel,0.0,0.0,0.0,0.1,0.0,0.0,'144e+06','[-0.5]', 'true'] ] )
                else:
                    sky_model = np.array( [ ['ME0','GAUSSIAN','P0',ra,dec,smodel,0.0,0.0,0.0,0.1,0.0,0.0,'144e+06','[-0.5]', 'true'] ] )
        else:
            print('generating a point source model')
            if opt_coords is not None:
                sky_model = np.array( [ ['ME0','GAUSSIAN','P0',opt_coords.ra.value,opt_coords.dec.value,smodel,0.0,0.0,0.0,0.1,0.0,0.0,'144e+06','[-0.5]', 'true'] ] )
            else:
                sky_model = np.array( [ ['ME0','GAUSSIAN','P0',ra,dec,smodel,0.0,0.0,0.0,0.1,0.0,0.0,'144e+06','[-0.5]', 'true'] ] )

    ## edit spectral index information if necessary
    if a_1 is not None:
        for i in range(len(sky_model)):
            sky_model[i][13] = f'[{a_1:.3f},{a_2:.3f}]'

    write_skymodel (sky_model,'skymodel.txt')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Skynet script to handle LBCS calibrators.")

    parser.add_argument('MS', type=str, help='Measurement set for which to run skynet')
    parser.add_argument('--delay-cal-file', required=True, type=str,help='delay calibrator information')
    parser.add_argument('--model-image', type=str, help='image for generating starting model', default='')
    parser.add_argument('--astrometric-search-radius', type=float, help='search radius in arcsec to accept a match',default=3.0)
    parser.add_argument('--skip-vlass', action='store_true',dest='skip_vlass', help='skip vlass search and generate point source model')

    args = parser.parse_args()

    main( args.MS, delayCalFile=args.delay_cal_file, modelImage=args.model_image, skip_vlass=args.skip_vlass )
