#!/usr/bin/env python3
import numpy as np
import re
import os

from astropy.coordinates import SkyCoord
from astropy.table import Table
import bdsf

#def new_write_skymodel (

def write_skymodel (model, outname = None):

    print(f'writing the skymodel for: {model}')
    if outname:
        with open(outname, 'w') as skymodel:
            skymodel.write( "# (Name, Type, Patch, Ra, Dec, I, Q, U, V, MajorAxis, MinorAxis, Orientation, ReferenceFrequency='144e+06', SpectralIndex='[]') = format\n" )
            skymodel.write(', , P0, 00:00:00, +00.00.00\n')
            for i in range(len(model)):
                ra = model[i][3]
                dec = model[i][4]
                # the angles RA and DEC should be sexigesimal coordinates, which for
                # RA is in hours, minutes, seconds (format "XXhYYmZZs") and for
                # DEC is in degrees, minutes, seconds (format "XXdYYmZZs").
                # These should be formatted as strings. If, instead, the angles are
                # given in decimal degrees (floats), a conversion to the previous format is applied.
                if isinstance( (ra, dec), (str, str) ):
                    sra = ra
                    sdec = dec
                else:
                    s = SkyCoord(ra,dec,unit='degree')
                    s = s.to_string(style='hmsdms')
                    sra = s.split()[0]
                    sdec = s.split()[1]
                sra = sra.replace('h',':').replace('m',':').replace('s','')
                sdec = sdec.replace('d','.').replace('m','.').replace('s','')
                model[i][3] = sra
                model[i][4] = sdec
                ss_to_write = ''
                for j in np.arange(0,len(model[i])):
                    ss_to_write = ss_to_write + str(model[i][j]) + ','
                ss_to_write = ss_to_write.rstrip(',')
                skymodel.write( '{:s}\n'.format(ss_to_write) )

################## skynet ##############################

def main (MS, delayCalFile, modelImage=''):

    ## make sure the parameters are the correct format
    # MS is assumed to be of the form:
    # /path/to/MS/{observation_id}_*
    # We are only interested in {observation_id} here,
    # and discard the rest.

    # {observation_id} should start with either 'S', 'L', 'I'
    # followed by a number of digits
    if not re.search(r'\/([SLI]\d+)\_', MS):
        ValueError(f"{MS} does not contain a valid observation ID")
    MS = MS.rstrip('/')
    tmp = MS.split('/')[-1]
    MS_src = tmp.split('_')[0]

    ## get flux from best_delay_calibrators.csv
    t = Table.read( delayCalFile, format='csv' )
    ## find the RA column

    mycols = t.colnames
    # Check if the skymodel uses a LBCS-format catalogue,
    # which has coordinates in columns 'RA' and 'DEC',
    # or if it uses a LoTSS catalogue, which has its
    # coordinates in columns called 'RA_LOTSS' and 'DEC_LOTSS'
    if (('RA' in mycols) and ('DEC' in mycols)):
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

    if modelImage == '':
        print('generating point model')
        sky_model = np.array( [ ['ME0','GAUSSIAN','P0',ra,dec,smodel,0.0,0.0,0.0,0.1,0.0,0.0,'144e+06','[-0.5]'] ] )
        write_skymodel (sky_model,'skymodel.txt')
    else:
        print('using input image to generate model')
        img = bdsf.process_image(modelImage, mean_map='zero', rms_map=True, rms_box = (100,10))
        img.write_catalog(format='fits', outfile='tmp.fits')
        t = Table.read('tmp.fits',format='fits')
        maxval = np.max(t['Total_flux'])
        img = bdsf.process_image(modelImage, mean_map='zero', rms_map=True, rms_box = (100,10), advanced_opts=True, blank_limit=0.01*maxval)
        img.write_catalog(format='fits',outfile='tmp.fits',clobber=True)
        t = Table.read('tmp.fits', format='fits')
        total_flux = np.sum(t['Total_flux'])
        fluxes = t['Total_flux']*smodel/total_flux
        tmp = []
        for i in np.arange(0,len(t)):
            component = [ 'ME{:s}'.format(str(i)),'GAUSSIAN','P0',t['RA'][i],t['DEC'][i],fluxes[i],0.0, 0.0, 0.0, t['DC_Maj'][i]*3600., t['DC_Min'][i]*3600., t['DC_PA'][i], '144e+06', "[-0.7]" ]
            tmp.append(component)
        sky_model = np.array(tmp)
        write_skymodel( sky_model, 'skymodel.txt')
        os.system( 'rm tmp.fits' )

        ## scale to LoTSS
        #Name, Type, Ra, Dec, I, MajorAxis, MinorAxis, Orientation
        ## add spectral index and Q, U, V

        ## want to write
        #format = Name, Type, Patch, Ra, Dec, I, Q, U, V, MajorAxis, MinorAxis, Orientation, ReferenceFrequency='3.00000e+09', SpectralIndex='[]'
        #use ReferenceFrequency --> 144e6 and the type of specindex


        #img.write_catalog(format='bbs', bbs_patches='source', outfile='test_skymodel.txt', clobber=True)
        ## update the flux density based on LoTSS

    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Skynet script to handle LBCS calibrators.")

    parser.add_argument('MS', type=str, help='Measurement set for which to run skynet')
    parser.add_argument('--delay-cal-file', required=True, type=str,help='delay calibrator information')
    parser.add_argument('--model-image', type=str, help='model image to start with', default='')

    args = parser.parse_args()

    main( args.MS, delayCalFile=args.delay_cal_file, modelImage=args.model_image )
