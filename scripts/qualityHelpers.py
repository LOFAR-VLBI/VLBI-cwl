from astropy.io import fits
import numpy as np
import os
import sys
from glob import glob
import pandas as pd
import csv
from pprint import pprint
from glob import glob
from astropy.modeling import models, fitting
import matplotlib.pyplot as plt
from astropy.coordinates import *
from datetime import timedelta, datetime
from astropy.time import Time

def calculate_expected_rms(ra,dec,obsdate,observation_time=8):
    # Define observer's latitude and longitude
    observer_lat = 52.9088  # latitude of LOFAR
    observer_lon = 6.8674   # longitude of LOFAR

    obs = {'integration':8}
    field = {'ra':ra,'dec':dec,'obsdate_start':obsdate}
    def calculate_elevation(ra,dec, observer_lat, observer_lon, observation_time):
        location = EarthLocation(lat=observer_lat, lon=observer_lon)
        sky_coord = SkyCoord(ra=ra, dec=dec, unit='deg')
        altaz = sky_coord.transform_to(AltAz(obstime=observation_time, location=location))
        return altaz.alt.deg

    elevations = []
    for timeoffset in  np.arange(0, obs['integration'], 0.5):
        obs_datetime = field['obsdate_start'] + timedelta(hours=timeoffset)
        elevation = calculate_elevation(field['ra'], field['dec'], observer_lat, observer_lon, Time(obs_datetime))
        elevations.append(elevation)
    mean_elevation = np.mean(elevations)

    elevation_correction_factor = np.cos(np.radians(90 - mean_elevation))**2.0
    #flagging_correction_factor = 1 - median_flagged #once we know the glagging factor
    duration_correction_factor = (np.sum(obs['integration'])/(8.0*231.0)) # Here its assuming an 8hr observation with 231 subbands (48MHz)
    #corrected_rms = meanrms * elevation_correction_factor * (flagging_correction_factor * duration_correction)
    correction_factor_rms = elevation_correction_factor * duration_correction_factor #(flagging_correction_factor * duration_correction)
    return correction_factor_rms

####
def image_quality(rms, DR, peak, expected_rms, cut_DR=10, cut_rms=3):
    """
    Get image quality acceptance column

    Args:
        csv_table: CSV with image-based scores
        :cut_DR: Dynamic range of image
        :cut_rms: RMS of image
    """
   # df = pd.read_csv(csv_table)
   # if expected_rms == False:
   #     expected_rms = calculate_expected_rms(ra,dec)

    #df['accept_image'] = False

    # Filter for bad data
   # mask = ~((df.Dyn_range > DR_limit) |
    #         (df.RMS < rms_limit))
    #df.loc[mask, 'accept_image'] = True
   # df.to_csv(csv_table, index=False)
    #return csv_table
    rms_limit = expected_rms*cut_rms
    DR_limit = cut_DR
    valid = (rms <= rms_limit) and (DR >= DR_limit)
    diagnostics = {
            'rms': rms,
            'dynamic_range': DR,
            'peak': peak,
            'rms_limit': rms_limit,
            'dynamic_range_limit': DR_limit,
            'valid': valid,
            }
    return valid, diagnostics

class ImageData(object):
    """ Load lofar image and get basic info
    """
    def __init__(self,
                 fits_file="",
                 residual_file="",
                 catalogues=[],
                 noise_method="",
                 load_general_info="",
                ):
        self.fits_file = fits_file
        self.residual_file = residual_file
        self.rms = None,
        if fits_file!="":
            with fits.open(fits_file) as hdu:
                self.hdu_list = hdu
                self.image_data = hdu[0].data
                self.header = hdu[0].header
            try:
                self.Z = self.image_data[0, 0, :, :]
            except:
                self.Z = self.image_data

            self.pixelscale = abs(self.header["CDELT1"]) #in deg
            self.imagesize = self.header["NAXIS1"]
            self.RA = self.header["CRVAL1"]
            self.DEC = self.header["CRVAL2"]
            self.Date = self.header["DATE-OBS"]
            self.Date = datetime.strptime(self.Date, '%Y-%m-%dT%H:%M:%S.%f')
            self.freq = self.header["CRVAL3"]
        else:
            sys.stdout.write("Provide fits file\n")
            exit()

        if residual_file!="":
            with fits.open(residual_file) as hdu:
                self.residual_data = hdu[0].data
                try:
                    self.residual_Z = self.residual_data[0,0, :, :]
                except:
                    self.residual_Z = self.residual_data
    def get_rms(self,
                maskSup:float = 1e-7,
                sigma:int = 3.,
                noise_method:str = "Histogram Fit",
                use_residual_img = True,
                plotfile = 'Histogram_fit.png',
               ):
        """
        Get rms from map

        Args:
            :maskSup: mask theshold
        """
        if self.residual_file !="":
            if use_residual_img:
                print("Using residual image for estimating rms.\n")
                self.rms = get_image_rms(self.residual_Z, maskSup=maskSup, noise_method=noise_method, residual_image=self.residual_Z,plotfile=plotfile)
            else:
                print("Using image for estimating rms.\n")
                self.rms = get_image_rms(self.Z, maskSup=maskSup, noise_method=noise_method, residual_image=self.residual_Z,plotfile=plotfile,clip=True)
        else:
            print("Using image for estimating rms.\n")
            self.rms = get_image_rms(self.Z, maskSup=maskSup, noise_method=noise_method,plotfile=plotfile,clip=True)

        #self.expected_rms = calculate_expected_rms(self.RA,self.DEC,self.Date,observation_time=8)
        self.expected_rms = 75e-5 #until we have a better expectation/functioning calculation

        return self.rms  # jy/beam

    def get_statistics(self):
        self.peak = get_peakflux(self.image_data)
        self.min = get_min(self.image_data)
        self.minmax = get_minmax(self.image_data)
        if self.rms != None:
            self.dyn_range = get_dyn_range(self.peak,self.rms)
        else:
            self.rms = self.get_rms(self)
            self.dyn_range = get_dyn_range(self.peak,self.rms)

        print("Statistics derived:\nPeak={}\nMin={}\nMinMax={}\nDyn_range={}\n".format(self.peak,self.min,self.minmax,self.dyn_range))
        return self.peak,self.min,self.minmax,self.dyn_range

    def get_quality(self):
        valid, diagnostics = image_quality(self.rms, self.dyn_range, self.peak, self.expected_rms, cut_DR=10, cut_rms=3)

###########################

def get_peakflux(image):
    """
    Get peak intensity

    :return: peak intensity
    """
    sys.stdout.write("Deriving peakflux.\n")
    data_max = image.max()
    peak_location = np.where(image.max()==data_max)
    return data_max
####
def get_min(image):
    """
    Get peak intensity

    :return: peak intensity
    """
    sys.stdout.write("Deriving min.\n")
    data_min = image.min()
    return data_min
####
def get_minmax(image):
    """
    Get min/max
    """
    sys.stdout.write("Deriving minmax.\n")
    try:
        data_min = image.data_min
    except:
        data_min = get_min(image)
    try:
        data_max = image.data_max
    except:
        data_max = get_peakflux(image)
    data_minmax = np.abs(data_min/data_max)
    return data_minmax
####
def get_dyn_range(peak,rms):
    """
    Get the dynamic range.
    """
    print("Deriving dynamic range.\n")
    data_dyn_range = peak/rms
    return data_dyn_range
####
def get_image_rms(image,
            maskSup:float = 1e-7,
            sigma:int = 3.,
            noise_method:str = "Histogram Fit",
            residual_image = False,
            plotfile = 'Histofram_fit.png',
            plot_hist = True,
            clip = False,
           ):
    """
    Get rms from map

    Args:
        :maskSup: mask theshold
    """
    if noise_method == "Histogram Fit":
        print("Deriving noise using a Histogram Fit.\n")
        Z1 = image.flatten()
        sigma=10
        if clip:
            print("Clipping data to < rms*sigma")
            m = Z1[np.abs(Z1) > maskSup]
            rmsold = np.std(m)
            med = np.median(m)
            ind = np.where(np.abs(m-med) < rmsold*sigma)[0]
            Z1 = m[ind]
        Z1_2 = Z1 - np.min(Z1) + 10 ** (-5)
        bin_heights, bin_borders = np.histogram(Z1_2, bins=300)
        bin_widths = np.diff(bin_borders)
        bin_centers = bin_borders[:-1] + bin_widths / 2.
        bin_heights_err = np.where(bin_heights != 0, np.sqrt(bin_heights), 1)
        t_init = models.Gaussian1D(np.max(bin_heights), np.median(Z1_2), 0.001)
        fit_t = fitting.LevMarLSQFitter()
        t = fit_t(t_init, bin_centers, bin_heights, weights=1. / bin_heights_err)
        rms = t.stddev.value
        print("rms estimated. Now plotting the histogram.\n")

        # Plot
        if plot_hist:
            plt.figure(figsize=(8,5))
            plt.stairs(bin_heights, bin_borders,fill=True)
            plt.plot(bin_centers, t(bin_centers), 'r-', linewidth=2, label='Gaussian Fit')
            plt.xlabel('Pixel Value')
            plt.ylabel('Count')
            plt.title('Histogram and Gaussian Fit')
            # Fit parameters in box
            param_text = (
                f"Amplitude: {t.amplitude.value:.1f}\n"
                f"Mean: {t.mean.value:.4f}\n"
                f"Stddev: {t.stddev.value:.6f}"
            )
            plt.gca().text(0.95, 0.95, param_text, transform=plt.gca().transAxes,
                           fontsize=10, verticalalignment='top', horizontalalignment='right',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
            plt.legend(loc=7)
            plt.tight_layout()
            plt.savefig(plotfile, dpi=150)
            plt.show()
           # plt.close()
            print(f"Saved plot to {plotfile}")

    elif noise_method == "Image RMS":
        """
        Get the RMS (code from Cyril Rasse/kMS)

        Args:
            :param inp: FITS file
            :param maskSup: mask threshold
        """
        print("Deriving noise from the image by subtracting emission regions.\n")
        mIn = np.ndarray.flatten(image)
        m = mIn[np.abs(mIn) > maskSup]
        rmsold = np.std(m)
        diff = 1e-1
        med = np.median(m)

        for i in range(10):
            print("Loop {} to derive noise while extracting sources".format(i))
            ind = np.where(np.abs(m - med) < rmsold * sigma)[0]
            rms = np.std(m[ind])
            if np.abs((rms - rmsold) / rmsold) < diff:
                break
            rmsold = rms

    elif noise_method == "Residual":
        sys.stdout.write("Residual method for noise used\n")
        try:
            Z1 = residual_image.flatten()
            rms = np.nanstd(Z1)
           # levs1 = sigma * noise
        except:
            raise Exception("If using Residual-method for rms, please provide the residual_image arg.")
    else:
        raise  Exception("Please define valid noise method ('Histogram Fit','IMage RMS','Residual)")
    if noise_method=="box" or (noise_method=="Histogram Fit" and rms<=0):
        if (noise_method=="Histogram Fit" and rms<=0):
            sys.stdout.write("Could not do Histogram Fit for noise, will use 'box' method\n")
        rms = 1.8*np.std(image[0:round(len(image)/10),0:round(len(image[0])/10)]) #factor 1.8 from self-cal errors
        #levs1 = rms*sigma
    print("RMS calculated using method {} with value {}\n".format(noise_method,rms))
    return rms

def get_val_scores(fits_files,image_id=[],residual_files=[]):
    """
    Calculate scores for validation

    Args:
        image: input FITS images
    """

    # Get validation metrics
    with open ('validation_images.csv','w') as csvfile:
        fieldnames = ['Image_id', 'Peak_flux', 'Dyn_range', 'RMS', 'RMS_residual', 'score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        fitsF = glob(fits_files)
        try:
            resF = glob(residual_files)
            images = [ImageData(fits_file= fits,residual_file=res) for fits,res in zip(fitsF,resF)]
        except:
            sys.stdout.write("No residual loaded")
            images = [ImageData(fits_file= fits) for fits in fitsF]

        for i,image in enumerate(images):
            try:
                id = image_id[i]
            except:
                id = fitsF[i].split('/')[-1].replace('.fits','')
            print(id)
            rms_res = image.get_rms(noise_method="Histogram Fit")
            rms= image.get_rms(noise_method="Image RMS")
            peak = image.peakflux()
            dyn_range = image.dyn_range() #peak/rms (image)

            scores = {
                'Image_id' : id,
                'Peak_flux': peak,
                'Dyn_range': dyn_range,
                'RMS': rms,
                'RMS_residual': rms_res
            }
            pprint(scores)
            writer.writerow(scores)



