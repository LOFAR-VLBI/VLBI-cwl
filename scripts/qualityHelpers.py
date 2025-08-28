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
import pyregion



def calculate_expected_rms(ra,dec,obsdate,observation_time=8):
    # Define observer's latitude and longitude
    observer_lat = 52.9088  # latitude of LOFAR
    observer_lon = 6.8674   # longitude of LOFAR

    obs = {'integration':8,'num_subbands':231.0}
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
    duration_correction_factor = (obs['integration']*obs['num_subbands'])/(8.0*231.0) # Here its assuming an 8hr observation with 231 subbands (48MHz)
    #corrected_rms = meanrms * elevation_correction_factor * (flagging_correction_factor * duration_correction)
    correction_factor_rms = elevation_correction_factor * duration_correction_factor #(flagging_correction_factor * duration_correction)
    return correction_factor_rms

####
def image_quality(stats_matrix, expected_rms=75e-5, cut_peak=0.1, cut_rms=3):
    """
    Get image quality acceptance column

    Args:
        stats_matrix: dict with image-based scores
        :cut_peak: Peak of image
        :cut_rms: RMS of image
    """
    rms = stats_matrix["rms"]
    peak = stats_matrix["peak"]
    # Add this functions once we have useful values to input to calculate_expected_rms
#    if not expected_rms:
#        expected_rms = calculate_expected_rms(self.RA,self.DEC,self.Date,observation_time=8)
    rms_limit = expected_rms*cut_rms
    peak_limit = cut_peak
    valid = False
    valid = (rms <= rms_limit) and (peak >= peak_limit)
    stats_matrix ['rms_limit']= rms_limit
    stats_matrix ['peak_limit']= peak_limit
    stats_matrix ['valid']= valid

    print("Final diagnostics of inage:\n")
    print(stats_matrix)
    return stats_matrix

class ImageData(object):
    """ Load lofar image and get basic info
    """
    def __init__(self,
                 fits_file="",
                 residual_file="",
                 reg_file="",
                 catalogues=[],
                 noise_method="",
                 load_general_info="",
                ):
        self.fits_file = fits_file
        self.residual_file = residual_file
        self.reg_file= reg_file
        self.rms = None
        self.facets = None
        if fits_file!="":
            with fits.open(fits_file) as hdu:
                self.hdu_list = hdu
                self.image_data = hdu[0].data
                self.header = hdu[0].header
            try:
                self.Z = self.image_data[0, 0, :, :]
            except:
                self.Z = self.image_data

            self.id = fits_file.split('/')[-1].split('.')[0]
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

    def get_rms(self,sigma=10.,noise_method= "Histogram Fit", use_residual_img = True, plotfile='Global_histogram_fit.png',use_facet = None):
        """
        Get rms from map

        Args:
            :sigma: sigma for cutting the map
            :noise_method: Noise method: "Histogram Fit"...
            :use_residual_img: Set True if residual image should be used for histogram fit, False if total intensity image be used
            :plot_file: plot name for saving the histogram plot
        """
        if use_facet is not None:
            print("Calculating Facet rms")
            return get_image_rms(use_facet, noise_method=noise_method,plotfile=plotfile,clip=True,sigma=sigma)
             
        print("Calculating Image rms.")
        if use_residual_img and self.residual_file !="":
            print("Using residual image for estimating rms")
            self.rms = get_image_rms(self.residual_Z, noise_method=noise_method, residual_image=self.residual_Z,plotfile=plotfile,sigma=sigma)
        else:
            if self.residual_file =="":
                print("\n No residual data found")
            print("Using image for estimating rms.\n")
            self.rms = get_image_rms(self.Z, noise_method=noise_method, residual_image=self.residual_Z,plotfile=plotfile,clip=True,sigma=sigma)
            #self.rms = get_image_rms(self.Z, noise_method=noise_method, plotfile=plotfile,clip=True,sigma=sigma)

        return self.rms # jy/beam

    def get_statistics(self):
        """
        Get image statistics.
        Returns:
            :dict: with statistics
        """
        self.peak = get_peakflux(self.image_data)
        self.min = get_min(self.image_data)
        self.minmax = get_minmax(self.min,self.peak)
        if self.rms:
            self.dyn_range = get_dyn_range(self.peak,self.rms)
        else:
            self.rms = self.get_rms()
            self.dyn_range = get_dyn_range(self.peak,self.rms)
        stats_matrix ={
                'id': self.id,
                'rms': self.rms,
                'peak': self.peak,
                'dyn_range': self.dyn_range,
            }
        
        self.stats_matrix = image_quality(stats_matrix)
        return stats_matrix

    def get_facet_statistics(self):
        facet_ids = np.unique(self.facets)
        facet_ids = np.delete(facet_ids, facet_ids==-1)    #remove the default -1 value
        self.facet_stats_matrix = []
        for facet_id in facet_ids:
            mask = (self.facets == facet_id)
            facet_px = self.Z[mask]
            if np.count_nonzero(mask) == 0:
                continue #skip empty facets
            rms = self.get_rms(plotfile=f"{self.id}_facet_{facet_id}_histogram_fit.png", use_facet=facet_px)
            peak = get_peakflux(facet_px)
            dyn_range = get_dyn_range(peak,rms)

            facet_stats_matrix={
                'facet': facet_id,
                'rms': rms,
                'peak': peak,
                'dyn_range': dyn_range,
            }
            
            print(f"{facet_id}: stats: {facet_stats_matrix}")
            self.facet_stats_matrix.append(image_quality(facet_stats_matrix))
        return self.facet_stats_matrix

    def make_facets_from_reg(self, save_facets_im=True):
        shapes = pyregion.open(self.reg_file).as_imagecoord(self.header)
        facets = np.zeros((self.imagesize,self.imagesize), dtype=np.int32) - 1

        for n, shape in enumerate(shapes):
            mask = pyregion.ShapeList([shape]).get_mask(shape=(self.imagesize,self.imagesize))
            facets[mask.astype(bool)] = n
            print(f"masked for facet number:{n} of {len(shapes)}")

        self.facets = facets
        print("facets file created")

        if save_facets_im:
            plt.imsave(f"{self.id}_facets_map.png", facets)

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
def get_minmax(data_min,data_max):
    """
    Get min/max
    """
    sys.stdout.write("Deriving minmax.\n")
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
            sigma:int = 10.,
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
        if clip:
            m = Z1[np.abs(Z1) > maskSup]
            rmsold = np.std(m)
            med = np.median(m)
            print("Clipping data to < rms*sigma= {}".format(rmsold,sigma))
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
           # plt.show()
            plt.close()
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
        except:
            raise Exception("If using Residual-method for rms, please provide the residual_image arg.")
    else:
        raise  Exception("Please define valid noise method ('Histogram Fit','IMage RMS','Residual)")
    if noise_method=="box" or (noise_method=="Histogram Fit" and rms<=0):
        if (noise_method=="Histogram Fit" and rms<=0):
            sys.stdout.write("Could not do Histogram Fit for noise, will use 'box' method\n")
        rms = 1.8*np.std(image[0:round(len(image)/10),0:round(len(image[0])/10)]) #factor 1.8 from self-cal errors
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



