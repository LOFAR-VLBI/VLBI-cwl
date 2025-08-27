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

def image_quality(csv_table,cut_DR=10,cut_rms=0.01, cut_peak = 0.01):
    """
    Get image quality acceptance column

    Args:
        csv_table: CSV with image-based scores
        :cut_DR: Dynamic range of image
        :cut_rms: RMS of image
    """
    df = pd.read_csv(csv_table)
    df['accept_image'] = False

    # Filter for bad data
    mask = ~((df.Dyn_range > cut_DR) |
             (df.Peak_flux > cut_peak) |
             (df.RMS < cut_rms))
    df.loc[mask, 'accept_image'] = True
    df.to_csv(csv_table, index=False)
    return csv_table


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
                noise_method:str = "Image RMS",
               ):
        """
        Get rms from map

        Args:
            :maskSup: mask theshold
        """
        if self.residual_file !="":
            self.rms = get_image_rms(self.Z, maskSup=maskSup, noise_method=noise_method, residual_image=self.residual_Z)
        else:
            self.rms = get_image_rms(self.Z, maskSup=maskSup, noise_method=noise_method)


        return self.rms  # jy/beam

    def peakflux(self):
        self.data_max = get_peakflux(self.image_data)
        return self.data_max

    def min(self):
        self.data_min = get_min(self.image_data)

        return self.data_min

    def minmax(self):
        self.data_minmax = get_minmax(self.image_data)
        return self.data_minmax

    def dyn_range(self):
        self.data_dyn_range = get_dyn_range(self.image_data)

        return self.data_dyn_range


###########################

def get_peakflux(image):
    """
    Get peak intensity

    :return: peak intensity
    """
    sys.stdout.write("Deriving peakflux.\n")
    data_max = image.max()
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
def get_dyn_range(image):
    """
    Get the dynamic range.
    """
    print("Deriving dunamic range.\n")
    try:
        data_max = image.data_max
    except:
        data_max = get_peakflux(image)
    try:
        data_rms = image.rms
    except:
        data_rms = get_rms(image)
    data_dyn_range = data_max/data_rms
    return data_dyn_range
####
def get_image_rms(image,
            maskSup:float = 1e-7,
            sigma:int = 3.,
            noise_method:str = "Image RMS",
            residual_image = False,
           ):
    """
    Get rms from map

    Args:
        :maskSup: mask theshold
    """
    if noise_method == "Histogram Fit":
        print("Deriving noise using a Histogram Fit.\n")
        plotname = "histogram_fit.png"
        try:
            Z1 = image.flatten()
            bin_heights, bin_borders = np.histogram(Z1 - np.min(Z1) + 10 ** (-5), bins="auto")
            bin_widths = np.diff(bin_borders)
            bin_centers = bin_borders[:-1] + bin_widths / 2.
            bin_heights_err = np.where(bin_heights != 0, np.sqrt(bin_heights), 1)
            t_init = models.Gaussian1D(np.max(bin_heights), np.median(Z1 - np.min(Z1) + 10 ** (-5)), 0.001)
            fit_t = fitting.LevMarLSQFitter()
            t = fit_t(t_init, bin_centers, bin_heights, weights=1. / bin_heights_err)
            rms = t.stddev.value

            # Plot
            plt.figure(figsize=(8,5))
            plt.errorbar(bin_centers, bin_heights, yerr=bin_heights_err, fmt='o', label='Histogram')
            plt.plot(bin_centers, t(bin_centers), 'r-', label='Gaussian Fit')
            plt.xlabel('Pixel Value')
            plt.ylabel('Count')
            plt.title('Histogram and Gaussian Fit')
            # Fit parameters in box
            param_text = (
                f"Amplitude: {t.amplitude.value:.1f}\n"
                f"Mean: {t.mean.value:.2f}\n"
                f"Stddev: {t.stddev.value:.4f}"
            )
            plt.gca().text(0.95, 0.95, param_text, transform=plt.gca().transAxes,
                           fontsize=10, verticalalignment='top', horizontalalignment='right',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
            plt.legend()
            plt.tight_layout()
            plt.savefig(filename, dpi=150)
            plt.close()
            print(f"Saved plot to {filename}")
        except:
            rms=0

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
            rms_res = image.rms(noise_method="Residual")
            rms= image.rms(noise_method="Image RMS")
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



