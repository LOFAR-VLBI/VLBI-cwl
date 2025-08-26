from astropy.io import fits
import numpy as np
import os
import sys
from glob import glob
import pandas as pd
import csv
from pprint import pprint
from glob import glob

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
        if fits_file!="":
            with fits.open(fits_file) as hdu:
                self.hdu_list = hdu
                self.image_data = hdu[0].data
                self.header = hdu[0].header
            try:
                self.Z = self.image_data[0, 0, :, :]
            except:
                self.Z = self.image_data
    
            self.pixelscale = self.header["CDELT1"] #in deg
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
                noise:float = 0,
               ):
        """
        Get rms from map

        Args:
            :maskSup: mask theshold
        """
        self.rms = get_image_rms(self.Z, maskSup=maskSup, noise_method=noise_method, noise=noise, residual_image=self.residual_Z)

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
    
    def _default_center(self, img2d):
        """
        Determine phase center in pixel coordinates (0-based).
        Uses FITS CRPIX if available (converting from 1-based to 0-based),
        otherwise image geometric center.
        """
        h = getattr(self, "header", {}) or {}
        cx = h.get("CRPIX1", None)
        cy = h.get("CRPIX2", None)
        if cx is not None and cy is not None:
            # FITS CRPIX is 1-based; convert to 0-based pixel index
            cx = float(cx) - 1.0
            cy = float(cy) - 1.0
        else:
            # geometric center
            ny, nx = img2d.shape
            cx = (nx - 1) / 2.0
            cy = (ny - 1) / 2.0
        return (cx, cy)

    def radial_rms(self, img2d=None, dr=20, center=None, r_max=None, nan_policy="omit"):
        """
        Compute RMS in concentric annuli [R, R+dr) from the phase center.
        Parameters
        ----------
        img2d : 2D np.ndarray
            Image to analyze. Defaults to self.residual_image if present, else self.Z.
        dr : int
            Annulus thickness in pixels.
        center : (cx, cy) or None
            Center in pixel coords (x, y). If None, uses CRPIX (converted) or image center.
        r_max : float or None
            Maximum radius in pixels. If None, uses min(image_dim)/2.
        nan_policy : {"omit", "propagate"}
            If "omit", ignore NaNs when computing RMS; if "propagate", return NaN when any are present.
        Returns
        -------
        radii : 1D np.ndarray
            Annulus mid-point radii in pixels.
        rms : 1D np.ndarray
            RMS in each annulus.
        counts : 1D np.ndarray
            Number of pixels contributing to each annulus.
        """
        # pick the image
        if img2d is None:
            img2d = getattr(self, "residual_image", None)
            if img2d is None:
                img2d = self.Z

        ny, nx = img2d.shape
        if center is None:
            cx, cy = self._default_center(img2d)
        else:
            cx, cy = center  # (x, y)

        if r_max is None:
            r_max = min(nx, ny) / 2.0

        # radius map (in pixels)
        y, x = np.indices((ny, nx))
        r = np.hypot(x - cx, y - cy)

        # bin edges and midpoints
        edges = np.arange(0, r_max + dr, dr, dtype=float)
        mids = 0.5 * (edges[:-1] + edges[1:])

        rms_vals = np.full(mids.shape, np.nan, dtype=float)
        counts = np.zeros_like(mids, dtype=int)

        for i, (r_in, r_out) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (r >= r_in) & (r < r_out)
            vals = img2d[mask]
            if nan_policy == "omit":
                vals = vals[np.isfinite(vals)]
            elif not np.all(np.isfinite(vals)):
                # propagate NaN/inf
                rms_vals[i] = np.nan
                counts[i] = vals.size
                continue

            counts[i] = vals.size
            if vals.size > 0:
                # RMS = sqrt(mean(x^2))
                rms_vals[i] = np.sqrt(np.mean(vals**2))
            else:
                rms_vals[i] = np.nan

        return mids, rms_vals, counts

    def plot_radial_rms(self, img2d=None, dr=20, center=None, r_max=None, nan_policy="omit", show_counts=False):
        """
        Convenience wrapper: compute and plot RMS vs radius (pixels).
        """
        radii, rms, counts = self.radial_rms(img2d=img2d, dr=dr, center=center, r_max=r_max, nan_policy=nan_policy)
        plt.figure()
        plt.plot(radii, rms, marker="o", lw=1)
        plt.xlabel("Radius (pixels)")
        plt.ylabel("RMS")
        plt.title(f"Radial RMS profile (dr={dr} px)")
        plt.grid(True, alpha=0.3)
        if show_counts:
            # annotate a few bins with counts to judge statistics
            for x, y, n in zip(radii[::max(1, len(radii)//10)], rms[::max(1, len(radii)//10)], counts[::max(1, len(radii)//10)]):
                if np.isfinite(y):
                    plt.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(5,5), fontsize=8)
        plt.tight_layout()
        return radii, rms, counts
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
    sys.stdout.write("Deriving dunamic range.\n")
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
            noise:float = 0,
            residual_image = False,
           ):
    """
    Get rms from map

    Args:
        :maskSup: mask theshold
    """
    #image = image
    if noise_method == "Histogram Fit":
        try:
            Z1 = image.flatten()
            bin_heights, bin_borders = np.histogram(Z1 - np.min(Z1) + 10 ** (-5), bins="auto")
            bin_widths = np.diff(bin_borders)
            bin_centers = bin_borders[:-1] + bin_widths / 2.
            bin_heights_err = np.where(bin_heights != 0, np.sqrt(bin_heights), 1)

            t_init = models.Gaussian1D(np.max(bin_heights), np.median(Z1 - np.min(Z1) + 10 ** (-5)), 0.001)
            fit_t = fitting.LevMarLSQFitter()
            t = fit_t(t_init, bin_centers, bin_heights, weights=1. / bin_heights_err)
            noise = t.stddev.value

            # Set contourlevels to mean value + 3 * rms_noise * 2 ** x
            rms = t.mean.value + np.min(Z1) - 10 ** (-5) + sigma * noise

        except:
            rms=0

    elif noise_method == "Image RMS":
        """
        Get the RMS (code from Cyril Rasse/kMS)

        Args:
            :param inp: FITS file
            :param maskSup: mask threshold
        """
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

    elif noise_method == "Residual" or (noise_method=="Histogram Fit" and rms>=0):
        try:
            Z1 = residual_image.flatten()
            noise = np.nanstd(Z1)
            rms = sigma * noise
        except:
            raise Exception("If using Residual-method for rms, please provide the residual_image arg.")
    else:
        raise  Exception("Please define valid noise method ('Histogram Fit','IMage RMS','Residual)")
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



