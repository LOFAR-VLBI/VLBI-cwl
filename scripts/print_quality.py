from helpers import *
import argparse
import sys
import csv
from astropy.io import fits
import numpy as np
import os
from pathlib import Path

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
                try:
                    self.Z = self.image_data[0, 0, :, :]
                except:
                    self.Z = self.image_data
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
        rms = get_image_rms(self.Z, maskSup=maskSup, noise_method=noise_method, noise=noise, residual_image=self.residual_Z)

        return rms  # jy/beam

    def get_peakflux(self):
        """
        Get peak intensity

        :return: peak intensity
        """

        data = self.image_data
        return data.max()

    def get_minmax(self):
        """get min/max"""
        data = self.image_data
        return np.abs(data.min() / data.max())


def main():
    ap = argparse.ArgumentParser(
        description="Compute min/max, peak, and RMS (Image RMS / Residual / Histogram Fit) from FITS images." )
    ap.add_argument("image_fits", help="Primary (PB-corrected) image FITS file")
    ap.add_argument("--residual-fits", help="Residual image FITS file (optional, enables 'Residual' RMS)", default=None)
    ap.add_argument("--out-csv", help="name of output csv file")
    args = ap.parse_args()
    out_csv = args.out_csv

    # Build constructor args for your class
    kwargs = {"fits_file": args.image_fits}
    if args.residual_fits:
        kwargs["residual_file"] = args.residual_fits

    image = ImageData(**kwargs)

    # Min/Max and Peak
    img_minmax = image.get_minmax()
    peak_flux  = image.get_peakflux()

    # RMS computations
    rms_image = image.get_rms(noise_method="Image RMS")
    rms_hist  = image.get_rms(noise_method="Histogram Fit")

    rms_res = None
    if args.residual_fits:
        rms_res = image.get_rms(noise_method="Residual")
    
    # Print results
    print(f"\n== Global Quality Metrics for: {args.image_fits} ==")
    print(f"Min / Max: {img_minmax}")
    print(f"Peak flux: {peak_flux}")

    print("\nRMS estimates:")
    print(f"  Image RMS:       {rms_image}")
    if args.residual_fits:
        print(f"  Residual RMS:    {rms_res}")
    else:
        print("  Residual RMS:    (skipped; no --residual-fits provided)")
    print(f"  Histogram Fit:   {rms_hist}")

    # Dynamic ranges
    print("\nDynamic range (Peak / RMS):")
    print(f"  DR (Image RMS):  {peak_flux/rms_image}")
    if args.residual_fits:
        print(f"  DR (Residual):   {peak_flux/rms_res}")
    print(f"  DR (Hist Fit):   {peak_flux/rms_hist}\n")

    dr_valueImage = (peak_flux / rms_image) if (rms_image and np.isfinite(rms_image)) else float("nan")
    dr_valueResidual = (peak_flux / rms_res) if (rms_res and np.isfinite(rms_res)) else float("nan")
    name="Global"

    row = {
        "Name": name,
        "Peak": peak_flux,
        "RMS_Image": rms_image,
        "RMS_residual": (rms_res if rms_res is not None else ""),
        "DR_Image": dr_valueImage,
        "DR_Residul": dr_valueResidual
    }
    
    header = ["Name", "Peak", "RMS_Image", "RMS_residual", "DR_Image", "DR_Residul"]
    write_header = not Path(out_csv).exists()
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow(row)

if __name__ == "__main__":
    sys.exit(main())
