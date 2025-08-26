from qualityHelpers import *
import argparse
import sys
import csv
from astropy.io import fits
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt 

def radial_rms_square(image2d, dr=20, center=None):
    """
    Compute RMS as a function of radius in concentric annuli.
    
    Parameters
    ----------
    image2d : 2D np.ndarray
        Square image array.
    dr : int
        Annulus thickness in pixels.
    center : (cx, cy) or None
        Phase center in pixel coordinates. If None, uses the geometric center.
    
    Returns
    -------
    radii : np.ndarray
        Midpoint radii of annuli in pixels.
    rms : np.ndarray
        RMS in each annulus.
    """
    n = image2d.shape[0]   # since square, nx = ny = n
    if center is None:
        cx = cy = (n - 1) / 2.0
    else:
        cx, cy = center

    # distance map
    y, x = np.indices((n, n))
    r = np.hypot(x - cx, y - cy)

    r_max = n / 2.0
    edges = np.arange(0, r_max + dr, dr)
    mids = 0.5 * (edges[:-1] + edges[1:])

    rms_vals = []
    for r_in, r_out in zip(edges[:-1], edges[1:]):
        mask = (r >= r_in) & (r < r_out)
        vals = image2d[mask]
        if vals.size > 0:
            rms_vals.append(np.sqrt(np.mean(vals**2)))
        else:
            rms_vals.append(np.nan)

    return mids, np.array(rms_vals)


def plot_radial_rms(image2d, dr=20, center=None):
    radii, rms = radial_rms_square(image2d, dr=dr, center=center)
    plt.figure()
    plt.plot(radii, rms, "o-")
    plt.xlabel("Radius (pixels)")
    plt.ylabel("RMS")
    plt.title(f"Radial RMS profile (dr={dr} px)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return radii, rms


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
    peak_flux  = image.peakflux()

    # RMS computations
    rms_image = image.get_rms(noise_method="Image RMS")

    rms_res = None
    if args.residual_fits:
        rms_res = image.get_rms(noise_method="Residual")
    
    # Print results
    print(f"\n== Global Quality Metrics for: {args.image_fits} ==")
    print(f"Peak flux: {peak_flux}")

    print("\nRMS estimates:")
    print(f"  Image RMS:       {rms_image}")
    if args.residual_fits:
        print(f"  Residual RMS:    {rms_res}")
    else:
        print("  Residual RMS:    (skipped; no --residual-fits provided)")

    # Dynamic ranges
    print("\nDynamic range (Peak / RMS):")
    print(f"  DR (Image RMS):  {peak_flux/rms_image}")
    if args.residual_fits:
        print(f"  DR (Residual):   {peak_flux/rms_res}")

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

    r_pix, rms, n = image.plot_radial_rms(img2d=image.residual_data, dr=20)

if __name__ == "__main__":
    sys.exit(main())
