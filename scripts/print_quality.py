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
    dr = n/20   # a function can be added todo
    print(f"image length is {n} pixels")
    if center is None:
        cx = cy = (n - 1) / 2.0
    else:
        cx, cy = center
    
    print(f"image center is at ({cx,cy})")

    # distance map
    y, x = np.indices((n, n))
    r = np.hypot(x - cx, y - cy)

    r_max = n / 2.0
    edges = np.arange(0, r_max + dr, dr)
    mids = 0.5 * (edges[:-1] + edges[1:])
    
    print(f"mids: {mids}")

    rms_vals = []
    for r_in, r_out in zip(edges[:-1], edges[1:]):
        mask = (r >= r_in) & (r < r_out)
        vals = image2d[mask]
        if vals.size > 0:
            rms_vals.append(np.sqrt(np.mean(vals**2)))
        else:
            rms_vals.append(np.nan)
    
    print(f"rms: {np.array(rms_vals)}")
    return mids, np.array(rms_vals), dr


def plot_radial_rms(image2d, pixelscale, center=None, outfile="radial_rms.png"):
    radii, rms, dr = radial_rms_square(image2d, center=center)
    radii_arcsec = np.array(radii)*pixelscale
    plt.figure()
    plt.plot(radii_arcsec, rms, "o-")
    plt.xlabel("Radius (deg)")
    plt.ylabel("RMS (Jy/Beam)")
    plt.title(f"Radial RMS profile (dR={np.round(dr*pixelscale*3600)} arcsec)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)   # save as PNG
    plt.close()                     # close to avoid extra plots in notebooks
    print(f"Saved radial RMS plot as {outfile}")
    return radii, rms

############
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
    mask = ~((df.Dyn_range < cut_DR) |
             (df.Peak_flux < cut_peak) |
             (df.RMS > cut_rms))
    df.loc[mask, 'accept_image'] = True
    df.to_csv(csv_table, index=False)
    return csv_table

############
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
    
    print(f"redsidual data: {image.residual_Z.shape}")

    r, rms = plot_radial_rms(image.residual_Z, image.pixelscale)

if __name__ == "__main__":
    sys.exit(main())
