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
            rms_vals.append(np.sqrt(np.mean(vals**2))) #todo: change this rmsvalue calculation to call Anne's method instead
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
    
    #First Saving Global Statistics 
    global_statistics = {"Name": "Global"}
    global_statistics.update(image.get_statistics())

    header = global_statistics.keys()
    write_header = not Path(out_csv).exists()
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow(global_statistics)

    #plot rms as a function of distance from phase center. RMS calculated in annular regions
    r, rms = plot_radial_rms(image.residual_Z, image.pixelscale)

if __name__ == "__main__":
    sys.exit(main())
