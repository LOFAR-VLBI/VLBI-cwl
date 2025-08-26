from qualityHelpers import *
import argparse
import sys
import csv
from astropy.io import fits
import numpy as np
import os
from pathlib import Path

class ImageData(object):
    """Load LOFAR image and get basic info"""
    def __init__(self,
                 fits_file="",
                 residual_file="",
                 catalogues=[],
                 noise_method="",
                 load_general_info="",
                ):
        if fits_file!="":
            from astropy.io import fits
            with fits.open(fits_file) as hdu:
                self.hdu_list = hdu
                self.image_data = hdu[0].data
                self.header = hdu[0].header
            try:
                self.Z = self.image_data[0, 0, :, :]
            except Exception:
                self.Z = self.image_data

            self.pixelscale = self.header.get("CDELT1", np.nan)  # deg/pix
            self.imagesize = int(self.header.get("NAXIS1", self.Z.shape[1]))
            self.RA = self.header.get("CRVAL1", np.nan)
            self.DEC = self.header.get("CRVAL2", np.nan)
        else:
            import sys
            sys.stdout.write("Provide fits file\n")
            raise SystemExit

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

    r_pix, rms, n = image.plot_radial_rms(img2d=image.residual_image, dr=20)

if __name__ == "__main__":
    sys.exit(main())
