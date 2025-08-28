#!/usr/bin/env python
from argparse import ArgumentParser
import os
import json
from typing import Tuple

import numpy as np
from numpy import ndarray
from regions import Regions


def calculate_image_size(
    ras: ndarray, decs: ndarray, pixel_size: float, padding: float = 1.0
) -> Tuple[int, int, float]:
    """Calculates the image size needed to image a polygonal area delimited by the coordinates given,
    alongside the allowed baseline averaging factor for WSClean.

    Args:
        ras (ndarray): NumPy array of containing right ascensions of the polygon vertices.
        decs (ndarray): NumPy array of containing declinations of the polygon vertices.
        pixel_size (float): pixel size of the image in arcseconds.
        padding (float): padding factor by which to increase the image size.

    Returns:
        width, height, blavg (int, int, float): width, height and baseline averaging factor for the imaging.
    """
    width_ra = abs(max(ras) - min(ras))
    width_dec = max(decs) - min(decs)
    dec_centre = (min(decs) + max(decs)) / 2

    pix_size_deg = pixel_size / 3600
    imwidth, imheight = (
        width_ra * np.cos(np.deg2rad(dec_centre)) / pix_size_deg,
        width_dec / pix_size_deg,
    )

    imsize = max(imwidth, imheight)
    blavg = 1.5e3 * 60000.0 * 2.0 * 3.1415 * 1.5 / (24.0 * 60.0 * 60 * imsize)
    return int(padding * imwidth), int(padding * imheight), blavg


def main():
    parser = ArgumentParser(
        description="Estimate the required image size based on a given pixel size and DS9 region file."
    )
    parser.add_argument(
        "--region", type=str, help="DS9 region file describing the facet."
    )
    parser.add_argument(
        "--pixel_size", type=float, help="Pixel size of the image to be made."
    )
    parser.add_argument(
        "--resolution",
        type=str,
        help="Angular resolution that will be passed to WSClean. Used for naming the output image.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        help="Padding factor to pad the calculated image size with. This allows some extra freedom in tweaking the final image size.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        help="Name of the ouput file that will contain JSON with the image name, width, height and baseline averaging factor.",
        default="cwl.output.json",
    )
    args = parser.parse_args()

    reg = Regions.read(args.region, format="ds9")
    ra = reg[0].vertices.ra.value
    dec = reg[0].vertices.dec.value
    width, height, blavg = calculate_image_size(ra, dec, args.pixel_size, args.padding)
    fits_name = (
        os.path.basename(args.region).replace(".reg", "") + f"_{args.resolution}"
    )
    with open(args.filename, "w") as f:
        json.dump(
            {"name": fits_name, "image_size": [width, height], "blavg": blavg}, f
        )


if __name__ == "__main__":
    main()
