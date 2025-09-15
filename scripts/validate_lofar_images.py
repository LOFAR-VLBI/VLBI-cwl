#!/usr/bin/env python3

__author__ = "Jurjen de Jong (jurjendejong@strw.leidenuniv.nl)"

import csv
from argparse import ArgumentParser, Namespace
from pprint import pprint

import numpy as np
import pandas as pd
from astropy.io import fits

from make_config_international import parse_source_id
from submods.source_selection.image_score import get_nn_model, predict_nn  # from lofar_facet_selfcal


def get_rms(inp: str, maskSup: float = 1e-7) -> float:
    """
    Get the RMS of a FITS image (adapted from DDF code by Cyril Tasse/kMS).

    Args:
        inp: input FITS file
        maskSup: mask threshold

    Returns:
        RMS value of the image
    """

    with fits.open(inp) as hdul:
        data = hdul[0].data

    mIn = np.ndarray.flatten(data)
    m = mIn[np.abs(mIn) > maskSup]
    rmsold = np.std(m)
    diff = 1e-1
    cut = 3.
    med = np.median(m)

    for i in range(10):
        ind = np.where(np.abs(m - med) < rmsold * cut)[0]
        rms = np.std(m[ind])
        if np.abs((rms - rmsold) / rmsold) < diff:
            break
        rmsold = rms

    return rms  # jy/beam


def get_peakflux(inp: str) -> float:
    """
    Get the peak intensity of a FITS image.

    Args:
        inp: input FITS file

    Returns:
        Peak intensity value
    """

    with fits.open(inp) as hdul:
        data = hdul[0].data

    return np.nanmax(data)


def get_minmax(inp: str) -> float:
    """
    Get the minimum-to-maximum pixel ratio of a FITS image.

    Args:
        inp: input FITS file

    Returns:
        Ratio of minimum to maximum pixel value
    """

    with fits.open(inp) as hdul:
        data = hdul[0].data

    return np.abs(np.nanmin(data) / np.nanmax(data))


def get_validation_scores(images: list[str], model: str, model_cache: str):
    """
    Get peak flux, dynamic range, RMS, and neural network score and write to CSV.

    Args:
        images: input FITS images
        model: neural network model name
        model_cache: local cache for model
    """

    # Get neural network model
    nn_model = get_nn_model(model=model, cache=model_cache)

    # Get validation metrics
    with open('validation_images.csv', 'w') as csvfile:
        fieldnames = ['source_id', 'Peak_flux', 'Dyn_range', 'RMS', "NN_score"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Loop over FITS images
        for image in images:
            id = parse_source_id(image)
            if not id:
                id = image.replace('.fits','')
            print(id)
            minmax = get_minmax(image)
            rms = get_rms(image)
            peak = get_peakflux(image)
            if nn_model is not None:
                nn_score = predict_nn(image, nn_model)
            else:
                nn_score = 999.

            scores = {
                'source_id': id,
                'Peak_flux': peak,
                'Dyn_range': minmax,
                'RMS': rms,
                'NN_score': nn_score
            }
            pprint(scores)
            writer.writerow(scores)


def image_quality(csv_table: str):
    """
    Make image quality acceptance column based on a combination of image quality criteria

    Args:
        csv_table: CSV with image-based scores
    """

    df = pd.read_csv(csv_table)
    df['accept_image'] = False

    # Filter for weak self-calibration sources
    mask = ~((df.Dyn_range > 0.04) |
                   ((df.NN_score>0.5) & (df.Peak_flux<0.03)) |
                   ((df.Dyn_range/df.Peak_flux>1) & (df.Peak_flux<0.01)) |
                   (df.Dyn_range/df.Peak_flux>5) |
                   (df.Peak_flux/df.RMS<100))

    # Flag sources with accept_solutions=False if scores below threshold
    df.loc[mask, 'accept_image'] = True

    df.to_csv(csv_table, index=False)


def parse_args() -> Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments
    """

    parser = ArgumentParser("Get validation scores for images (currently only for DD calibration).")
    parser.add_argument('images', nargs='+', help='FITS images', default=None)
    parser.add_argument('--nn_model', help='Neural network model name', default='surf/dino_big_lora_tune_posclsreg_may_O2_aug_099')
    parser.add_argument('--nn_model_cache', help='Cache folder with neural network model', default='.cache/cortexchange')

    return parser.parse_args()


def main():
    args = parse_args()
    get_validation_scores(args.images, model=args.nn_model, model_cache=args.nn_model_cache)
    image_quality("validation_images.csv")


if __name__ == '__main__':
    main()