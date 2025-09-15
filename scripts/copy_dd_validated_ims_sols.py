#!/usr/bin/env python3

__author__ = "Jurjen de Jong (jurjendejong@strw.leidenuniv.nl)"

import os
from argparse import ArgumentParser, Namespace

import pandas as pd
from numpy import sum


def merge_csv(validation_images_csv: str, validation_solutions_csv: str) -> pd.DataFrame:
    """
    Merge validation CSVs into one CSV

    Args:
        validation_images_csv: CSV with image validation information
        validation_solutions_csv: CSV with calibration solutions validation information

    Returns: The merged validation CSV
    """

    df1 = pd.read_csv(validation_images_csv)
    df2 = pd.read_csv(validation_solutions_csv)
    # Validate required column
    for name, df, path in [("validation_images_csv", df1, validation_images_csv),
                           ("validation_solutions_csv", df2, validation_solutions_csv)]:
        if "source_id" not in df.columns:
            raise ValueError(f"'source_id' column missing in {name} ({path})")
    merged_csv = pd.merge(df1, df2, on='source_id', how='inner')
    merged_csv.to_csv('validate.csv', index=False)
    return merged_csv


def copy_to_local_directory(df_val: pd.DataFrame, images: list[str], h5parms: list[str]):
    """
    Copy images and calibration solutions from calibrator sources with good quality to the current working directory.
    Good quality means that both accept_image and accept_solutions are set to True.
    Note: this overwrites any potential files in the current working directory with the same name.

    Args:
        df_val: CSV with validation information
        images: FITS images
        h5parms: Calibration solutions
    """

    for n, source in df_val.iterrows():
        for image in images:
            if source['source_id'] in image and source['accept_image'] and source['accept_solutions']:
                os.system(f'cp {image} select_{os.path.basename(image)}')
        for h5parm in h5parms:
            if source['source_id'] in h5parm and source['accept_image'] and source['accept_solutions']:
                os.system(f'cp {h5parm} select_{os.path.basename(h5parm)}')


def parse_args() -> Namespace:
    """
    Command line argument parser

    Return: parsed arguments
    """

    parser = ArgumentParser("Copy final images and calibration solutions after direction-dependent calibration.")
    parser.add_argument('--images', nargs='+', help='FITS images', default=None)
    parser.add_argument('--h5parms', nargs='+', help='Calibration solutions')
    parser.add_argument('--validation_images_csv', help='CSV with image validation information', default='validation_images.csv')
    parser.add_argument('--validation_solutions_csv', help='CSV with calibration solutions validation information', default='validation_solutions.csv')
    parser.add_argument('--copy_selected', action='store_true', help='Copy selected images to local directory')
    parser.add_argument('--error_on_bad_solutions', action='store_true', help='Return error if there are sources with bad solutions according to metrics')

    return parser.parse_args()


def main():
    args = parse_args()
    validation_csv = merge_csv(args.validation_images_csv, args.validation_solutions_csv)
    if args.copy_selected:
        copy_to_local_directory(validation_csv, args.images, args.h5parms)

    # Return an error if there are sources with bad solutions
    if args.error_on_bad_solutions and sum(~validation_csv[validation_csv['accept_image']]['accept_solutions'])>0:
        exit("ERROR: Following directions have bad solutions and should be inspected: \n"
             f"{'\n'.join(list(validation_csv[~validation_csv.accept_solutions].source_id))}")


if __name__ == '__main__':
    main()
