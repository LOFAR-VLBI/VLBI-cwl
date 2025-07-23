#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Jurjen de Jong"

import os
import re
import sys
from argparse import ArgumentParser

import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u


def filter_too_nearest_neighbours(csv: str = None, sep: float = 0.1):
    """
    Identify sources that have a nearest neighbour within 0.1 degrees distance.
    Keep the source with the highest spd_score.

    Args:
        csv: CSV file with RA/DEC and spd_score
        sep: separation threshold in degrees

    Returns:

    """

    df = pd.read_csv(csv)

    # SkyCoord objects for all sources
    coords = SkyCoord(ra=df['RA'].values * u.deg, dec=df['DEC'].values * u.deg)

    # Entries collection to remove
    to_remove = set()

    for i, row in df.iterrows():
        if i in to_remove:
            continue  # Skip already marked rows

        # Separations between the current source and all others
        separations = coords[i].separation(coords).degree

        # Sources within seperation threshold
        close_sources = (separations < sep) & (separations > 0)
        close_indices = df[close_sources].index

        # If there are neighbouring sources, compare the spd_scores
        for j in close_indices:
            if df.loc[j, 'spd_score'] < row['spd_score']:
                # Remove the current source if its spd_score is lower
                to_remove.add(i)

    # Remove rows marked for removal
    print('Removing:')
    print(df.iloc[list(to_remove)])
    filtered_df = df.drop(index=to_remove).reset_index(drop=True)

    return filtered_df


def parse_source_id(inp_str: str = None):
    """
    Parse ILTJ... source_id string

    Args:
        inp_str: ILTJ source_id

    Returns: parsed output

    """

    parsed_inp = re.findall(r'ILTJ\d+\..\d+\+\d+.\d+', inp_str)[0]

    return parsed_inp


def match_source_id(mslist: list = None, source_id: str = None):
    """
    Return MS name by matching source ID with items from list with MS names

    Parameters
    ----------
    mslist: List with MS
    source_id: Source ID

    Returns
    -------
    Most similar element
    """

    for ms in mslist:
        if parse_source_id(ms) in source_id:
            return ms

    # If no match (should not arrive here)
    sys.exit(f"ERROR: No matching MS for {source_id}")


def rename_folder(old_name, new_name):
    """
    Rename folder name

    Parameters
    ----------
    old_name: Old name
    new_name: New name
    """

    try:
        os.rename(old_name, new_name)
        print(f"Folder renamed from '{old_name}' to '{new_name}'")
    except OSError as e:
        print(f"Error: {e}")


def parse_args():
    """
    Command line argument parser

    Returns
    -------
    Parsed arguments
    """

    parser = ArgumentParser()
    parser.add_argument('--csv', help='CSV with names and phasediff scores', default=None)
    parser.add_argument('--ms', nargs="+", help='Input MS', default=None)
    parser.add_argument('--best_score', type=float,
                        help='Optimal selection score (See Section 3.3.1 https://arxiv.org/pdf/2407.13247)',
                        default=2.3)
    parser.add_argument('--suffix', help='suffix', default='_best')
    return parser.parse_args()


def main():
    """
    Main function
    """

    args = parse_args()

    # Get dataframe after filtering for sources within 0.1 degrees distance from each other
    df = filter_too_nearest_neighbours(args.csv)

    # Sort values
    df = df.sort_values("spd_score", ascending=True)

    for source in df.set_index('source').iterrows():
        name = source[0]
        score = source[1]['spd_score']
        if score < args.best_score:
            ms_name = match_source_id(args.ms, name)

            # Rename folder to return best directions in CWL workflow
            rename_folder(ms_name, ms_name.split('/')[-1]+args.suffix+'.ms')


if __name__ == '__main__':
    main()
