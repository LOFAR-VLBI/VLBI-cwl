#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Jurjen de Jong"

from argparse import ArgumentParser

from numpy import pi
import pandas as pd
from pandas.errors import ParserError

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table as astropy_table
from casacore.tables import table as casacore_table


def read_catalogue(file_path):
    """
    Read catalogue regardless if it is a CSV or FITS table

    Args:
        file_path: The path to the file to check.

    Returns: catalogue
    """
    try:
        return pd.read_csv(file_path)
    except (ParserError, UnicodeDecodeError):
        return astropy_table.read(file_path).to_pandas()


def ra_dec_to_iltj(ra_deg, dec_deg):
    """
    Convert RA/DEC floats to ILTJ source name format: ILTJhhmmss.ss±ddmmss.s

    Args:
        ra_deg (float): Right Ascension in degrees
        dec_deg (float): Declination in degrees

    Returns:
        str: Source name in ILTJhhmmss.ss±ddmmss.s format
    """

    coord = SkyCoord(ra=ra_deg * u.degree, dec=dec_deg * u.degree, frame='icrs')

    ra_h = int(coord.ra.hms.h)       # Hours component
    ra_m = int(coord.ra.hms.m)       # Minutes component
    ra_s = coord.ra.hms.s            # Seconds component

    sign = '+' if coord.dec.deg >= 0 else '-'  # Sign character
    dec_d = int(abs(coord.dec.dms.d))  # Degrees component
    dec_m = int(coord.dec.dms.m)       # Minutes component
    dec_s = coord.dec.dms.s            # Seconds component

    # Build the formatted ILTJ string
    source_name = (
        f"ILTJ"
        f"{ra_h:02d}{ra_m:02d}{ra_s:05.2f}"
        f"{sign}{dec_d:02d}{dec_m:02d}{dec_s:04.1f}"
    )
    return source_name


def get_phase_centre(ms):
    """
    Get phase centre from MeasurementSet in degrees

    Args:
        ms: MeasurementSet

    Returns: Phase centre
    """
    t = casacore_table(ms + '::FIELD', ack=False)
    phasedir = t.getcol("PHASE_DIR").squeeze()
    phasedir *= 180 / pi
    phasedir_coor = SkyCoord(ra=phasedir[0] * u.degree, dec=phasedir[1] * u.degree, frame='fk5')
    return phasedir_coor


def select_bright_sources(phase_centre, catalogue, fluxcut, fov=2.5):
    """
    Produces a data frame of sources collected from catalogue which are within a box of size `fov` degrees
    centred on phase_centre, and have a flux density exceeding `fluxcut`.

    Args:
        phase_centre: SkyCoord object (RA/Dec in degrees)
        catalogue: Catalogue file name
        fluxcut: Peak flux density cut (Jy/beam)
        fov: Field-of-view in degrees

    Returns:
        df: DataFrame with selected sources
    """

    df = read_catalogue(catalogue)
    df = df[df['Peak_flux'] > fluxcut]

    # Convert source coordinates
    sourcedir_x = SkyCoord(ra=phase_centre.ra.value * u.deg, dec=df['DEC'].values * u.deg)
    sourcedir_y = SkyCoord(ra=df['RA'].values * u.deg, dec=phase_centre.dec.value * u.deg)

    # Get separation
    dra, ddec = (phase_centre.separation(sourcedir_x.transform_to('fk5')),
                 phase_centre.separation(sourcedir_y.transform_to('fk5')))
    ra_diff = abs(dra.value)
    dec_diff = abs(ddec.value)

    # Apply box cut
    half_fov = fov / 2.0
    mask = (ra_diff < half_fov) & (dec_diff < half_fov)
    df = df[mask]

    # Generate source IDs
    df['Source_id'] = [ra_dec_to_iltj(ra, dec) for ra, dec in zip(df['RA'], df['DEC'])]

    return df[['Source_id', 'RA', 'DEC', 'Peak_flux', 'Total_flux']]


def argparse():
    """
    Argument parser
    """
    parser = ArgumentParser("Pre-select sources based on flux density.")
    parser.add_argument('--ms', type=str, help='MeasurementSet to read phase centre from.')
    parser.add_argument('--catalogue', type=str, help='Catalog to select candidate calibrators from (FITS or CSV format).')
    parser.add_argument('--fluxcut', type=float, help='Minimal peak flux density in Jy/beam', default=0.0)
    parser.add_argument('--fov', type=float, help='Field-of-view size in degrees', default=2.5)
    return parser.parse_args()


def main():
    args = argparse()

    phase_centre = get_phase_centre(args.ms)
    out_df = select_bright_sources(phase_centre, args.catalogue, args.fluxcut, args.fov)
    out_df.to_csv("bright_cat.csv", index=False)


if __name__ == '__main__':
    main()
