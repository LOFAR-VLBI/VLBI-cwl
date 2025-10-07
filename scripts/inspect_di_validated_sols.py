#!/usr/bin/env python3

__author__ = "Jurjen de Jong (jurjendejong@strw.leidenuniv.nl)"

from argparse import ArgumentParser

import pandas as pd
from numpy import sum


def parse_args():
    """
    Command line argument parser

    :return: parsed arguments
    """

    parser = ArgumentParser("Inspect direction-independent calibration solution validation.")
    parser.add_argument('--h5parms', nargs='+', help='Calibration solutions')
    parser.add_argument('--validation_solutions_csv', help='CSV with calibration solutions validation information', default='validation_solutions.csv')
    parser.add_argument('--return_error', action='store_true', help='Return error if there are sources with bad solutions according to metrics')

    return parser.parse_args()


def main():
    args = parse_args()
    validation_csv = pd.read_csv(args.validation_solutions_csv)

    # Return an error if there are sources with bad solutions
    if args.return_error and sum(~validation_csv['accept_solutions'])>0:
        exit(f"ERROR: Following directions have bad solutions and should be inspected: \n"
             f"{'\n'.join(list(validation_csv[~validation_csv.accept_solutions].Source_id))}")


if __name__ == '__main__':
    main()
