#!/usr/bin/env python3

import argparse
import json

from astropy.table import Table


def plugin_main(**kwargs):
    """
    Takes in a catalogue with a target and returns appropriate coordinates

    Parameters
    ----------
    target_file: str
        Name of the file containing the target info
    mode: str
        The name of the processing mode. must be either
        'delay_calibration' or 'split_directions'

    Returns
    -------
    result : dict
        Output coordinates
    """

    # parse the input
    target_file = kwargs["target_file"]
    mode = kwargs["mode"]

    # read in the catalogue to get source_id, RA, and DEC
    t = Table.read(target_file, format="csv")
    RA_val = t["RA"].data
    DEC_val = t["DEC"].data
    Source_id = t["Source_id"].data
    if mode == "delay_calibration":
        RA_val = [RA_val[0]]
        DEC_val = [DEC_val[0]]
        Source_id = Source_id[0]
        if isinstance(Source_id, str):
            Source_id = [Source_id]
        else:
            Source_id = ["S" + str(Source_id)]

    # make a string of coordinates for the DP3 command
    ss = ["[" + str(x) + "deg," + str(y) + "deg]" for x, y in zip(RA_val, DEC_val)]
    if mode == "delay_calibration":
        ss = ss[0]
    elif mode == "split_directions":
        ss = "[" + (ss[0] if len(ss) == 1 else ",".join(ss)) + "]"
    else:
        raise ValueError(
            "Argument mode must be one of"
            + ' "delay_calibration", "split_directions"'
            + f' but was "{mode}".'
        )

    return {"name": ",".join(Source_id), "coords": ss}


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target_file",
        help="The path of the file containing the properties and coordinates of the delay calibrator",
        type=str,
    )
    parser.add_argument(
        "--mode",
        help="The name of the processing mode",
        choices=["delay_calibration", "split_directions"],
        type=str,
    )

    return parser.parse_args()


def main():

    arguments = parse_arguments()

    output = plugin_main(
        target_file=arguments.target_file,
        mode=arguments.mode,
    )

    with open("./out.json", "w") as fp:
        json.dump(output, fp)


if __name__ == "__main__":
    main()
