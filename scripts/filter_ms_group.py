#!/usr/bin/env python3

"""
Selects MeasurementSets from a pre-determined frequency
group and writes the result to a JSON file
"""

import argparse
import json
import os
import sys
import warnings


def main():

    arguments = parse_arguments()
    ms_list = arguments.mss
    group_id = arguments.group_id
    json_file = arguments.json_file
    output_file = arguments.output_filename

    ms_by_name = {
        os.path.basename(ms): {"class": "Directory", "path": ms} for ms in ms_list
    }

    with open(json_file, "r") as f_stream:
        try:
            selected_ms = json.load(f_stream)[group_id]
        except KeyError as e:
            print(f"Group ID {group_id} is not a valid frequency group in {json_file}")
            sys.exit(1)

    selected_ms = [os.path.basename(ms_name) for ms_name in selected_ms]
    selected_ms_names = [
        ms_by_name[ms_name]
        for ms_name in selected_ms
        if (ms_name != "dummy.ms" and ms_name in ms_by_name.keys())
    ]
    if len(selected_ms_names) == 0:
      warnings.warn("No frequency groups were selected", RuntimeWarning)

    with open(output_file, "w") as f_stream:
        json.dump({"selected_ms": selected_ms, "output": selected_ms_names}, f_stream)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--group_id",
        help="A string that determines which frequency group of json_file should be selected",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--json_file",
        help="A path to a JSON file containing frequency groups of MeasurementSets",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--output_filename",
        help="A string which determines the name of the output file",
        default="out.json",
        type=str,
    )
    parser.add_argument(
        "mss",
        help="File paths to the input MeasurementSets to be processed",
        nargs="+",
        type=str,
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
