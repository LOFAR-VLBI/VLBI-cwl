#!/usr/bin/env python3

import argparse
import json
import os
import sys


def main():

    arguments = parse_arguments()
    ms_list = arguments.mss
    group_id = arguments.group_id
    json_file = arguments.json_file

    ms_by_name = {
        os.path.basename(ms): {"class": "Directory", "path": ms} for ms in ms_list
    }

    with open(json_file, "r") as f_stream:
        selected_ms = json.load(f_stream)[group_id]

    selected_ms = [os.path.basename(ms_name) for ms_name in selected_ms]
    with open("./out.json", "w") as fp:
        json.dump({"selected_ms": selected_ms}, fp)

    selected_ms = [
        ms_by_name[ms_name] for ms_name in selected_ms if ms_name != "dummy.ms"
    ]
    with open("./selected_ms.json", "w") as f_stream:
        json.dump(selected_ms, f_stream)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--group_id",
        help="A string that determines which MeasurementSets should be combined",
        type=str,
    )
    parser.add_argument(
        "--json_file",
        help="A path to a file containing directories of MeasurementSets",
        type=str,
    )
    parser.add_argument(
        "mss",
        help="File paths to the input MeasurementSets to be processed",
        nargs="*",
        type=str,
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
