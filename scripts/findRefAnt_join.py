#!/usr/bin/env python3

import argparse
import ast
import json
import re
import sys



def main(flagged_fraction_dict, station_filter, state):

    with open(flagged_fraction_dict, "r") as f_stream:
        flagged_fraction_dict_list = json.load(f_stream)

    no_station_selected = True
    while no_station_selected:
        print("Applying station filter " + str(station_filter))
        flagged_fraction_data = {}
        no_station_selected = False
        for flagged_fraction_dict in flagged_fraction_dict_list:
            entry = ast.literal_eval(flagged_fraction_dict)
            antennas = entry.keys()
            selected_stations = [
                station_name
                for station_name in antennas
                if re.match(station_filter, station_name)
            ]
            if len(selected_stations) == 0:
                print(
                    "No stations left after filtering."
                    + "Station(s) do(es) not exist in all subbands."
                    + "No filter is used."
                )
                station_filter = ""
                no_station_selected = True
                break
            for antenna in selected_stations:
                try:
                    flagged_fraction_data[antenna].append(float(entry[antenna]))
                except KeyError:
                    flagged_fraction_data[antenna] = [float(entry[antenna])]

    flagged_fraction_list = []
    sorted_stations = sorted(flagged_fraction_data.keys())

    flagged_fraction_antenna = {}

    for antenna in sorted_stations:
        flagged_fraction = sum(flagged_fraction_data[antenna]) / len(
            flagged_fraction_data[antenna]
        )
        flagged_fraction_list.append(flagged_fraction)
        flagged_fraction_antenna[antenna] = flagged_fraction
        try:
            flagged_fraction_data[flagged_fraction].append(antenna)
        except KeyError:
            flagged_fraction_data[flagged_fraction] = [antenna]

    min_flagged_fraction = min(flagged_fraction_list)
    refant = flagged_fraction_data[min_flagged_fraction][0]
    print(
        "Selected station "
        + str(refant)
        + " as reference antenna. Fraction of flagged data is "
        + "{:>3}".format("{:.1f}".format(100 * min_flagged_fraction) + "%")
    )

    flagged_fraction_antenna["state"] = state

    cwl_output = {"refant": str(refant)}

    with open("./out.json", "w") as fp:
        json.dump(cwl_output, fp)

    with open("./flagged_fraction_antenna.json", "w") as fp:
        json.dump(flagged_fraction_antenna, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flagged_fraction_dict", type=str)
    parser.add_argument("--station_filter", type=str, const="", nargs="?")
    parser.add_argument("--state", type=str)

    arguments = parser.parse_args()
    main(arguments.flagged_fraction_dict, arguments.station_filter, arguments.state)
