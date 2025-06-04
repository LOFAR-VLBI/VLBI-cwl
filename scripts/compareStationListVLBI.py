#!/usr/bin/env python3

import argparse
import json
import logging

import casacore.tables as pt
from losoto.h5parm import h5parm


def plugin_main(args, **kwargs):
    """
    Takes in list of targets and an h5parm solution set and returns a list of stations
    in the target data which mismatch the calibrator solutions antenna table

    Parameters
    ----------
    mapfile_in : str
        Mapfile for input measurement sets
    h5parmdb: str
        Location of the solution h5parm set
    filter: str
        Default filter constrains for the ndppp_prep_target step (usually removing International Baselines)

    Returns
    -------
    result : dict
        Output station names to filter
    """
    h5parmdb = kwargs["h5parmdb"]
    solset_name = kwargs["solset_name"]
    filter = kwargs["filter"]
    mslist = args

    if len(mslist) == 0:
        raise ValueError("Did not find any existing directory in input MS list!")
    else:
        MS = mslist[0]

    ## reading ANTENNA table of MS
    logging.info("Collecting information from the ANTENNA table.")
    antennaTable = pt.table(MS + "::ANTENNA", ack=False)
    antennaNames = antennaTable.getcol("NAME")

    if solset_name == "vlbi":
        ## reading in h5parm
        data = h5parm(h5parmdb, readonly=True)
        ## reading ANTENNA information from target / phase
        target = data.getSolset("target")
        names = target.getSoltabNames()
        phstab = [xx for xx in names if "RMextract" not in xx][0]
        soltab = target.getSoltab(phstab)
        phsants = soltab.getAxisValues("ant")
        dutch_ants = [xx for xx in phsants if "CS" in xx or "RS" in xx]
        ## reading ANTENNA information from calibrator
        solset = data.getSolset("calibrator")
        station_names = solset.getAnt().keys()
        int_ants = [xx for xx in station_names if "CS" not in xx and "RS" not in xx]
        cal_dutch = [xx for xx in station_names if "CS" in xx or "RS" in xx]
        ## remove core/remote stations not present for calibrator
        all_dutch_ants = [xx for xx in dutch_ants if xx in cal_dutch]
        station_names = all_dutch_ants + int_ants
    else:
        ## reading ANTENNA information of h5parm
        data = h5parm(h5parmdb, readonly=True)
        solset = data.getSolset(solset_name)
        station_names = solset.getAnt().keys()
    ## check whether there are more stations in the target than in the calibrator solutions
    missing_stations = list(set(antennaNames) - set(station_names))
    for missing_station in missing_stations:
        filter += ";!" + missing_station + "*"

    data.close()

    return str(filter)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "mss", help="Input data in MeasurementSet format", nargs="*", type=str
    )
    parser.add_argument(
        "--solset",
        help="The solution set from the LINC pipeline",
        type=str,
        const="",
        nargs="?",
    )
    parser.add_argument("--solset_name", help="Name of the solution set", type=str)
    parser.add_argument(
        "--filter_baselines",
        help="Filter constrains for the dp3_prep_target step",
        type=str,
    )

    return parser.parse_args()


def main():

    arguments = parse_arguments()

    filter_out = plugin_main(
        arguments.mss,
        h5parmdb=arguments.solset,
        solset_name=arguments.solset_name,
        filter=arguments.filter_baselines
    )

    with open("./out.json", "w") as fp:
        json.dump({"filter_out": filter_out}, fp)


if __name__ == "__main__":
    main()
