#!/usr/bin/env python3

import os

from TargetListToCoords import plugin_main as main_target_list
from skynet import main as main_skynet
from compareStationListVLBI import plugin_main as main_compare_stations

# VLBI_ROOT_DIR is defined in pyproject.toml
data_dir = os.environ["VLBI_ROOT_DIR"] + "/tests/data"
ref_dir = os.environ["VLBI_ROOT_DIR"] + "/tests/reference"

catalogue = f"{ref_dir}/test_sources.csv"
skymodel = f"{ref_dir}/reference.skymodel"

def test_target_delay_calibration():
    mode = "delay_calibration"
    reference = {"name" : "ILTJ140815.23+522952.0",
                 "coords": "[212.063470875937deg,52.4977880964425deg]"}

    result = main_target_list(target_file=catalogue,
                              mode=mode)
    assert result == reference

def test_target_split_directions():
    mode = "split_directions"
    reference = {"name": "ILTJ140815.23+522952.0,"
                         "ILTJ141506.65+522419.2,"
                         "ILTJ141149.95+524856.9",
                 "coords": "[[212.063470875937deg,52.4977880964425deg],"
                           "[213.777714152472deg,52.4053592376457deg],"
                           "[212.958116589858deg,52.8158211539728deg]]"}

    result = main_target_list(target_file=catalogue,
                              mode=mode)
    assert result == reference

def test_skynet():
    import filecmp

    main_skynet(f"{data_dir}/ILTJ140815.23+522952.0", catalogue)
    assert filecmp.cmp(skymodel, "skymodel_ILTJ140815.23+522952.0.txt", shallow=False)

def test_compare_stations():
    import glob

    reference = "*&"

    filter = "*&" 
    solset = f"{data_dir}/results_target/cal_solutions.h5"
    solset_name = "vlbi"
    msin = glob.glob(f"{data_dir}/L667520*.MS")
    result = main_compare_stations(msin, filter=filter, h5parmdb=solset, solset_name=solset_name)
    assert result == reference
