#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Jurjen de Jong"

from argparse import ArgumentParser
import re

import numpy as np
import pandas as pd

import casacore.tables as ct


def make_config(solint, ms, with_dutch_sols):
    """
    Make config for facetselfcal

    Args:
        solint: solution interval
        ms: MeasurementSet
        with_dutch_sols: Have used pre-applied Dutch calibration solutions.
    """

    # Get time array
    with ct.table(ms, readonly=True, ack=False) as t:
        time = np.unique(t.getcol('TIME'))

    deltime = np.abs(time[1]-time[0])
    fulltime = np.max(time)-np.min(time)

    # Solint in minutes for scalarphase
    solint_scalarphase_1 = min(max(deltime/60, np.sqrt(solint)/2), 2) # max 2 minutes
    solint_scalarphase_2 = min(max(deltime/60, np.sqrt(solint)), 3) # max 3 minutes
    if with_dutch_sols:
        solint_scalarphase_3 = min(max(deltime/60, 3 * np.sqrt(solint)), 5) # max 5 minutes
    else:
        solint_scalarphase_3 = min(max(deltime/60, 2 * np.sqrt(solint)), 3) # max 3 minutes

    # Solint in minutes for scalar complexgain (amplitudes)
    solint_complexgain_1 = max(20.0, 45 * np.sqrt(solint)) # min 20 minutes
    if with_dutch_sols:
        solint_complexgain_2 = 2.0 * solint_complexgain_1
    else:
        solint_complexgain_2 = 1.5 * solint_complexgain_1

    # Decide if amplitude solve or not based on solint size
    cg_cycle_1 = 3
    if solint_complexgain_1/60 > 5:
        cg_cycle_1 = 999
    elif solint_complexgain_1/60 > 3:
        solint_complexgain_1 = 240.

    # Decide if amplitude solve or not based on solint size
    cg_cycle_2 = 4
    if solint_complexgain_2/60 > 5:
        cg_cycle_2 = 999
    elif solint_complexgain_2/60 > 3:
        solint_complexgain_2 = 240.

    # UV-min larger for high S/N sources and smaller for low S/N sources
    uvmin = int(40000 - 20000 * np.exp(-1/solint))

    stop = 16
    imsize = 2048

    # Extra time-averaging when solint larger than 60 seconds
    if solint_scalarphase_1 * 60 > deltime * 2:
        avgstep = 2
    else:
        avgstep = 1

    # Different configurations for different S/N
    if solint<0.05:
        smoothness_phase = 7.5
        smoothness_complex = 10.0
        soltypecycles_list = f'[0,0,1,{cg_cycle_1},{cg_cycle_2}]'
        soltype_list = "['scalarphase','scalarphase','scalarphase','scalarcomplexgain','scalarcomplexgain']"
        solint_list = f"['{int(solint_scalarphase_1 * 60)}s','{int(solint_scalarphase_2 * 60)}s','{int(solint_scalarphase_3 * 60)}s','{int(solint_complexgain_1 * 60)}s','{int(solint_complexgain_2 * 60)}s']"
        smoothnessconstraint_list = f"[{smoothness_phase},{smoothness_phase},{smoothness_phase*1.5},{smoothness_complex},{smoothness_complex}]"
        smoothnessreffrequency_list = "[120.0,120.0,120.0,0.0,0.0]"
        smoothnessspectralexponent_list = "[-1.0,-1.0,-1.0,-1.0,-1.0]"
        if with_dutch_sols:
            resetsols_list = "['alldutchandclosegerman','alldutch','coreandfirstremotes','alldutch','coreandfirstremotes']"
        else:
            resetsols_list = "['alldutchandclosegerman','alldutch',None,'alldutch',None]"

    elif solint<0.1:
        smoothness_phase = 10.0
        smoothness_complex = 10.0
        soltypecycles_list = f'[0,0,1,{cg_cycle_1},{cg_cycle_2}]'
        soltype_list = "['scalarphase','scalarphase','scalarphase','scalarcomplexgain','scalarcomplexgain']"
        solint_list = f"['{int(solint_scalarphase_1 * 60)}s','{int(solint_scalarphase_2 * 60)}s','{int(solint_scalarphase_3 * 60)}s','{int(solint_complexgain_1 * 60)}s','{int(solint_complexgain_2 * 60)}s']"
        smoothnessconstraint_list = f"[{smoothness_phase},{smoothness_phase},{smoothness_phase*1.5},{smoothness_complex},{smoothness_complex+5.0}]"
        smoothnessreffrequency_list = "[120.0,120.0,120.0,0.0,0.0]"
        smoothnessspectralexponent_list = "[-1.0,-1.0,-1.0,-1.0,-1.0]"
        if with_dutch_sols:
            resetsols_list = "['alldutchandclosegerman','alldutch','coreandallbutmostdistantremotes','alldutch','coreandallbutmostdistantremotes']"
        else:
            resetsols_list = "['alldutchandclosegerman','alldutch',None,'alldutch',None]"

    elif solint<1:
        smoothness_phase = 10.0
        smoothness_complex = 12.5
        soltypecycles_list = f'[0,0,1,{cg_cycle_1},{cg_cycle_2}]'
        soltype_list = "['scalarphase','scalarphase','scalarphase','scalarcomplexgain','scalarcomplexgain']"
        solint_list = f"['{int(solint_scalarphase_1 * 60)}s','{int(solint_scalarphase_2 * 60)}s','{int(solint_scalarphase_3 * 60)}s','{int(solint_complexgain_1 * 60)}s','{int(solint_complexgain_2 * 60)}s']"
        smoothnessconstraint_list = f"[{smoothness_phase},{smoothness_phase},{smoothness_phase*1.5},{smoothness_complex},{smoothness_complex+5.0}]"
        smoothnessreffrequency_list = "[120.0,120.0,120.0,0.0,0.0]"
        smoothnessspectralexponent_list = "[-1.0,-1.0,-1.0,-1.0,-1.0]"
        if with_dutch_sols:
            resetsols_list = "['alldutchandclosegerman','alldutch','coreandallbutmostdistantremotes','alldutch','coreandallbutmostdistantremotes']"
        else:
            resetsols_list = "['alldutchandclosegerman','alldutch',None,'alldutch',None]"

    elif solint<5:
        smoothness_phase = 10.0
        smoothness_complex = 15.0
        soltypecycles_list = f'[0,0,1,{cg_cycle_1},{cg_cycle_2}]'
        soltype_list = "['scalarphase','scalarphase','scalarphase','scalarcomplexgain','scalarcomplexgain']"
        solint_list = f"['{int(solint_scalarphase_1 * 60)}s','{int(solint_scalarphase_2 * 60)}s','{int(solint_scalarphase_3 * 60)}s','{int(solint_complexgain_1 * 60)}s','{int(solint_complexgain_2 * 60)}s']"
        smoothnessconstraint_list = f"[{smoothness_phase},{smoothness_phase},{smoothness_phase*1.5},{smoothness_complex},{smoothness_complex+10.0}]"
        smoothnessreffrequency_list = "[120.0,120.0,120.0,0.0,0.0]"
        smoothnessspectralexponent_list = "[-1.0,-1.0,-1.0,-1.0,-1.0]"
        if with_dutch_sols:
            resetsols_list = "['alldutchandclosegerman','alldutch','coreandallbutmostdistantremotes','alldutch','coreandallbutmostdistantremotes']"
        else:
            resetsols_list = "['alldutchandclosegerman','alldutch',None,'alldutch',None]"

    elif solint<10:
        smoothness_phase = 10.0
        smoothness_complex = 20.0
        soltypecycles_list = f'[0,0,{cg_cycle_1},{cg_cycle_2}]'
        soltype_list = "['scalarphase','scalarphase','scalarcomplexgain','scalarcomplexgain']"
        solint_list = f"['{int(solint_scalarphase_1*60)}s','{int(solint_scalarphase_2*60)}s','{int(solint_complexgain_1*60)}s','{int(solint_complexgain_2*60)}s']"
        smoothnessconstraint_list = f"[{smoothness_phase},{smoothness_phase*1.25},{smoothness_complex},{smoothness_complex+5.0}]"
        smoothnessreffrequency_list = "[120.0,120.0,0.0,0.0]"
        smoothnessspectralexponent_list = "[-1.0,-1.0,-1.0,-1.0]"
        if with_dutch_sols:
            resetsols_list = "['alldutchandclosegerman','alldutch','alldutchandclosegerman','alldutch']"
        else:
            resetsols_list = "['alldutch',None,'alldutch',None]"

    else:
        soltypecycles_list = f'[0,0,{cg_cycle_1}]'
        soltype_list = "['scalarphase','scalarphase','scalarcomplexgain']"
        solint_list = f"['{int(solint_scalarphase_1*60)}s','{int(solint_scalarphase_2*60)}s','{int(solint_complexgain_1*60)}s']"
        smoothnessconstraint_list = f"[10.0,15.0,25.0]"
        smoothnessreffrequency_list = "[120.0,120.0,0.0]"
        smoothnessspectralexponent_list = "[-1.0,-1.0,-1.0]"
        if with_dutch_sols:
            resetsols_list = "['alldutchandclosegerman','alldutch','alldutch']"
        else:
            resetsols_list = "['alldutch',None,None]"


    config=f"""imagename                       = "{parse_source_id(ms)}"
phaseupstations                 = "core"
forwidefield                    = True
autofrequencyaverage            = True
update_multiscale               = True
soltypecycles_list              = {soltypecycles_list}
soltype_list                    = {soltype_list}
smoothnessconstraint_list       = {smoothnessconstraint_list}
smoothnessreffrequency_list     = {smoothnessreffrequency_list}
smoothnessspectralexponent_list = {smoothnessspectralexponent_list}
solint_list                     = {solint_list}
uvmin                           = {uvmin}
imsize                          = {imsize}
resetsols_list                  = {resetsols_list}
paralleldeconvolution           = 1024
targetcalILT                    = "scalarphase"
stop                            = {stop}
parallelgridding                = 6
channelsout                     = 12
fitspectralpol                  = 5
early_stopping                  = True
"""
    if avgstep>1:
        config+=f"""avgtimestep                     = {avgstep}
"""

    # write to file
    with open(ms+".config.txt", "w") as f:
        f.write(config)


def parse_source_id(inp_str: str = None):
    """
    Parse ILTJhhmmss.ss±ddmmss.s source_id string

    Args:
        inp_str: ILTJ source_id

    Returns: ILT‐coordinate string

    """

    try:
        parsed_inp = re.findall(r'ILTJ\d{6}\.\d{2}[+\-]\d{6}\.\d{1}', inp_str)[0]
    except IndexError:
        print(f"Error: {inp_str} does not contain a valid source ID")

    return parsed_inp


def get_solint(ms, phasediff_output):
    """
    Get solution interval from phase-diff CSV output.

    Args:
        ms: MeasurementSet.
        phasediff_output: Path to the Phase-diff CSV output.

    Returns:
        solint: Solution interval in minutes.
    """

    phasediff = pd.read_csv(phasediff_output)
    sourceid = parse_source_id(ms.split("/")[-1])

    possible_columns = {"Source_id", "source",}  # Handling possible column variations (versions)
    matching_columns = list(set.intersection(possible_columns, phasediff.columns))
    if len(matching_columns) != 1:
        raise KeyError("CSV must contain one of", possible_columns)
    else:
        solint = phasediff[phasediff[matching_columns[0]].apply(parse_source_id) == sourceid]["best_solint"].min()

    return solint



def parse_args():
    """
    Command line argument parser

    Returns: parsed arguments
    """

    parser = ArgumentParser(description='Make parameter configuration file for facetselfcal.')
    parser.add_argument('--ms', type=str, help='Input MeasurementSet.')
    parser.add_argument('--phasediff_output', type=str, help='Phasediff CSV output.')
    parser.add_argument('--dutch_multidir_h5', action="store_true", help='Have used pre-applied Dutch calibration solutions.')
    return parser.parse_args()


def main():
    """
    Main function
    """

    args = parse_args()

    solint = get_solint(args.ms, args.phasediff_output)
    make_config(solint, args.ms, args.dutch_multidir_h5)


if __name__ == "__main__":
    main()
