#!/usr/bin/env bash
# Find all the DIS2 solutions from a ddf-pipeline run,
# convert them to H5parm format and merge them into a
# single H5parm.
SOLSDIR=${1}
MSNAME=${2}

C=0
for f in ${SOLSDIR}/*pre-cal.ms/*DIS2*.sols.npz; do
    killMS2H5parm.py --solset sol000 --verbose DIS2_$(printf "%02d" $C).h5 $f
    ((C++))
done

h5_merger --h5_tables DIS2*.h5 --h5_out DIS2_full.h5 --propagate_flags --merge_diff_freq --add_ms_stations --ms ${MSNAME} --h5_time_freq True
