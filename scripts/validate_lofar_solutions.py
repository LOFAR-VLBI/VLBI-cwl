#!/usr/bin/env python3

__author__ = "Roland Timmerman, Jurjen de Jong (jurjendejong@strw.leidenuniv.nl)"

import csv
from argparse import ArgumentParser, Namespace
from pprint import pprint
from typing import Union, Sequence, Literal

import numpy as np
import pandas as pd
import tables
from scipy.stats import circstd
from scipy.optimize import leastsq
from losoto.lib_operations import reorderAxes

from make_config_international import parse_source_id


def make_utf8(inp) -> str:
    """
    Convert input to UTF-8 string.

    Args:
        inp: String or bytes input.

    Returns:
        UTF-8 encoded string.
    """
    try:
        inp = inp.decode('utf8')
        return inp
    except (UnicodeDecodeError, AttributeError):
        return inp


def sigmoid(val: Union[float, np.ndarray],
            threshold: float = 0,
            sigma: float = 1.0,
            inverted: bool = False
            ) -> Union[float, np.ndarray]:
    """
    Apply sigmoid.

    Args:
        val: Input value or array.
        threshold: Midpoint (threshold) of the sigmoid in degrees.
        sigma: Controls the steepness of the sigmoid.
        inverted: If True, invert the sigmoid (monotonically decreasing).

    Returns:
        Sigmoid-transformed value(s).
    """
    sign = 1 if inverted else -1
    return 1 / (1 + np.exp(sign * (val - threshold) / sigma))


def parabola(x: Union[float, np.ndarray],
             p: Sequence[float] | None = None,
             a: float | None = None,
             b: float = 0.0,
             c: float = 0.0,
             ) -> Union[float, np.ndarray]:
    """
    Evaluate a quadratic function a*x^2 + b*x + c.

    Either provide coefficients as a sequence `p = (a, b, c)`
    or pass `a`, `b`, and `c` explicitly.

    Args:
        x: Input value(s).
        p: Sequence of coefficients [a, b, c].
        a: Quadratic coefficient (used if `p` is not given).
        b: Linear coefficient (default: 0.0).
        c: Constant term (default: 0.0).

    Returns:
        Evaluated parabola value(s).
    """
    if p is not None:
        a_, b_, c_ = p
    elif a is not None:
        a_, b_, c_ = a, b, c
    else:
        raise ValueError("Provide either p=(a, b, c) or at least a (with optional b, c).")

    return a_ * np.asarray(x) ** 2 + b_ * np.asarray(x) + c_


def get_chi(y: Union[float, np.ndarray],
            model_y: Union[float, np.ndarray]
            ) -> Union[float, np.ndarray]:
    """
    Compute wrapped residuals between observed and model values.

    The difference (y - model_y) is wrapped into the range [-π, π],
    ensuring residuals are bounded within this interval.

    Args:
        y: Observed value(s).
        model_y: Model-predicted value(s).

    Returns:
        Residuals wrapped to the interval [-π, π].
    """
    return normalise_phases(y - model_y)


def residuals(p: tuple[float, float, float],
              y: Union[float, np.ndarray],
              x: Union[float, np.ndarray]
              ) -> Union[float, np.ndarray]:
    """
    Get residuals between observed values and parabola model,
    wrapped into the range [-π, π].

    Args:
        p: Sequence of coefficients [a, b, c].
        y: Observed value(s).
        x: Input value(s).

    Returns:
        Residuals wrapped to [-π, π].
    """
    model_y = parabola(x=x, p=p)
    return get_chi(y, model_y)


def residuals_brute(a: float,
                    y: Union[float, np.ndarray],
                    x: Union[float, np.ndarray],
                    c: float = 0.0):
    """
    Get residuals for quadratic model with fixed constant term,
    wrapped into the range [-π, π].

    Args:
        a: Quadratic coefficient or array of coefficients.
        y: Observed value(s).
        x: Input value(s).
        c: Constant term.

    Returns:
        Sum of squared residuals for each coefficient in `a`.
    """
    model_y = parabola(x=x, a=a, c=c)
    chi = get_chi(y, model_y)
    return np.sum(chi**2, axis=1)


def subtract_parabola(phases: np.ndarray):
    """
    Subtract best-fit parabola from phase solutions.

    Args:
        phases: 2D array of phase values (time × interval).

    Returns:
        2D array of phases with parabola removed, wrapped to [-π, π].
    """
    sample_range = np.linspace(-5e-4,5e-4,314) # TODO: Investigate if these are reasonable numbers for general usage
    new_phases = np.zeros(phases.shape)
    for idx, time_interval in enumerate(phases.T):
        p_0 = [0,0,0]

        if np.sum(~np.isnan(time_interval)) > 5:
            x = np.arange(-len(time_interval)//2,len(time_interval)//2)

            #Initial estimate of parameters
            p_0[2] = np.mean(time_interval[max(len(time_interval)//2-10, 0):len(time_interval)//2+10])
            chi2 = residuals_brute(sample_range[:,None], time_interval[None,~np.isnan(time_interval)], x[None,~np.isnan(time_interval)], p_0[2])
            p_0[0] = sample_range[np.argmin(chi2)]

            #Get proper fitted estimate
            p, cov = leastsq(residuals, p_0, args=(time_interval[~np.isnan(time_interval)], x[~np.isnan(time_interval)]))
            model_phases=p[0]*x**2 + p[1]*x + p[0]
            new_phases[:, idx] = time_interval - model_phases
        else:
            new_phases[:, idx] = time_interval
    norm_new_phases = normalise_phases(new_phases)

    return norm_new_phases


def calc_wraps(phase_solutions: np.ndarray,
               axes: list[str]
               ) -> float:
    """
    Calculate maximum number of phase wraps around 2pi from phase calibration solutions

    Args:
        phase_solutions: Phase calibration solutions
        axes: Axes names

    Returns: Number of phase wraps
    """

    # Get diffs
    phase_freq_diff = normalise_phases(np.diff(phase_solutions, axis=axes.index('freq')))

    # Get phase wrapping score
    freqsum = np.nansum(phase_freq_diff / (2*np.pi), axis=axes.index('freq'))
    wrap_count = np.max(np.abs(freqsum))

    return wrap_count


def normalise_phases(phase_sols: np.ndarray) -> np.ndarray:
    """
    Normalise phase solutions to the interval [-π, π].

    This ensures all phase values are wrapped into a consistent range,
    which is useful for comparing or averaging phases.

    Args:
        phase_sols: Array of phase solutions in radians.

    Returns:
        Array of phase solutions wrapped to [-π, π].
    """
    return (phase_sols - np.pi) % (2 * np.pi) - np.pi


def get_phase_noise_statistic(phase_sols: np.ndarray,
                              idx_ant: int,
                              freqs: np.ndarray
                              ) -> float:
    """
    Get phase noise statistic, which is the 90th percentile on the circular standard deviation of the phase solutions,
    after taking the 10% margin of the phase solutions, and subtracting parabola from the phase solutions.

    Args:
        phase_sols: Station referenced phase solutions
        idx_ant: Antenna index
        freqs: Frequency axis

    Returns: Phase score
    """

    phase_sols_sub = phase_sols[idx_ant]
    phase_freq_diff_sub = normalise_phases(phase_sols_sub[:-1, :] - phase_sols_sub[1:, :])

    # Slice with 10% margin on eiter side of the frequency band to discard for de-sloping
    slice_size = phase_freq_diff_sub.shape[0]//10

    delay_slope = np.nanmean(phase_freq_diff_sub[slice_size:-(slice_size+1)], axis=0)
    deslopvals = np.arange(-len(freqs) // 2, len(freqs) // 2)[:, None] * delay_slope[None, :]
    phases_desloped = normalise_phases(phase_sols_sub + deslopvals)
    phases_desloped = normalise_phases(phases_desloped - np.nanmean(phases_desloped[slice_size:-(slice_size+1)], axis=0))
    phases_desloped = subtract_parabola(phases_desloped)
    phases_desloped = normalise_phases(phases_desloped - np.nanmean(phases_desloped[slice_size:-(slice_size+1)], axis=0))

    # Get phase score
    phase_noise = circstd(phases_desloped, axis=0, nan_policy='omit')
    phase_score = np.nanpercentile(phase_noise, 90) / np.pi * 180 # Phase score based on 90th percentile of the phase noise

    return phase_score


def get_phase_score(h5: str):
    """
    Get phase score, which is based on the circular standard deviation and wrap count

    Args:
        h5: h5parm solution file

    Returns: phase score, max wrap count
    """

    with tables.open_file(h5) as H:
        phase_table = H.root.sol000.phase000
        freqs = phase_table.freq[:]
        ants = phase_table.ant[:]
        phase_sols = phase_table.val[:] * phase_table.weight[:]
        axes = make_utf8(phase_table.val.attrs["AXES"]).split(',')

        # Remove polarisation and direction axis
        if 'pol' in axes:
            phase_sols = np.take(phase_sols, 0, axis=axes.index('pol'))
        if 'dir' in axes:
            phase_sols = np.take(phase_sols, 0, axis=axes.index('dir'))

    # Reference solutions to first station
    ref_phase = np.take(phase_sols, [0], axis=axes.index('ant'))
    phase_sols -= ref_phase

    # Get wraps
    wrap_count = calc_wraps(phase_sols, axes)

    # De-slope
    phase_sols = reorderAxes(phase_sols, ['ant', 'freq', 'time'],
                                  [a for a in axes if a not in ['pol', 'dir']])

    # Get 90-percentile scores
    perc90_scores = [get_phase_noise_statistic(phase_sols, idx_ant, freqs) for idx_ant, _ in enumerate(ants)]

    phase_score = sigmoid(max(perc90_scores), 45, 10, True)

    return phase_score, wrap_count


def get_amp_score(h5: str) -> float:
    """
    Get amplitude score, which is based on the number of amplitude values above 1.5 and below 0.67, multiplied with
    (1-std), which is the standard deviation of the amplitude solutions.

    Args:
        h5: h5parm solution file

    Returns: Amplitude score between 0 and 1
    """

    with tables.open_file(h5) as H:
        amplitude_table = H.root.sol000.amplitude000
        axes = make_utf8(amplitude_table.val.attrs["AXES"]).split(',')

        # Remove polarisation axis and only select amplitude corrections where weights!=0
        if 'pol' in axes:
            weights = np.take(amplitude_table.weight[:], [0], axis=axes.index('pol'))!=0
            amplitude_sols = np.take(amplitude_table.val[:], [0], axis=axes.index('pol'))[weights]
        else:
            weights = amplitude_table.weight[:]!=0
            amplitude_sols = amplitude_table.val[:][weights]

    amplitude_std = np.std(amplitude_sols)

    # Score for large amplitude offsets
    amplitude_offset_score = 1 - len(amplitude_sols[(amplitude_sols > 1.5) | (amplitude_sols < 0.67)])/amplitude_sols.size

    # Final score
    amplitude_score = amplitude_offset_score * (1-amplitude_std)

    return amplitude_score


def get_validation_scores(solutions: list[str]):
    """
    Get validation scores from calibration solutions, based on the phase noise and amplitude variation, and write
    this to a CSV.

    Args:
        solutions: input h5parms
    """

    # Get validation metrics
    with open('validation_solutions.csv', 'w') as csvfile:
        fieldnames = ['source_id', 'amp_score', 'phase_score', 'wrap_count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for sol in solutions:
            id = parse_source_id(sol)
            if not id:
                id = sol.replace('.h5','')

            # Amplitude tables do not always exist
            try:
                amplitude_score = get_amp_score(sol)
            except tables.NoSuchNodeError:
                amplitude_score = None

            phase_score, wrap_count = get_phase_score(sol)

            scores = {
                'source_id': id,
                'amp_score': amplitude_score,
                'phase_score': phase_score,
                'wrap_count': wrap_count,
            }

            pprint(scores)

            writer.writerow(scores)


def sol_quality(csv_table: str,
                mode: Literal["DI", "DD"]):
    """
    Make solution quality acceptance column based on a combination of calibration solution quality criteria.
    The new column has a boolean to accept calibration solutions (True) or not (False).

    The thresholds for the phase and amplitude scores are based on manual tests with data from ELAIS-N1 and various
    delay calibrators from LoTSS-HR, during the VLBI busy week August 2025.

    Args:
        csv_table: CSV with image-based scores
        mode: DI or DD mode

    """

    df = pd.read_csv(csv_table)
    df['comment'] = ''
    df['accept_solutions'] = True

    # 'bad' qualifications
    if mode == 'DD':
        mask = ((df['amp_score'] < 0.9) |(df['phase_score'] < 0.8))
    elif mode == 'DI':
        mask = ((df['amp_score'] < 0.85) |(df['phase_score'] < 0.5))
    else:
        raise ValueError(f"Unsupported mode '{mode}'. Only 'DI' and 'DD' are allowed.")

    # Flag sources with accept_solutions=False if scores below threshold
    df.loc[mask, 'accept_solutions'] = False

    df.to_csv(csv_table, index=False)


def parse_args() -> Namespace:
    """
    Command line argument parser

    :return: parsed arguments
    """

    parser = ArgumentParser("Get validation scores for calibration solutions.")
    parser.add_argument('h5parms', nargs='+', help='h5parm calibration solution files with amplitude '
                        'and phase solutions.', default=None)
    parser.add_argument('--mode', help='choices=["DI" and "DD"]. The DI-mode is for '
                        'direction-independent calibration and DD-mode is for direction-dependent calibration.', default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    get_validation_scores(args.h5parms)
    sol_quality("validation_solutions.csv", args.mode)


if __name__ == '__main__':
    main()
