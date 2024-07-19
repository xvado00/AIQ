#
# Estimate AIQ from a file of log results
#
# Copyright Shane Legg 2011
# Copyright Ondřej Vadinský 2018, 2023
# Copyright Petr Zeman 2023
# Copyright Jan Štipl 2023, 2024
# Released under GNU GPLv3

import argparse
import dataclasses
import itertools
from os.path import basename
from typing import Any

import numpy as np
from numpy import ones, array, sqrt, cov
from scipy import stats

from AIQ_continue_from_log import load_log_file, LogResult


def antithetic_std(sample1: np.ndarray, sample2: np.ndarray) -> float:
    """
    This formula computes more accurate std thanks to stratified sampling,
    common random numbers, and antithetic runs in AIQ.
    Can probably be only used for grouping by strata
    Formula can be found in original Legg and Veness's paper

    :param sample1: positive antithetic runs
    :param sample2: negative antithetic runs
    :return:
    """
    s1 = sample1.std(ddof=1)  # 1 degree of freedom
    s2 = sample2.std(ddof=1)  # 1 degree of freedom
    covariance = cov(sample1, sample2)[0, 1]  # default is 1 df

    var = 0.25 * (s1 * s1 + s2 * s2 + 2.0 * covariance)

    # Covariance can be negative 0 -> sqrt from negative -> NaN
    #   Cov in general can be negative, but it didn't happen in testing
    #   Wrong calculation of cov results in it being not quite zero instead of zero
    #   Will fix when the problem occurs
    if abs(var) <= 1e-10:
        var = 0.0

    std = sqrt(var)
    return std


def print_missing_key_values(previous_key: int, current_key: int):
    """
    Adds dummy prints for missing key values.
    Will be useful for later matching for analysis purposes.
    :param previous_key:
    :param current_key:
    :return:
    """
    missing_value = "-"
    for key in range(previous_key + 1, current_key):
        # <program length> <number of programs with given length> <AAR> <HCI> <SD>
        print(f"{key:>3} {0:>3} {missing_value:^7} +/- {missing_value:^4.1} SD {missing_value:^4.1}")


@dataclasses.dataclass
class AverageByKeyResult:
    """
    AAR is the average earned reward
    HCI half of the confidence interval
    SD is the standard deviation
    """
    group_key: Any
    rewards_len: int
    AAR: float
    HCI: float
    SD: float


def print_average_by_key_results(results: list[AverageByKeyResult]) -> None:
    """
    Print format:
    <group_key> <number of programs with given length> <AAR> <HCI> <SD>

    :param results:
    :return:
    """
    prev_key = results[0].group_key
    for result in results:

        print_missing_key_values(prev_key, result.group_key)
        prev_key = result.group_key
        if result.rewards_len < 4:
            # Don't report half CI with less than 4 samples
            # program length> <number of programs with given length> <AAR>
            print(f"{result.group_key: >3} {result.rewards_len: >3} {result.AAR:>7.1f}")
        else:
            # <program length> <number of programs with given length> <AAR> <HCI> <SD>
            print(f"{result.group_key: >3} {result.rewards_len: >3} {result.AAR:>7.1f} "
                  f"+/- {result.HCI:>4.1f} SD {result.SD:>4.1f}")


def average_by_key(file_name: str, group_key=lambda x: len(x.program)) -> list[AverageByKeyResult]:
    """
    Calculates AAR, HCI, SD for groupings with given group key
    Default behavior groups on program length

    AAR is the average earned reward
    HCI half of the confidence interval
    SD is the standard deviation

    :param file_name: Path to log file
    :param group_key:
        lambda x: x.stratum_number # by stratum
        lambda x: len(x.program)   # average_by_length
    :return:
    """

    _stratum_distribution, results = load_log_file(file_name)

    try:
        # Aggregate by group key
        results.sort(key=group_key)
    except TypeError as ex:
        print(f"{file_name} Incorrect log format: group_key is not recorded, run AIQ with --log_agent_failures")
        raise ex

    result_stats = []
    groupings = itertools.groupby(results, group_key)
    for key, group in groupings:
        group: list[LogResult] = list(group)

        # Get rewards from positive and negative runs
        rewards = array([x.reward_1 for x in group] + [x.reward_2 for x in group])

        mean_reward = rewards.mean()

        # Compute standard deviation
        std_dev = rewards.std(ddof=1)

        # Handle Positive and negative run having the same reward
        if std_dev == 0:
            half_conf_int = 0
        else:
            # Compute confidence intervals
            confidence_interval = stats.norm.interval(0.95, loc=mean_reward, scale=std_dev)
            # Same as 1.96 * std_dev / sqrt(len(rewards))
            half_conf_int = (confidence_interval[1] - confidence_interval[0]) / 2 / sqrt(len(rewards))

        result_stats.append(AverageByKeyResult(key, len(rewards), mean_reward, half_conf_int, std_dev))

    return result_stats


def estimate(file, detailed):
    # load in the strata distribution
    dist_line = ["0.0"]
    dist_line += file.readline().split()
    dist = array(dist_line, float)

    p = dist  # probability of a program being in each strata
    I = len(dist)  # number of strata, including passive
    A = I - 1  # active strata

    Y = [[] for i in range(I)]  # empty collection of samples divided up by stratum
    Y[0] = [0]
    s = ones((I))  # estimated standard deviations for each stage & strata

    # read in log file results
    num_samples = 0
    for result in file:
        split_result = result.split()
        # stamp = split_result[0]
        stratum = split_result[1]
        perf1 = split_result[2]
        perf2 = split_result[3]
        # fail1 = split_result[4]
        # fail2 = split_result[5]
        # program = split_result[6]
        z = int(stratum)
        if True:  # z > 10:
            Y[int(stratum)].append((float(perf1), float(perf2)))
            num_samples += 2

    # compute empirical standard deviations for each stratum
    for i in range(1, I):
        if p[i] > 0.0 and len(Y[i]) > 2:

            YA = array(Y[i])
            sample1 = YA[:, 0]  # positive antithetic runs
            sample2 = YA[:, 1]  # negative antithetic runs

            s[i] = antithetic_std(sample1, sample2)
        else:
            s[i] = 1.0

    # report current estimates by strata
    if detailed:
        for i in range(1, I):
            stratum_samples = len(Y[i]) * 2.0
            print(" % 3d % 5d" % (i, stratum_samples), end=' ')

            if stratum_samples == 0:
                # no samples, so skip mean and half CI
                print()
            elif stratum_samples < 4:
                # don't report half CI with less than 4 samples
                print(" % 6.1f" % (array(Y[i]).mean()))
            else:
                # do a full report
                print(" % 6.1f +/- % 5.1f SD % 5.1f"
                      % (array(Y[i]).mean(), 1.96 * s[i] / sqrt(stratum_samples), s[i]))

        print()
    # compute the current estimate and 95% confidence interval
    est = 0.0
    for i in range(1, I):
        stratum_samples = len(Y[i]) * 2.0
        if p[i] > 0.0 and stratum_samples > 2:
            est += p[i] / stratum_samples * array(Y[i]).sum()

    ssd = sum(p * s)
    delta = 1.96 * ssd / sqrt(num_samples)

    print(f"{num_samples:6d}  {est: 5.1f} +/- {delta: 5.1f} SD {ssd: 5.1f}", end=' ')

    return


def main():
    parser = argparse.ArgumentParser(description="Compute AIQ from log file results, version 1.1")

    parser.add_argument("--full", action="store_true",
                        help="Reports also the strata statistics")
    parser.add_argument("--by_program_length", action="store_true",
                        help="Reports average accumulated rewards by program length. "
                             "Needs log format from AIQ --log_agent_failures")
    parser.add_argument("log_files", nargs="+", help="Path to log files")
    args = parser.parse_args()

    for file_name in args.log_files:
        if args.by_program_length:
            results = average_by_key(file_name, group_key=lambda x: len(x.program))
            print_average_by_key_results(results)
            print(f":{basename(file_name)}")
            print()

        else:
            with open(file_name, 'r') as file:
                estimate(file, args.full)
                print(":" + basename(file_name))
                if args.full:
                    print()


if __name__ == "__main__":
    main()
