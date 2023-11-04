
#
# Restore a previous AIQ test run from log files
#
# Copyright Jan Štipl 2023
# Released under GNU GPLv3
#

from dataclasses import dataclass
from typing import Any
from pathlib import Path


@dataclass
class LogResult:
    """Class for keeping track of results in the log"""

    time: str  # %Y_%m%d_%H:%M:%S
    stratum_number: int
    reward_1: float
    reward_2: float
    fail1: Any
    fail2: Any
    program: str


def load_log_file(log_path: str) -> tuple[list[float], list[LogResult]]:
    """
    Loads log file
    :param log_path:
    :return: stratum_distribution, results
    """
    with open(log_path, "r") as file:
        # First line are probabilities for each stratum (distribution)
        # Load and parse to float
        stratum_distribution = ["0.0"] + file.readline().split()
        stratum_distribution = [float(x) for x in stratum_distribution]

        results = []
        for line in file:
            line = line.split(" ")
            assert len(line) in [4, 7], f"Can't parse line '{line}', Invalid format: {log_path}"

            if len(line) == 4:
                time, stratum_number, reward_1, reward_2 = line
                fail1 = fail2 = program = None
            else:
                # We have failure logging enabled
                time, stratum_number, reward_1, reward_2, fail1, fail2, program = line
                program: str = program.strip()
            stratum_number = int(stratum_number)
            reward_1 = float(reward_1)
            reward_2 = float(reward_2)

            results.append(LogResult(time, stratum_number, reward_1, reward_2, fail1, fail2, program))

    return stratum_distribution, results


def copy_verbose_log_el(old_main_log: str, current_main_log: str, N: list[float]):
    """
    Copies old verbose logs in log-el/<old_run> to log-el/<current_run>

    :param old_main_log:
    :param current_main_log:
    :param N: list of stage thresholds
    :return:
    """
    # Get the old log
    old_base_name = Path(old_main_log).name.replace(".log", "")
    current_base_name = Path(current_main_log).name.replace(".log", "")

    old_dir = Path("./log-el/" + old_base_name)
    curr_dir = Path("./log-el/" + current_base_name)

    msg = "Can't copy old verbose logs"
    assert old_dir.is_dir(), f"{old_dir} is not directory, {msg}"
    assert curr_dir.is_dir(), f"{curr_dir} is not directory, {msg}"

    old_log_paths = list(old_dir.iterdir())
    curr_log_paths = list(curr_dir.iterdir())

    assert len(old_log_paths) == len(curr_log_paths), f"Old verbose config is not tha same as current, {msg}"
    if len(old_log_paths) == len(curr_log_paths) == 0:
        print(f"Verbose logging is not enables, {msg}")
        return

    # Get the completed stage len
    completed_stage_lines = calculate_completed_stage_lines(N, old_log_paths[0])

    for old_log, curr_log in zip(old_log_paths, curr_log_paths):
        # Remove date from name and check if they are same
        old_check = "_".join(old_log.name.split("_")[:-5])
        curr_check = "_".join(curr_log.name.split("_")[:-5])
        assert old_check == curr_check, (f"Old setup is not the same as current"
                                         f" {old_check} != {curr_check}, {msg}")

        with open(old_log, 'r') as old_file, open(curr_log, 'w') as curr_file:
            # Hopefully this will not explode my RAM, if it does use sequential
            lines = old_file.readlines()
            #  completed_stage_lines + 1 because of the "header" line
            curr_file.writelines(lines[:completed_stage_lines + 1])

            print(f"Complete stages in verbose log loaded {old_log} -> {curr_log}")


def calculate_completed_stage_lines(N: list[float], log_file_path: Path) -> int:
    """
    Calculates number of lines in completed stages
    :param N: list of stage thresholds
    :param log_file_path:
    :return:
    """

    with open(log_file_path, 'r') as file:
        log_lines = len(file.readlines())

    # Subtract 1 because there is 1 "header" line
    # # 1 line is 2 agent runs
    runs = 2 * (log_lines - 1)

    # https://stackoverflow.com/questions/36275459/find-the-closest-elements-above-and-below-a-given-number
    lower = [i for i in N if runs >= i]
    last_completed = max(lower)

    print(f"{log_lines} lines loaded -> last completed stage is {len(lower) - 1}\\{len(N) - 1}")
    # We want log lines not runs -> divide by 2
    return int(last_completed / 2)
