from dataclasses import dataclass
from typing import Any


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
