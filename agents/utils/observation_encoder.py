
#
# Observation encoder for agents to be tested in the AIQ test
#
# Copyright Jan Štipl 2024
# Released under GNU GPLv3
#

import numpy as np


def encode_observations_n_hot(observations, obs_cells, obs_symbols) -> np.ndarray:
    """
    Perform one hot encoding for each observation
    Order is Big-endian
        First observation is order of X^0, second X^1, ...
    e.g.:
    obs_cells = 2
    obs_symbols = 3
    e.g. first = 0, second = 2
    => observations = [0, 2]
    returned = np.array([
        [1., 0., .0],
        [0., 0., 1.],
    ]).reshape(-1)

    :return: flat tensor of shape=[observations * obs_cells, 1] with observations encoded
    """
    observations_encoding = np.zeros(shape=[obs_cells, obs_symbols], dtype=np.float32)
    for i, obs in enumerate(observations):
        # Observations are in range <0, obs_symbols -1>
        observations_encoding[i][obs] = 1

    return observations_encoding.reshape(-1)


def encode_observations_int(observations: list[int], obs_symbols: int) -> int:
    """
    Convert observations into a single number
    First observation is order of X^0, second X^1, ...
    e.g.:
    obs_cells = 2
    obs_symbols = 3
    e.g. first = 2, second = 1
        => observations = [2, 1]
        => encoding = 5
    e.g. [0,0] => 0, [1,0] => 1, [2,0] => 2 ...

    speed is O(1)
    :param observations:
    :param obs_symbols:
    :return:
    """
    encoded = 0
    for i, obs in enumerate(observations):
        encoded += obs * (obs_symbols ** i)

    return encoded


def test_more_obs():
    import torch
    # BF 5,2
    observations = [3, 2]
    obs_cells = 2
    obs_symbols = 5
    config = [observations, obs_cells, obs_symbols]
    expected = [
        [0., 0., 0., 1., 0.],
        [0., 0., 1., 0., 0.],
    ]
    expected = torch.as_tensor(expected).reshape(-1)
    assert len(expected) == obs_symbols * obs_cells
    received = encode_observations_n_hot(*config)
    received = torch.as_tensor(received, dtype=torch.float32)
    print(f"Expected: {expected}, Received: {received}")
    print(f"Expected == Received: {expected == received}")


def test_encode_observations_int():
    obs_symbols = 3

    expected = (
        [0, 0], [1, 0], [2, 0],
        [0, 1], [1, 1], [2, 1],
        [0, 2], [1, 2], [2, 2]
    )
    for i, obs in enumerate(expected):
        assert encode_observations_int(obs, obs_symbols) == i

    obs_symbols = 5
    assert encode_observations_int([0, 0, 0], obs_symbols) == 0
    assert encode_observations_int([1, 0, 0], obs_symbols) == 1
    assert encode_observations_int([4, 4, 4], obs_symbols) == 124


def test_history_encode_observations_n_hot():
    """
    Previous observation is expected to be in the same place as 2nd observation
    when running with obs_cells=2

    First observation is order of X^0, second X^1, ...
    eg. first = 1, second = 2
        => observations = [1, 2]
        => n_hot encoding = [0, 1, 0, 1, 0, 0]
    :return:
    """
    _obs_cells = 1
    obs_symbols = 3

    current_obs = [0]
    prev_obs = [2]

    observations = current_obs + prev_obs
    expected = np.array([
        [1., 0., .0],
        [0., 0., 1.],
    ]).reshape(-1)
    rec = encode_observations_n_hot(observations, len(observations), obs_symbols)
    assert (rec == expected).all()


def test_all():
    test_more_obs()
    test_encode_observations_int()
    test_history_encode_observations_n_hot()


def main():
    test_all()


if __name__ == '__main__':
    main()
