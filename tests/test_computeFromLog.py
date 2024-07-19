import io
import re
import tempfile
from contextlib import redirect_stdout

from ComputeFromLog import average_by_key, estimate


def test_average_by_key_key_gap():
    """
    Test printing dummy values for missing keys
    :return:
    """
    aiq_data = """0.20683000000013646 0.11868500000004831 0.027855000000003193 0.006694999999999811 0.02275500000000163 0.021925000000001377
2024_0707_11:41:14 1 0.205 0.59 False False ,.>>%#
2024_0707_11:41:14 1 0.66 -0.375 False False ,.+>[-.+]++#
2024_0707_11:41:14 1 -1.645 -0.135 False False <,.>-[.,]%#
2024_0707_11:41:14 2 -0.54 0.455 False False %>.,<#
2024_0707_11:41:14 6 -0.15 -1.36 False False <,-.%.%.#
2024_0707_11:41:15 6 -0.14 -0.68 False False ,-.-<<<%>#"""
    group_key = lambda x: x.stratum_number
    output = io.StringIO()
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(aiq_data.encode("utf-8"))
        # Go back to start
        tmp.seek(0)
        with redirect_stdout(output):
            average_by_key(tmp.name, group_key)

    # Trim print of the file (file name is random)
    output = "\n".join(output.getvalue().splitlines()[:-1])

    expected = """  1   6    -0.1 +/-  0.7 SD  0.8
  2   2    -0.0
  3   0    -    +/-  -   SD  -
  4   0    -    +/-  -   SD  -
  5   0    -    +/-  -   SD  -
  6   4    -0.6 +/-  0.6 SD  0.6"""
    assert output == expected


def test_average_by_key_antithetic_std():
    """
    Compare AAR, STD and CI intervals
    average_by_key doesn't use antithetic_std optimization
    -> outputs should NOT be the same
    :return:
    """
    aiq_data = """0.20683000000013646 0.11868500000004831 0.027855000000003193 0.006694999999999811 0.02275500000000163 0.021925000000001377
2024_0707_11:41:14 1 0.205 0.59 False False ,.>>%#
2024_0707_11:41:14 1 0.66 -0.375 False False ,.+>[-.+]++#
2024_0707_11:41:14 1 -1.645 -0.135 False False <,.>-[.,]%#
2024_0707_11:41:14 2 -0.54 0.455 False False %>.,<#
2024_0707_11:41:14 2 -0.37 0.22 False False .,#
2024_0707_11:41:14 3 -0.45 -1.055 False False .>,,#
2024_0707_11:41:14 4 0.225 -0.76 False False .,,%<,#
2024_0707_11:41:14 5 2.3 -0.94 False False ,+.<[,<]#
2024_0707_11:41:14 6 -0.15 -1.36 False False <,-.%.%.#
2024_0707_11:41:15 6 -0.14 -0.68 False False ,-.-<<<%>#"""
    group_key = lambda x: x.stratum_number
    output = io.StringIO()
    expected = io.StringIO()
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(aiq_data.encode("utf-8"))

        # Go back to start
        tmp.seek(0)
        with redirect_stdout(output):
            average_by_key(tmp.name, group_key)

        # Go back to start
        tmp.seek(0)
        with redirect_stdout(expected):
            estimate(tmp, detailed=True)

    # Squeeze white spaces
    output = re.sub(r" +", " ", output.getvalue())
    expected = re.sub(r" +", " ", expected.getvalue())

    # Trim difference in the output ending
    output = "\n".join(output.splitlines()[:-1])
    expected = "\n".join(expected.splitlines()[:-2])
    assert not output == expected, "Antithetic std in original implementation didn't improve accuracy"


def test_average_by_key():
    """
    Compare only AAR
    :return:
    """
    aiq_data = """0.20683000000013646 0.11868500000004831 0.027855000000003193 0.006694999999999811 0.02275500000000163 0.021925000000001377
2024_0707_11:41:14 1 0.205 0.59 False False ,.>>%#
2024_0707_11:41:14 2 -0.54 0.455 False False %>.,<#
2024_0707_11:41:14 3 -0.45 -1.055 False False .>,,#
2024_0707_11:41:14 4 0.225 -0.76 False False .,,%<,#
2024_0707_11:41:14 5 2.3 -0.94 False False ,+.<[,<]#
2024_0707_11:41:14 6 -0.15 -1.36 False False <,-.%.%.#"""
    group_key = lambda x: x.stratum_number
    output = io.StringIO()
    expected = io.StringIO()
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(aiq_data.encode('utf-8'))

        # Go back to start
        tmp.seek(0)
        with redirect_stdout(output):
            average_by_key(tmp.name, group_key)

        # Go back to start
        tmp.seek(0)
        with redirect_stdout(expected):
            estimate(tmp, detailed=True)

    # Squeeze white spaces
    output = re.sub(r" +", " ", output.getvalue())
    expected = re.sub(r" +", " ", expected.getvalue())

    # Trim difference in the output ending
    output = "\n".join(output.splitlines()[:-1])
    expected = "\n".join(expected.splitlines()[:-2])
    print()
    print(output)
    print()
    print(expected)

    assert output == expected
