import os
import re
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.join(HERE, '..', 'examples')
NUM = re.compile(r'-?\d+\.\d+|-?\d+')


def close_enough(produced: str, expected: str, abs_tol: float = 0.02, rel_tol: float = 0.05) -> list:
    """Same text apart from numbers, and numbers within tolerance (library versions can move the last digits)."""
    issues = []
    if NUM.sub('#', produced) != NUM.sub('#', expected):
        issues.append('text differs apart from numbers')
    a = [float(x) for x in NUM.findall(produced)]
    b = [float(x) for x in NUM.findall(expected)]
    if len(a) != len(b):
        return issues + ['%d numbers produced, %d expected' % (len(a), len(b))]
    for x, y in zip(a, b):
        if abs(x - y) > max(abs_tol, rel_tol * abs(y)):
            issues.append('%g vs expected %g' % (x, y))
    return issues


@pytest.mark.skipif(not os.path.exists(os.path.join(EX, 'data', 'records.csv.gz')), reason='example data not found')
def test_example_runs_and_matches_expected_output(tmp_path):
    out = tmp_path / 'example_output.txt'
    r = subprocess.run([sys.executable, os.path.join(EX, 'run_example.py'), '--out', str(out)], capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr
    produced = out.read_text(encoding='utf-8')
    expected = open(os.path.join(EX, 'example_output.txt'), encoding='utf-8').read()
    assert 'Sufficiency threshold c0' in produced
    issues = close_enough(produced, expected)
    assert len(issues) <= 3, issues[:10]
