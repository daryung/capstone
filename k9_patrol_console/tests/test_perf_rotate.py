import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rotate_detect as rd
from perf_meter import PerfMeter


def person(code, too_close=False):
    return dict(target='person', posture=dict(code=code, metrics=dict(too_close=True) if too_close else {}))


def test_should_run_modes():
    assert rd.should_run([person('standing')], 'always')
    assert not rd.should_run([person('standing')], 'unknown')
    assert rd.should_run([person('standing'), person('unknown')], 'unknown')
    assert not rd.should_run([person('unknown', too_close=True)], 'unknown')
    assert not rd.should_run([], 'unknown')
    assert rd.should_run([], 'unknown_or_empty')


def test_perf_summary_and_log(tmp_path):
    m = PerfMeter(tmp_path, window=10, log_every=0)
    for i in range(11):
        m.add(i * .5, .1, i % 2 == 0)
    s = m.summary()
    assert s['fps'] == 2.0 and s['infer_ms'] == 100.0 and 0.5 <= s['rotate_ratio'] <= 0.6
    assert (tmp_path / 'perf_log.csv').read_text(encoding='utf-8').startswith('time,rotate,rotate_when')
