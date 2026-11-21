"""탐지 처리 속도(FPS) 측정과 기록.

최근 window 초 동안 탐지 결과가 나온 프레임 수로 FPS를 계산하고, 한 장 처리 시간과
회전 검출이 돈 비율을 함께 본다. root 가 주어지면 10초마다 root/perf_log.csv 에 한 줄씩 남긴다
(회전 검출 설정이 함께 적히므로 켬/끔 비교에 그대로 쓴다).
"""
import collections
import csv
import os
import time
from datetime import datetime


class PerfMeter:
    def __init__(self, root=None, window=10., log_every=10.):
        self.window, self.log_every = window, log_every
        self.samples = collections.deque()   # (time, infer_seconds, rotated)
        self.path = None if root is None else os.path.join(str(root), 'perf_log.csv')
        self.last_log = None

    def add(self, now, infer_seconds, rotated):
        self.samples.append((now, infer_seconds, bool(rotated)))
        while self.samples and self.samples[0][0] < now - self.window:
            self.samples.popleft()
        if self.path and (self.last_log is None or now - self.last_log >= self.log_every) and len(self.samples) >= 2:
            self.last_log = now
            self._log()

    def summary(self):
        if len(self.samples) < 2:
            return dict(fps=0., infer_ms=0., rotate_ratio=0., frames=len(self.samples))
        span = self.samples[-1][0] - self.samples[0][0]
        n = len(self.samples)
        return dict(fps=round((n - 1) / span, 2) if span > 0 else 0.,
                    infer_ms=round(1000 * sum(s[1] for s in self.samples) / n, 1),
                    rotate_ratio=round(sum(s[2] for s in self.samples) / n, 2), frames=n)

    def _log(self):
        import rotate_detect
        s = self.summary()
        row = dict(time=datetime.now().isoformat(timespec='seconds'),
                   rotate=rotate_detect.MODE, rotate_when=rotate_detect.WHEN, **s)
        new = not os.path.exists(self.path)
        try:
            with open(self.path, 'a', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
        except OSError:
            pass
