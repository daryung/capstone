"""새 촬영 영상으로 자세 판별을 다시 채점한다 (최종 수치용).

1단계 (판별·정답지 만들기)
  python eval/score_shoot.py frames --date 2026-11-21 --out eval/shoot_20261121
  - 그날 저장된 클립을 0.5초 간격으로 콘솔과 같은 방식(포즈 모델 + 영상 회전 + 자세 규칙)으로 판별
  - 사람마다 관절을 그린 사진(sheets/*.jpg)과 정답지 labels.csv 를 만든다.
    labels.csv 의 truth 칸에 사진을 보고 standing / sitting / lying / part(너무 가깝거나 일부만) 를 적는다.
2단계 (채점)
  python eval/score_shoot.py score --out eval/shoot_20261121
  - 정답별로 판별 결과를 세어 result.md 로 쓴다. 회전 검출을 켠 결과와 끈 결과를 함께 보인다.
정답은 판별 결과를 보기 전에 사진만 보고 적는다(sheets 에는 판별 결과를 쓰지 않는다).
"""
import argparse
import collections
import csv
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from posture_rules import classify_posture  # noqa: E402
import rotate_detect  # noqa: E402

EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def frames(a):
    import cv2
    import numpy as np
    from ultralytics import YOLO
    out = Path(a.out); (out / 'sheets').mkdir(parents=True, exist_ok=True)
    model = YOLO(str(ROOT / a.model))
    db = sqlite3.connect(ROOT / 'data' / 'records.sqlite3')
    clips = db.execute("select id,clip from records where time like ? and (clip like '%.avi' or clip like '%.mp4') order by time",
                       (a.date + '%',)).fetchall()
    rows, tiles = [], []
    for rid, clip in clips:
        cap = cv2.VideoCapture(str(ROOT / 'data' / 'media' / Path(clip).name))
        fps = cap.get(cv2.CAP_PROP_FPS) or 10
        step, i = max(1, round(fps * a.step)), 0
        while True:
            ok, im = cap.read()
            if not ok:
                break
            if i % step == 0:
                r = model.predict(im, conf=.45, classes=[0], imgsz=512, verbose=False)[0]
                dets = []
                if r.boxes is not None and r.keypoints is not None:
                    for j, b in enumerate(r.boxes):
                        dets.append(dict(src='original', xyxy=[float(v) for v in b.xyxy[0].tolist()],
                                         keypoints=r.keypoints.data[j].tolist()))
                for d in dets:
                    d['posture'] = classify_posture(d['keypoints'], d['xyxy'])
                if rotate_detect.should_run([dict(target='person', posture=d['posture']) for d in dets], a.when):
                    for turn in ('cw', 'ccw'):
                        for d in rotate_detect.rotated_people(model, im, .45, dets, turn):
                            d['src'] = 'rotated'; d['posture'] = classify_posture(d['keypoints'], d['xyxy']); dets.append(d)
                for k, d in enumerate(dets):
                    name = f'{rid[:8]}_{i:04d}_{k}'
                    rows.append(dict(id=name, clip=rid[:8], frame=i, src=d['src'], pred=d['posture']['code'],
                                     pred_label=d['posture']['label'], truth=''))
                    t = cv2.resize(im.copy(), (480, 270)); s = 480 / im.shape[1]
                    x0, y0, x1, y1 = [int(v * s) for v in d['xyxy']]
                    cv2.rectangle(t, (x0, y0), (x1, y1), (0, 200, 255), 2)
                    for u, v in EDGES:
                        p, q = d['keypoints'][u], d['keypoints'][v]
                        if p[2] >= .45 and q[2] >= .45:
                            cv2.line(t, (int(p[0] * s), int(p[1] * s)), (int(q[0] * s), int(q[1] * s)), (80, 220, 80), 2)
                    cv2.putText(t, name, (4, 16), 0, .5, (255, 255, 255), 2)
                    tiles.append(t)
            i += 1
    for n in range(0, len(tiles), 20):
        t = tiles[n:n + 20]; t += [np.zeros_like(t[0])] * (-len(t) % 4)
        cv2.imwrite(str(out / 'sheets' / f'sheet_{n // 20:03d}.jpg'), np.vstack([np.hstack(t[k:k + 4]) for k in range(0, len(t), 4)]))
    with open(out / 'labels.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['id', 'truth']); w.writeheader()
        for r in rows:
            w.writerow(dict(id=r['id'], truth=''))
    (out / 'predictions.json').write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
    print(f'클립 {len(clips)}개, 사람 검출 {len(rows)}건, 사진 {len(tiles)}장 → {out}')


def score(a):
    out = Path(a.out)
    pred = {r['id']: r for r in json.loads((out / 'predictions.json').read_text(encoding='utf-8'))}
    truth = {r['id']: r['truth'].strip() for r in csv.DictReader(open(out / 'labels.csv', encoding='utf-8-sig')) if r['truth'].strip()}
    codes = ['standing', 'sitting', 'lying', 'unknown']
    lines = ['# 새 촬영 영상 자세 판별 채점', '', f'- 정답이 적힌 사람 검출 {len(truth)}건', '']
    for title, keep in (('회전 검출 켬', lambda r: True), ('회전 검출 끔 (원본 검출만)', lambda r: r['src'] == 'original')):
        tab = collections.defaultdict(collections.Counter)
        for k, t in truth.items():
            if k in pred and keep(pred[k]):
                tab[t][pred[k]['pred']] += 1
        lines += [f'## {title}', '', '| 정답 \\ 판별 | 서 있음 | 앉음 | 누움 | 판단 불가 | 맞음 |', '|---|---|---|---|---|---|']
        for t in ('standing', 'sitting', 'lying', 'part'):
            c = tab[t]; n = sum(c.values())
            ok = c['unknown'] if t == 'part' else c[t]
            lines.append(f'| {t} ({n}) | ' + ' | '.join(str(c[x]) for x in codes) + f' | {ok} / {n} |')
        lines.append('')
    (out / 'result.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('step_name', choices=['frames', 'score'])
    ap.add_argument('--date', default='2026-11-21'); ap.add_argument('--out', default='eval/shoot_20261121')
    ap.add_argument('--model', default='yolo11n-pose.pt'); ap.add_argument('--step', type=float, default=.5)
    ap.add_argument('--when', default='always', help='채점용 회전 검출 조건 (기본: 매 프레임)')
    a = ap.parse_args()
    frames(a) if a.step_name == 'frames' else score(a)
