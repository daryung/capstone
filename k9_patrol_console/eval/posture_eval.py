"""Run the console's pose model + posture rules on stored Go2 frames.

Same settings as vision.Vision.infer (imgsz 512, person conf 0.45, class 0),
so results match what the console shows during patrol.

Frames: every data/media/*.jpg snapshot, plus one frame per --step seconds
from each stored .avi/.mp4 clip.

Ground truth comes from a hand-made CSV (eval/labels.csv):
    frame,label,note
    label = standing | sitting | lying | none
    none  = no full person to judge (empty, lens covered, only legs, blur)

Usage (from k9_patrol_console/):
    python eval/posture_eval.py                      # current rules
    python eval/posture_eval.py --out eval/out_new   # e.g. after a rule change
Outputs: <out>/persons.csv, <out>/summary.md, <out>/img/ (annotated frames)
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from posture_rules import classify_posture, draw_postures  # noqa: E402


def frames(media, step):
    for p in sorted(media.glob('*.jpg')):
        im = cv2.imread(str(p))
        if im is not None:
            yield f'{p.stem[:8]}_snap', im
    for p in sorted(list(media.glob('*.avi')) + list(media.glob('*.mp4'))):
        cap = cv2.VideoCapture(str(p))
        fps = cap.get(cv2.CAP_PROP_FPS) or 10
        every = max(1, round(fps * step))
        i = 0
        while True:
            ok, im = cap.read()
            if not ok:
                break
            if i % every == 0:
                yield f'{p.stem[:8]}_t{i / fps:04.1f}', im
            i += 1
        cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--media', default=str(ROOT / 'data' / 'media'))
    ap.add_argument('--labels', default=str(ROOT / 'eval' / 'labels.csv'))
    ap.add_argument('--out', default=str(ROOT / 'eval' / 'out'))
    ap.add_argument('--model', default=str(ROOT / 'yolo11n-pose.pt'))
    ap.add_argument('--step', type=float, default=1.0)
    ap.add_argument('--no-img', action='store_true')
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)
    out = Path(args.out)
    (out / 'img').mkdir(parents=True, exist_ok=True)

    labels = {}
    if Path(args.labels).exists():
        with open(args.labels, encoding='utf-8') as f:
            labels = {r['frame']: r['label'] for r in csv.DictReader(f)}

    rows = []
    for name, im in frames(Path(args.media), args.step):
        r = model.predict(im, conf=.45, classes=[0], imgsz=512, verbose=False)[0]
        dets = []
        if r.boxes is not None:
            for i, b in enumerate(r.boxes):
                d = dict(target='person', confidence=float(b.conf[0]), xyxy=[float(v) for v in b.xyxy[0].tolist()])
                if r.keypoints is not None and len(r.keypoints.data) > i:
                    d['keypoints'] = r.keypoints.data[i].tolist()
                    d['posture'] = classify_posture(d['keypoints'], d['xyxy'])
                dets.append(d)
        if not dets:
            rows.append(dict(frame=name, person=0, truth=labels.get(name, ''), pred='no_person', reason='', conf='', metrics=''))
        for i, d in enumerate(dets, 1):
            p = d.get('posture', {})
            rows.append(dict(frame=name, person=i, truth=labels.get(name, '') if i == 1 else '',
                             pred=p.get('code', ''), reason=p.get('reason', ''),
                             conf=round(d['confidence'], 2), metrics=json.dumps(p.get('metrics', {}), ensure_ascii=False)))
        if dets and not args.no_img:
            cv2.imwrite(str(out / 'img' / f'{name}.jpg'), draw_postures(r.plot(), dets))

    with open(out / 'persons.csv', 'w', newline='', encoding='utf-8-sig') as f:
        # LF line endings: CRLF is stripped by `git am`, so patches would not round-trip.
        w = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)

    # Frame-level score uses person #1 (highest confidence) only on labelled frames.
    table = defaultdict(Counter)
    reasons = defaultdict(Counter)
    for r in rows:
        if r['truth'] and r['truth'] != 'none' and r['person'] in (0, 1):
            table[r['truth']][r['pred']] += 1
            if r['pred'] != r['truth']:
                reasons[r['truth']][r['reason'] or r['pred']] += 1
    preds = ['standing', 'sitting', 'lying', 'unknown', 'no_person']
    lines = ['# Go2 frame posture check', '',
             f'frames: {len({r["frame"] for r in rows})}, labelled (not none): '
             f'{sum(sum(c.values()) for c in table.values())}', '',
             '| truth \\ pred | ' + ' | '.join(preds) + ' |', '|---' * (len(preds) + 1) + '|']
    for t in ('standing', 'sitting', 'lying'):
        if t in table:
            lines.append(f'| {t} | ' + ' | '.join(str(table[t][p]) for p in preds) + ' |')
    lines += ['', '## miss reasons (person #1)']
    for t, c in reasons.items():
        for reason, n in c.most_common():
            lines.append(f'- {t}: {reason} ×{n}')
    (out / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
