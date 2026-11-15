"""누운 사람 검출 보완: 영상을 90도 돌려 한 번 더 검출한다.

포즈 모델은 서 있는 사람 위주로 학습되어, 바닥에 가로로 누운 사람을 아예 사람으로
잡지 못하는 경우가 많다(11/14 촬영의 누운 프레임 27장 중 9장만 검출). 영상을 90도 돌리면
누운 사람이 세로로 서 있는 모양이 되어 검출된다. 같은 방법이 넘어진 사람 검출 로봇 연구
(Maldonado-Bascón 외, Electronics 2019)에서 재현율을 77.7% → 92.1%로 올리는 데 쓰였다.

K9_ROTATE_DETECT=1 (기본): 호출할 때마다 한쪽 방향만 돌린다(시계/반시계 번갈아). 추론 시간 약 2배.
K9_ROTATE_DETECT=2: 매번 양쪽 방향 모두 돌린다. 추론 시간 약 3배, 누운 사람을 가장 많이 잡는다.
K9_ROTATE_DETECT=0: 끔.
"""
import os

MODE = os.environ.get('K9_ROTATE_DETECT', '1')
ENABLED = MODE != '0'
BOTH = MODE == '2'


def _iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., x1 - x0) * max(0., y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.


def map_back(x, y, turn, width, height):
    """돌린 영상의 좌표를 원본 좌표로 되돌린다. turn: 'cw'(시계) 또는 'ccw'(반시계)."""
    return (y, height - 1 - x) if turn == 'cw' else (width - 1 - y, x)


def rotated_people(model, image, confidence, existing, turn, person_class=0, imgsz=512):
    """돌린 영상에서만 잡힌 사람을 원본 좌표로 돌려준다.
    existing: 원본 영상에서 이미 잡힌 detection 목록(xyxy 포함). 겹치면(IoU>=0.3) 버린다.
    """
    import cv2
    height, width = image.shape[:2]
    code = cv2.ROTATE_90_CLOCKWISE if turn == 'cw' else cv2.ROTATE_90_COUNTERCLOCKWISE
    r = model.predict(cv2.rotate(image, code), conf=confidence, classes=[person_class], imgsz=imgsz, verbose=False)[0]
    if r.boxes is None or r.keypoints is None:
        return []
    found = []
    for i, b in enumerate(r.boxes):
        if len(r.keypoints.data) <= i:
            continue
        x0, y0, x1, y1 = [float(v) for v in b.xyxy[0].tolist()]
        p, q = map_back(x0, y0, turn, width, height), map_back(x1, y1, turn, width, height)
        box = [min(p[0], q[0]), min(p[1], q[1]), max(p[0], q[0]), max(p[1], q[1])]
        if any(_iou(box, d['xyxy']) >= .3 for d in list(existing) + found):
            continue
        keypoints = [[*map_back(x, y, turn, width, height), c] for x, y, c in r.keypoints.data[i].tolist()]
        found.append(dict(target='person', confidence=float(b.conf[0]), xyxy=box, keypoints=keypoints, rotated=turn))
    return found
