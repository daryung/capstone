"""COCO-17 자세 후보 규칙 v3 (프레임 단위, 표준 라이브러리만 사용).

v2와 다른 점
- 2D 관절 '각도' 대신 관절의 '세로 위치 비율'을 쓴다. 정면에서 앉으면 허벅지가
  카메라 쪽을 향해 각도가 무너지지만, 무릎이 골반 높이 근처에 있다는 사실은 그대로다.
- 양쪽 관절을 모두 요구하지 않는다. 보이는 쪽만으로 계산한다.
- 판단 불가의 이유를 '너무 가깝거나 일부만 보임'과 '보이지만 조건 불충족'으로 나눈다.
라벨은 자세 후보이며 낙상·의식·건강 상태를 뜻하지 않는다.
"""
import math

LABELS = {'standing': '서 있음 후보', 'sitting': '앉아 있음 후보',
          'lying': '누워 있음 후보', 'unknown': '판단 불가'}
KEYPOINT_MIN = .45
STAND_SHARE = .40   # 골반→발목 세로 길이 중 허벅지(골반→무릎)가 차지하는 비율
SIT_SHARE = .35
FLOOR_RATIO = .45
TOO_CLOSE_LABEL = '너무 가까움'
TORSO_PER_SHOULDER = 1.4   # 몸통 길이 / 어깨 너비 (성인 평균에 가까운 값)
LYING_ELEVATION = 45       # 몸통이 바닥에서 일어선 각도가 이 이하면 누움 쪽으로 본다
FORESHORTEN = .7   # 몸통 길이 / 어깨 너비가 이보다 작으면 몸통이 카메라 방향으로 놓인 것으로 본다   # 골반→발목 세로 길이 / 몸통 길이


def _mid(points, a, b):
    got = [points[i] for i in (a, b) if i in points]
    if not got:
        return None
    return (sum(p[0] for p in got) / len(got), sum(p[1] for p in got) / len(got))


def _tilt(a, b):
    """0 = 세로, 90 = 가로."""
    return math.degrees(math.atan2(abs(b[0] - a[0]), abs(b[1] - a[1])))


def torso_elevation(dx, dy, shoulder_w, ratio=TORSO_PER_SHOULDER):
    """몸통이 바닥에서 일어선 각도(도)를 2D 좌표에서 추정한다. 90 = 똑바로 세움, 0 = 바닥과 평행.
    사람이 카메라를 비스듬히 향하면 몸통의 가로 성분이 줄어 실제보다 서 있는 것처럼 보인다.
    그만큼 어깨 너비는 넓게 찍히므로, 몸통 길이 = 어깨 너비 x ratio 라는 신체 비율로 방향을 소거한다.
        dy = L sin(e),  dx = L cos(e) |sin(yaw)|,  어깨너비 = (L/ratio) |cos(yaw)|
    를 풀면 u = sin^2(e) 는  k u^2 - (dy^2 + dx^2 + k) u + dy^2 = 0  (k = ratio^2 x 어깨너비^2) 의 작은 근이다.
    어깨가 한쪽만 보이면(shoulder_w = 0) 일반적인 2D 기울기와 같다.
    """
    dy2, dx2, k = dy * dy, dx * dx, (ratio * shoulder_w) ** 2
    if k < 1e-6:
        u = dy2 / (dy2 + dx2) if dy2 + dx2 > 0 else 1.
    else:
        a = dy2 + dx2 + k
        u = (a - math.sqrt(max(a * a - 4 * k * dy2, 0.))) / (2 * k)
    return math.degrees(math.asin(math.sqrt(min(max(u, 0.), 1.))))


def classify_posture(keypoints, box, min_conf=KEYPOINT_MIN, shin_only=True):
    result = dict(code='unknown', label=LABELS['unknown'], reason='', metrics={},
                  method='coco17_rules_v3.5', keypoint_threshold=min_conf)

    def finish(code, reason, **metrics):
        result['metrics'].update(metrics)
        result.update(code=code, label=LABELS[code], reason=reason)
        return result

    if keypoints is None or len(keypoints) != 17 or box is None or len(box) != 4:
        return finish('unknown', 'COCO 17개 관절 좌표 필요')
    try:
        box = [float(v) for v in box]
        w, h = box[2] - box[0], box[3] - box[1]
        if not all(math.isfinite(v) for v in box) or w <= 0 or h <= 0:
            raise ValueError()
    except (ValueError, TypeError):
        return finish('unknown', '사람 박스 오류')
    points = {}
    for i, p in enumerate(keypoints):
        try:
            x, y, c = map(float, p[:3])
        except (TypeError, ValueError):
            continue
        if all(math.isfinite(v) for v in (x, y, c)) and c >= min_conf and not (x == 0 and y == 0):
            points[i] = (x, y)

    shoulder, hip = _mid(points, 5, 6), _mid(points, 11, 12)
    legs = []
    for side, hi, kn, an in (('left', 11, 13, 15), ('right', 12, 14, 16)):
        leg = dict(side=side, hip=points.get(hi) or hip, knee=points.get(kn), ankle=points.get(an))
        if leg['knee'] or leg['ankle']:
            legs.append(leg)
    result['metrics']['box_aspect'] = round(w / h, 2)

    # ── 1. 몸통(어깨·골반)이 보이는 경우 ─────────────────────────────
    if shoulder and hip:
        dx, dy = hip[0] - shoulder[0], hip[1] - shoulder[1]
        torso = math.hypot(dx, dy)
        tilt = _tilt(shoulder, hip)
        body = [points[i] for i in (5, 6, 11, 12, 13, 14, 15, 16) if i in points]
        span_x = max(p[0] for p in body) - min(p[0] for p in body)
        span_y = max(p[1] for p in body) - min(p[1] for p in body)
        aspect = span_x / max(span_y, 1)
        shoulder_w = math.dist(points[5], points[6]) if 5 in points and 6 in points else 0.
        scale = max(torso, shoulder_w)
        low = [p[1] for leg in legs for p in (leg['knee'], leg['ankle']) if p]
        drop = (max(low) - hip[1]) if low else None   # 다리의 가장 낮은 점이 골반보다 얼마나 아래인가
        result['metrics'].update(basis='torso', torso_tilt_degrees=round(tilt, 1),
                                 body_aspect=round(aspect, 2),
                                 torso_over_shoulder_width=round(torso / shoulder_w, 2) if shoulder_w else None,
                                 leg_drop_over_scale=round(drop / scale, 2) if low and scale else None)
        if scale < max(8, .08 * max(w, h)):
            return finish('unknown', '몸통이 너무 작거나 관절 겹침')
        # 누움 ①: 몸통이 카메라 방향으로 놓여 어깨 너비보다 짧게 찍힘 (머리 쪽·발 쪽 모두)
        if shoulder_w and torso < FORESHORTEN * shoulder_w:
            if low and drop <= 1.5 * shoulder_w:
                return finish('lying', '몸통이 카메라 방향으로 놓여 어깨 너비보다 짧게 보이고 다리도 낮게 있음')
            return finish('unknown', '상체를 카메라 쪽으로 숙였거나 다리가 안 보여 누움으로 보기 어려움')
        # 누움 ②: 옆에서 본 경우. 몸통이 가로이고, 몸 전체가 가로로 길거나 다리가 골반 높이 근처에 있음
        #         웅크려 엎드린 경우는 관절이 모여 가로로 길지 않으므로, 사람 박스가 가로로 넓은지를 함께 본다.
        if tilt >= 60 and legs and ((aspect >= 1.25 and w / h >= 1.1) or drop <= .6 * scale
                                    or (w / h >= 1.4 and drop <= 1.1 * scale)):
            return finish('lying', '몸통이 가로로 놓였고 몸 전체가 낮고 길게 보임')
        # 누움 ④: 반쯤 누운 자세. 몸통이 바닥에서 일어선 각도를 추정해 45도 이하이고 다리도 낮으면 누움으로 본다.
        #         (몸의 중심선과 바닥의 각도 45도 기준: Chen 외, Symmetry 2020)
        rise = torso_elevation(dx, dy, shoulder_w)
        result['metrics']['torso_elevation_degrees'] = round(rise, 1)
        if dy > 0 and rise <= LYING_ELEVATION and low and drop <= 1.1 * scale:
            return finish('lying', '몸통이 바닥에서 절반 이하로만 일어서 있고 다리도 낮게 있음 (비스듬히 누운 형태)')
        # 누움 ③: 머리가 카메라 쪽. 어깨가 골반보다 아래에 찍히고 다리도 골반 아래로 내려가지 않음
        if dy <= -.25 * torso:
            if low and drop <= .35 * torso:
                return finish('lying', '어깨가 골반보다 아래에 보이고 다리가 그 뒤로 이어짐 (머리가 카메라 쪽)')
            return finish('unknown', '상체가 골반보다 아래로 숙여짐')
        if tilt > 55 or dy <= 0:
            return finish('unknown', '몸통이 많이 기울었지만 누움 조건에는 못 미침')
        if not legs:
            return finish('unknown', '몸통은 보이나 무릎·발목이 안 보임')
        info, votes = [], []
        for leg in legs:
            kn, an = leg['knee'], leg['ankle']
            row = dict(side=leg['side'])
            if kn and an:
                leg_v = an[1] - hip[1] if leg['hip'] is None else an[1] - leg['hip'][1]
                thigh_v = kn[1] - leg['hip'][1]
                share = thigh_v / leg_v if leg_v > 1 else 0.
                row.update(hip_to_ankle_over_torso=round(leg_v / torso, 2), thigh_share=round(share, 2))
                if leg_v < FLOOR_RATIO * torso:
                    votes.append('floor')
                elif share >= STAND_SHARE and leg_v >= .8 * torso:
                    votes.append('stand')
                elif share <= SIT_SHARE:
                    votes.append('seat')
                else:
                    votes.append('?')
            elif kn:
                thigh_v = kn[1] - leg['hip'][1]
                row.update(knee_below_hip_over_torso=round(thigh_v / torso, 2))
                votes.append('seat_knee' if thigh_v <= .25 * torso else '?')
            else:
                leg_v = an[1] - leg['hip'][1]
                row.update(hip_to_ankle_over_torso=round(leg_v / torso, 2))
                votes.append('floor' if leg_v < FLOOR_RATIO * torso else '?')
            row['vote'] = votes[-1]
            info.append(row)
        result['metrics']['legs'] = info
        known = [v for v in votes if v != '?']
        # 한쪽 다리만 서 있음 조건이어도 서 있음으로 본다(걷는 중 한 발을 든 프레임).
        if 'stand' in votes and 'floor' not in votes:
            return finish('standing', '몸통이 세워져 있고 골반이 발목보다 충분히 높으며 무릎이 그 중간에 있음')
        # 발목까지 보이는 다리 중 하나라도 애매하면 앉음으로 단정하지 않는다.
        if '?' in votes:
            return finish('unknown', '다리 한쪽이 서 있음과 앉음의 중간 값이라 단정하지 않음')
        if known and all(v == 'floor' for v in known):
            return finish('sitting', '몸통은 세워져 있고 발목이 골반 높이 근처에 있음 (바닥에 앉은 형태)')
        if known and all(v in ('seat', 'seat_knee', 'floor') for v in known):
            return finish('sitting', '몸통은 세워져 있고 무릎이 골반 높이 근처에 있음 (의자·쪼그려 앉은 형태)')
        return finish('unknown', '몸통·다리는 보이나 세 자세 조건 어디에도 들지 않음')

    # ── 2. 어깨가 안 보이고 골반·다리만 보이는 경우 ───────────────────
    full = [l for l in legs if l['hip'] and l['knee'] and l['ankle']]
    if full:
        info, votes = [], []
        for leg in full:
            hi, kn, an = leg['hip'], leg['knee'], leg['ankle']
            leg_v = an[1] - hi[1]
            share = (kn[1] - hi[1]) / leg_v if leg_v > 1 else 0.
            shin = _tilt(kn, an)
            ok_size = math.dist(hi, an) >= max(8, .08 * max(w, h))
            vote = '?'
            if ok_size and share >= STAND_SHARE and shin <= 35 and leg_v > .6 * math.dist(hi, an):
                vote = 'stand'
            # 상체 없이 앉음은 옆모습(허벅지가 가로로, 정강이와 비슷한 길이로 보임)만 인정한다.
            # 상체가 잘리면 골반 위치가 허벅지 중간쯤으로 잘못 잡히는 일이 있어, 정면 앉음과 구분할 수 없다.
            elif (ok_size and share <= SIT_SHARE and an[1] > hi[1] and _tilt(hi, kn) >= 45
                  and math.dist(hi, kn) >= .6 * math.dist(kn, an)):
                vote = 'seat'
            votes.append(vote)
            info.append(dict(side=leg['side'], thigh_share=round(share, 2), shin_tilt=round(shin, 1), vote=vote))
        result['metrics'].update(basis='lower_body', legs=info)
        if 'stand' in votes:
            return finish('standing', '상체가 화면 밖 · 골반 아래로 다리가 길게 내려옴 (하체 기준)')
        if all(v == 'seat' for v in votes):
            return finish('sitting', '상체가 화면 밖 · 무릎이 골반 높이 근처에 있음 (하체 기준)')
        return finish('unknown', '어깨가 안 보이고 하체 조건도 불충족')

    # ── 3. 골반도 안 보이고 무릎·발목만 보이는 경우 (아주 가까이 선 다리) ──
    shins = [l for l in legs if l['knee'] and l['ankle']]
    if shins and not hip:
        info = []
        for leg in shins:
            kn, an = leg['knee'], leg['ankle']
            info.append(dict(side=leg['side'], shin_tilt=round(_tilt(kn, an), 1),
                             shin_over_box=round((an[1] - kn[1]) / h, 2),
                             above_knee_over_shin=round((kn[1] - box[1]) / max(an[1] - kn[1], 1), 2)))
        result['metrics'].update(basis='shin_only', legs=info)
        # 정강이가 세로로 길고, 무릎 위로도 박스가 이어져야 한다(정강이 길이의 0.3배 이상).
        good = [r for r in info if r['shin_tilt'] <= 30 and r['shin_over_box'] >= .3 and r['above_knee_over_shin'] >= .3]
        if shin_only and good and len(good) == len(info):
            return finish('standing', '가까워서 다리만 보임 · 정강이가 세로로 길게 서 있음 (골반 미확인, 약한 근거)')
        return finish('unknown', '다리만 보이나 서 있음 조건 불충족')

    # 자세를 못 본 것이 아니라 사람이 너무 가까운 경우다. 코드는 unknown 그대로 두고 표시 문구만 나눈다.
    result['metrics'].update(basis='none', too_close=True, touches_top=box[1] <= 3)
    reason = '너무 가깝거나 일부만 보여 몸통·다리 관절을 못 찾음'
    if box[1] <= 3:
        # 카메라가 바닥에서 약 30 cm 높이라, 화면 위쪽 끝까지 닿은 몸은 그보다 높이 올라와 있다.
        reason += ' · 화면 위쪽 끝까지 닿아 있어 바닥에 누운 자세로 보기는 어려움'
    finish('unknown', reason)
    result['label'] = TOO_CLOSE_LABEL
    return result


def draw_postures(image, detections):
    """Draw Korean labels with Pillow when a Korean font is available; ASCII otherwise."""
    from pathlib import Path
    import os
    fonts=[os.environ.get('K9_KOREAN_FONT',''),'/mnt/c/Windows/Fonts/malgun.ttf',
           'C:/Windows/Fonts/malgun.ttf','/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
           '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']
    font_path=next((f for f in fonts if f and Path(f).is_file()),None)
    labels=[(i,d) for i,d in enumerate(detections,1) if 'posture' in d]
    if not labels:return image
    for i,d in labels:
        if d.get('rotated'):   # 돌린 영상에서만 잡힌 사람은 모델이 상자를 그리지 않으므로 여기서 그린다
            import cv2
            x0,y0,x1,y1=[int(v) for v in d['xyxy']];cv2.rectangle(image,(x0,y0),(x1,y1),(123,152,250),2)
    colors={'standing':(117,215,159),'sitting':(246,203,103),'lying':(250,152,123),'unknown':(190,199,206)}
    if font_path:
        from PIL import Image,ImageDraw,ImageFont
        import numpy as np
        canvas=Image.fromarray(image[:,:,::-1]);draw=ImageDraw.Draw(canvas);font=ImageFont.truetype(font_path,18)
        for i,d in labels:
            p=d['posture'];x=int(max(0,min(image.shape[1]-1,d['xyxy'][0])));y=int(max(0,min(image.shape[0]-24,d['xyxy'][1]+24)))
            text=f"#{i} {p['label']}";bounds=draw.textbbox((x,y),text,font=font)
            draw.rectangle(bounds,fill=(12,18,23));draw.text((x,y),text,font=font,fill=colors[p['code']])
        return np.asarray(canvas)[:,:,::-1].copy()
    import cv2
    for i,d in labels:
        p=d['posture'];x=int(max(0,d['xyxy'][0]));y=int(max(20,min(image.shape[0]-5,d['xyxy'][1]+40)))
        cv2.putText(image,f"#{i} {p['code'].upper()}?",(x,y),cv2.FONT_HERSHEY_SIMPLEX,.6,colors[p['code']][::-1],2,cv2.LINE_AA)
    return image
