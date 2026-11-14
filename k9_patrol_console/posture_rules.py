"""Conservative, frame-local rules for COCO-17 image-space keypoints.
Labels describe posture candidates, never falls, consciousness, or health.
No identity tracking; detection indices are local to each frame.
"""
import math

LABELS = {'standing':'서 있음 후보', 'sitting':'앉아 있음 후보',
          'lying':'누워 있음 후보', 'unknown':'판단 불가'}
# Keypoint confidence, not calibrated posture probability.
KEYPOINT_MIN = .45


def angle(a, b, c):
    u=(a[0]-b[0],a[1]-b[1]);v=(c[0]-b[0],c[1]-b[1])
    length=math.hypot(*u)*math.hypot(*v)
    if length<1e-8:return None
    return math.degrees(math.acos(max(-1.,min(1.,(u[0]*v[0]+u[1]*v[1])/length))))


def classify_posture(keypoints, box, min_conf=KEYPOINT_MIN):
    result=dict(code='unknown',label=LABELS['unknown'],reason='',metrics={},
                method='coco17_rules_v1',keypoint_threshold=min_conf)
    def finish(code, reason):
        result.update(code=code,label=LABELS[code],reason=reason)
        return result
    if keypoints is None or len(keypoints)!=17 or box is None or len(box)!=4:
        return finish('unknown','COCO 17개 관절 좌표 필요')
    try:
        box=[float(v) for v in box]
        if not all(math.isfinite(v) for v in box):raise ValueError()
        w,h=box[2]-box[0],box[3]-box[1]
        if w<=0 or h<=0:raise ValueError()
    except (ValueError,TypeError):
        return finish('unknown','사람 박스 오류')
    points={}
    for i,p in enumerate(keypoints):
        try:
            x,y,c=map(float,p[:3])
            if all(math.isfinite(v) for v in (x,y,c)) and c>=min_conf and not (x==0 and y==0):
                points[i]=(x,y)
        except (TypeError,ValueError):pass
    if not all(i in points for i in (5,6,11,12)):
        return finish('unknown','양쪽 어깨·골반 관절 신뢰도 부족')
    shoulder=tuple((points[5][i]+points[6][i])/2 for i in (0,1))
    hip=tuple((points[11][i]+points[12][i])/2 for i in (0,1))
    dx,dy=hip[0]-shoulder[0],hip[1]-shoulder[1]
    torso=math.hypot(dx,dy)
    if torso<max(8,.08*max(w,h)):
        return finish('unknown','몸통이 너무 작거나 관절 겹침')
    tilt=math.degrees(math.atan2(abs(dx),abs(dy)))
    valid_body=[points[i] for i in (5,6,11,12,13,14,15,16) if i in points]
    span_x=max(p[0] for p in valid_body)-min(p[0] for p in valid_body)
    span_y=max(p[1] for p in valid_body)-min(p[1] for p in valid_body)
    aspect=span_x/max(span_y,1)
    result['metrics'].update(torso_tilt_degrees=round(tilt,1),box_aspect=round(w/h,2),body_aspect=round(aspect,2))
    legs=[]
    for side,shoulder_i,hip_i,knee_i,ankle_i in [('left',5,11,13,15),('right',6,12,14,16)]:
        if all(i in points for i in (shoulder_i,hip_i,knee_i,ankle_i)):
            sh,hi,kn,an=(points[i] for i in (shoulder_i,hip_i,knee_i,ankle_i))
            knee=angle(hi,kn,an);hip_angle=angle(sh,hi,kn)
            if knee is None or hip_angle is None:continue
            standing=(knee>=155 and hip_angle>=150 and kn[1]-hi[1]>.1*torso and an[1]-hi[1]>.6*math.dist(hi,an))
            thigh_tilt=math.degrees(math.atan2(abs(kn[1]-hi[1]),abs(kn[0]-hi[0])))
            sitting=(55<=knee<=140 and 45<=hip_angle<=140 and thigh_tilt<=35 and an[1]>hi[1])
            legs.append(dict(side=side,knee=round(knee,1),hip=round(hip_angle,1),standing=standing,sitting=sitting))
    result['metrics']['legs']=legs
    if tilt>=65 and w/h>=1.25 and aspect>=1.25 and legs:
        return finish('lying','몸통·하체가 영상에서 수평으로 길게 관찰됨')
    if dy<=0:
        return finish('unknown','어깨·골반 상하 관계 불명확')
    if not legs:
        return finish('unknown','무릎·발목 관절 부족')
    # Require agreement of all fully visible legs. Walking and ambiguous frames
    # remain unknown rather than forcing them into one of the three classes.
    if tilt<=30 and all(leg['standing'] for leg in legs):
        return finish('standing','몸통이 수직에 가깝고 다리가 펴짐')
    if tilt<=40 and all(leg['sitting'] for leg in legs):
        return finish('sitting','몸통이 세워지고 골반·무릎이 굽혀짐')
    return finish('unknown','구부림·이동·가림 등으로 규칙 조건 불충족')


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
