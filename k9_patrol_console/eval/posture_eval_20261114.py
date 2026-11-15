"""11/14 촬영 기록(data/records.sqlite3)에 저장된 관절 좌표로 현재 posture_rules.py 결과를 다시 계산한다.
정답은 저장된 사진을 한 장씩 보고 붙였다. part = 너무 가깝거나 몸 일부만 찍혀 판단할 수 없는 경우.
실행: python eval/posture_eval_20261114.py [-v]"""
import json,sqlite3,sys,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from posture_rules import classify_posture
T={}  # 11/14 사진을 보고 붙인 정답 (id, idx) -> 자세 / part = 너무 가깝거나 일부만
for k in 'c896ff3f ff73b542 b66d437d 9b016f3c 55c51694 0868ca6c 3b156d42 afe15ef8 a1418699 2f8f72dd 5b860392 c1fe8101 55b01bf2 d6e3213d 002bbf10 45f7ea57 ecf85b4a 8210c891 7df5216d 9db3d44f 21a32d9a 659b92eb 9e964034 a8f20f4b be14691a'.split(): T[(k,0)]='part'
T[('2a52ee57',0)]='part'
for k in '2a52ee57:1 bde5973b 272ea898 010177d8 b0a09b15 af7170ef 311e31e3 eedf4695 3fab5a0f e60c6e04 b0591be9 a2a79722 7999cb96 f943e000 11544d3e 4c74342d ef03d14d fdfda863 8dbe6f45:1 c2487b4c 1e6ac9a9'.split():
    a=k.split(':');T[(a[0],int(a[1]) if len(a)>1 else 0)]='standing'
for k in 'e1c01537 052c3978 0adbe814 763161e3 cd6cb459 8dbe6f45'.split():T[(k,0)]='sitting'
for k in 'db8004e3 a6aa678c 5154e432'.split():T[(k,0)]='lying'
T[('8d0e62f5',0)]='?'
db=sqlite3.connect(ROOT/'data'/'records.sqlite3')
tab=collections.defaultdict(collections.Counter)
for rid,t,meta in db.execute("select id,time,metadata from records where time like '2026-11-14%' order by time"):
    for i,d in enumerate(json.loads(meta or '{}').get('detections',[])):
        if 'keypoints' not in d or (rid[:8],i) not in T:continue
        truth=T[(rid[:8],i)];new=classify_posture(d['keypoints'],d['xyxy']);old=d.get('posture',{}).get('code','-')
        tab[('촬영 때',truth)][old]+=1;tab[('현재 규칙',truth)][new['code']]+=1
        if '-v' in sys.argv or (truth in('standing','sitting','lying') and new['code']!=truth):
            print(rid[:8],i,'정답',truth,'| 촬영 때',old,'| 현재',new['code'],'|',new['reason'])
for k in sorted(tab):print(k,dict(tab[k]))
