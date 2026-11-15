"""Go2 카메라(높이 0.30 m, 수평)에서 본 사람 관절을 기하학적으로 만들어 규칙을 점검한다.
한계: 관절이 모두 정확히 잡힌다고 가정(가림·오검출 없음). 실제 영상 검증을 대신하지 못한다."""
import math,sys,collections,importlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
W,H,CAM=960,540,0.30
S,HP=.18,.10
def P(sh,hip,knL,anL,knR=None,anR=None,head=None):
    """각 점 (f 앞, l 왼쪽, u 위). sh/hip 은 중심, 좌우 폭은 자동."""
    knR=knR or (knL[0],-knL[1],knL[2]);anR=anR or (anL[0],-anL[1],anL[2])
    return {5:(sh[0],S,sh[2]),6:(sh[0],-S,sh[2]),11:(hip[0],HP,hip[2]),12:(hip[0],-HP,hip[2]),13:knL,14:knR,15:anL,16:anR,
            0:head or (sh[0],0,sh[2]+.22)}
POSES={
 'standing':{
  '바로 섬':P((0,0,1.40),(0,0,.90),(0,.1,.50),(0,.1,.08)),
  '걷는 중':P((0,0,1.40),(0,0,.90),(.15,.1,.52),(.25,.1,.08),(-.05,-.1,.50),(-.25,-.1,.15)),
  '한 발 들기':P((0,0,1.40),(0,0,.90),(.30,.1,.80),(.30,.1,.40),(0,-.1,.50),(0,-.1,.08)),
  '벽에 기대 섬':P((-.12,0,1.38),(0,0,.90),(.08,.1,.50),(.15,.1,.08)),
 },
 'sitting':{
  '의자':P((0,0,.95),(0,0,.45),(.40,.1,.47),(.42,.1,.08)),
  '의자·다리 뻗음':P((-.08,0,.93),(0,0,.45),(.38,.1,.40),(.75,.1,.08)),
  '쪼그려':P((.15,0,.70),(0,0,.25),(.30,.12,.45),(.10,.1,.08)),
  '바닥·다리 뻗음':P((-.05,0,.60),(0,0,.10),(.40,.1,.12),(.82,.1,.08)),
  '바닥·무릎 세움':P((0,0,.60),(0,0,.10),(.25,.1,.45),(.40,.1,.08)),
  '양반다리':P((.05,0,.60),(0,0,.10),(.25,.35,.15),(.30,-.05,.12),(.25,-.35,.15),(.30,.05,.12)),
  '벽에 기대 앉음':P((-.15,0,.58),(0,0,.10),(.40,.1,.12),(.82,.1,.08)),
 },
 'lying':{
  '바로 누움':P((-.50,0,.12),(0,0,.12),(.40,.1,.12),(.82,.1,.08),head=(-.72,0,.12)),
  '누움·무릎 세움':P((-.50,0,.12),(0,0,.12),(.25,.1,.42),(.45,.1,.08),head=(-.72,0,.12)),
  '옆으로 웅크림':{5:(-.48,.05,.44),6:(-.48,.05,.08),11:(0,0,.28),12:(0,0,.08),13:(.25,.30,.25),14:(.25,.30,.08),15:(0,.55,.20),16:(0,.55,.08),0:(-.68,.12,.15)},
  '비스듬히 기대 누움':P((-.42,0,.36),(0,0,.12),(.40,.1,.12),(.82,.1,.08),head=(-.60,0,.48)),
  '옆으로 누워 팔 괴기':{5:(-.40,.0,.50),6:(-.46,.0,.16),11:(0,0,.30),12:(0,0,.10),13:(.35,.15,.22),14:(.38,.05,.08),15:(.70,.25,.12),16:(.78,.05,.08),0:(-.55,0,.62)},
  '엎드림':P((-.50,0,.10),(0,0,.10),(.42,.1,.06),(.84,.1,.06),head=(-.70,0,.15)),
 },
 'unknown':{
  '서서 허리 숙임':P((.48,0,.98),(0,0,.90),(0,.1,.50),(0,.1,.08),head=(.62,0,.85)),
  '무릎 꿇고 엎드림':P((.45,0,.40),(0,0,.45),(.05,.1,.05),(-.38,.1,.05),head=(.62,0,.30)),
 }}
def project(pose,yaw,D,f):
    c,s=math.cos(math.radians(yaw)),math.sin(math.radians(yaw))
    kp=[[0,0,0] for _ in range(17)];xs=[];ys=[]
    for i,(a,l,u) in pose.items():
        X=a*s+l*c; Z=D+(-a*c+l*s)   # yaw 0 = 카메라를 정면으로 봄(앞쪽이 카메라 방향)
        if Z<.15:continue
        x=W/2+f*X/Z; y=H/2-f*(u-CAM)/Z
        xs.append(min(max(x,0),W));ys.append(min(max(y,0),H))
        if 0<=x<=W and 0<=y<=H and i!=0: kp[i]=[x,y,.9]
    if not xs:return None,None
    return kp,[min(xs)-8,max(0,min(ys)-8),max(xs)+8,min(H,max(ys)+8)]
def run(mod,verbose=False):
    m=importlib.import_module(mod);tab=collections.defaultdict(collections.Counter);bad=collections.defaultdict(list)
    for truth,ps in POSES.items():
        for name,pose in ps.items():
            for f in (350,500):
                for D in (1.5,2,3,4,6):
                    for yaw in range(0,360,30):
                        kp,box=project(pose,yaw,D,f)
                        if kp is None:continue
                        r=m.classify_posture(kp,box)
                        tab[(truth,name)][r['code']]+=1
                        if r['code']!=truth: bad[(name,r['code'],r['reason'])].append((f,D,yaw))
    tot=collections.Counter()
    for (t,n),c in tab.items():
        N=sum(c.values());wrong=N-c[t]-c['unknown']
        print(f"{t:9s} {n:12s} 맞음 {c[t]:3d}/{N}  판단불가 {c['unknown'] if t!='unknown' else 0:3d}  다른자세 {wrong:3d}  {dict(c)}")
        tot['n']+=N;tot['ok']+=c[t];tot['wrong']+=wrong
    print(tot)
    if verbose:
        for k,v in sorted(bad.items(),key=lambda kv:-len(kv[1])):print(len(v),k,v[:6])
if __name__=='__main__':run('posture_rules','-v' in sys.argv)
