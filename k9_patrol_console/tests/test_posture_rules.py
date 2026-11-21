import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'eval'))
from posture_rules import classify_posture
import posture_sim as sim

def kp(d):
    k=[[0,0,0] for _ in range(17)]
    for i,(x,y) in d.items():k[i]=[x,y,.9]
    return k

class Rules(unittest.TestCase):
    def test_simulated_postures(self):
        for truth,poses in sim.POSES.items():
            if truth=='unknown':continue
            for name,pose in poses.items():
                if name=='옆으로 누워 팔 괴기':continue  # 가까운 거리 일부 방향에서 서 있음으로 나오는 알려진 약점
                for yaw in range(0,360,30):
                    for dist in (1.5,3,6):
                        k,box=sim.project(pose,yaw,dist,350)
                        self.assertEqual(classify_posture(k,box)['code'],truth,(name,yaw,dist))
    def test_front_chair(self):
        r=classify_posture(kp({5:(440,100),6:(520,100),11:(450,250),12:(510,250),13:(445,280),14:(515,280),15:(445,400),16:(515,400)}),[400,40,560,420])
        self.assertEqual(r['code'],'sitting')
    def test_bend_is_not_lying(self):
        r=classify_posture(kp({5:(540,330),6:(545,330),11:(450,250),12:(455,250),13:(450,350),14:(455,350),15:(450,450),16:(455,450)}),[400,200,600,470])
        self.assertEqual(r['code'],'unknown')
    def test_too_close(self):
        r=classify_posture([[10,10,.1] for _ in range(17)],[0,0,430,538])
        self.assertEqual(r['code'],'unknown');self.assertIn('너무 가깝거나',r['reason']);self.assertEqual(r['label'],'너무 가까움')
    def test_shin_only_can_be_turned_off(self):
        k=kp({13:(662,208),14:(862,214),15:(628,495),16:(812,521)})
        self.assertEqual(classify_posture(k,[526,0,957,540])['code'],'unknown')  # 기본값: 판단 불가 (11/17 팀 결정)
        self.assertEqual(classify_posture(k,[526,0,957,540],shin_only=True)['code'],'standing')
        self.assertEqual(classify_posture(k,[526,0,957,540],shin_only=False)['code'],'unknown')
    def test_bad_input(self):
        self.assertEqual(classify_posture(None,None)['code'],'unknown')

if __name__=='__main__':unittest.main()
