import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import rotate_detect as rd

class MapBack(unittest.TestCase):
    def test_round_trip(self):
        W,H=960,540
        # 원본 (x,y) -> 시계방향 90도 회전 영상 좌표는 (H-1-y, x), 반시계는 (y, W-1-x)
        for x,y in ((0,0),(100,50),(959,539),(480,270)):
            self.assertEqual(rd.map_back(H-1-y,x,'cw',W,H),(x,y))
            self.assertEqual(rd.map_back(y,W-1-x,'ccw',W,H),(x,y))
    def test_iou(self):
        self.assertEqual(rd._iou([0,0,10,10],[20,20,30,30]),0)
        self.assertAlmostEqual(rd._iou([0,0,10,10],[0,0,10,10]),1)

if __name__=='__main__':unittest.main()
