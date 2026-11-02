import asyncio
import collections
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import uuid

TARGET_IDS={'person':0,'backpack':24,'handbag':26,'suitcase':28}


class Vision:
    def __init__(self, store):
        self.store=store
        self.lock=threading.RLock()
        self.session=uuid.uuid4().hex
        self.buffer=collections.deque()
        self.pending=[]
        self.latest=None
        self.serial=0
        self.jpeg=None
        self.raw_jpeg=None
        self.running=False
        self.tracks=set()
        self.model=None
        self.capture_tick=0.
        self.last_seen={}
        self.start_seen={}
        self.last_saved={}
        self.config=dict(model=os.environ.get('K9_YOLO_MODEL','yolo11n-pose.pt'),targets=['person'],confidence=.45,auto=True,cooldown=10,pre=3,post=3)
        self.info=dict(video=False,detecting=False,detector_error='',video_error='',record_error='',detections=[],last_frame=0.,model_path=self.config['model'])
        self.loop=None
        self.tasks=[]

    def status(self):
        with self.lock:
            s=dict(self.info)
            s['video']=bool(s['video'] and time.monotonic()-s['last_frame']<2)
            s['config']=dict(self.config)
            s['session']=self.session
            return s

    def configure(self, data):
        model=data.get('model',self.config['model'])
        if model not in ('yolo11n-pose.pt','yolo11s-pose.pt','yolo11n.pt'):raise ValueError('지원하는 탐지 모델을 선택하세요.')
        targets=data.get('targets',['person'])
        if '-pose' in model:targets=['person']
        if not isinstance(targets,list) or not targets or any(t not in TARGET_IDS for t in targets):raise ValueError('관심 대상을 하나 이상 선택하세요.')
        conf=float(data.get('confidence',.45));cool=float(data.get('cooldown',10))
        if not .1<=conf<=.95 or not 3<=cool<=120:raise ValueError('설정 범위를 확인하세요.')
        if not isinstance(data.get('auto',True),bool):raise ValueError('자동 저장 설정 오류')
        with self.lock:
            self.config.update(model=model,targets=targets,confidence=conf,cooldown=cool,auto=data.get('auto',True))
            self.last_seen.clear();self.start_seen.clear()
        if self.running and self.loop:self.loop.call_soon_threadsafe(self.restart_detector)

    def restart_detector(self):
        if self.running and self.tasks and self.tasks[0].done():
            self.tasks[0]=asyncio.create_task(self.detect())

    def attach(self, conn):
        self.loop=asyncio.get_running_loop()
        self.running=True
        self.session=uuid.uuid4().hex
        self.buffer.clear();self.last_seen.clear();self.start_seen.clear();self.last_saved.clear()
        self.info.update(video=False,video_error='',record_error='',detector_error='',detections=[])
        conn.video.add_track_callback(self.receive)
        conn.video.switchVideoChannel(True)
        self.tasks=[asyncio.create_task(self.detect()),asyncio.create_task(self.finalize_loop())]

    def encode(self, image):
        import cv2
        ok,jpg=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,78])
        if not ok:raise RuntimeError('JPEG 인코딩 실패')
        return jpg.tobytes()

    async def receive(self, track):
        self.tracks.add(asyncio.current_task())
        try:
            while self.running:
                frame=await track.recv()
                now=time.monotonic()
                if now-self.capture_tick<.1:continue
                self.capture_tick=now
                image=frame.to_ndarray(format='bgr24')
                import cv2
                h,w=image.shape[:2]
                if w>960:image=cv2.resize(image,(960,int(h*960/w)))
                jpg=await asyncio.to_thread(self.encode,image)
                with self.lock:
                    self.serial+=1
                    self.latest=(self.serial,image)
                    self.raw_jpeg=jpg
                    if not self.info['detecting']:self.jpeg=jpg
                    self.info.update(video=True,last_frame=now)
                    self.buffer.append((now,jpg))
                    while self.buffer and self.buffer[0][0]<now-4:self.buffer.popleft()
                    # Keep a bounded pre-roll buffer even if JPEG sizes grow.
                    while len(self.buffer)>45:self.buffer.popleft()
                    for job in self.pending:
                        if now<=job['end'] and now>job['frames'][-1][0]:job['frames'].append((now,jpg))
        except asyncio.CancelledError:raise
        except Exception as exc:
            self.info.update(video=False,video_error=f'{type(exc).__name__}: {exc}')
        finally:self.tracks.discard(asyncio.current_task())

    def infer(self,image,config):
        import cv2
        results=self.model.predict(image,conf=config['confidence'],classes=[TARGET_IDS[t] for t in config['targets']],imgsz=512,verbose=False)
        r=results[0];detections=[]
        if r.boxes is not None:
            for i,b in enumerate(r.boxes):
                item=dict(target=r.names[int(b.cls[0])],confidence=float(b.conf[0]),xyxy=[float(v) for v in b.xyxy[0].tolist()])
                if r.keypoints is not None and len(r.keypoints.data)>i:
                    item['keypoints']=r.keypoints.data[i].tolist()
                    item['pose_status']='관절 추정 · 쓰러짐 여부 미판정'
                detections.append(item)
        # Annotations belong to this analyzed frame, not an unrelated newer frame.
        return detections,self.encode(r.plot())

    async def detect(self):
        try:
            from ultralytics import YOLO
            path=self.config['model']
            self.info['model_path']=path
            self.model=await asyncio.to_thread(YOLO,path)
            self.info['detecting']=True
            previous=-1
            while self.running:
                with self.lock:latest=self.latest;config=dict(self.config)
                if config['model']!=path:
                    path=config['model']
                    self.info.update(detecting=False,model_path=path,detector_error='')
                    self.model=None
                    self.model=await asyncio.to_thread(YOLO,path)
                    self.info['detecting']=True
                    previous=-1
                if latest is None or latest[0]==previous:
                    await asyncio.sleep(.03);continue
                previous,image=latest
                det,jpg=await asyncio.to_thread(self.infer,image,config)
                now=time.monotonic()
                with self.lock:self.jpeg=jpg;self.info['detections']=det
                seen={d['target'] for d in det}
                for target in config['targets']:
                    if target in seen:
                        if now-self.last_seen.get(target,-1e10)>2:self.start_seen[target]=now
                        self.last_seen[target]=now
                        if config['auto'] and now-self.start_seen.get(target,now)>=.25 and now-self.last_saved.get(target,-1e10)>=config['cooldown']:
                            # One event per class presence, rearm after >=2 s absence.
                            if self.start_seen.get(target)==-1:continue
                            try:
                                self.capture('auto',target+' 등장', [target],det)
                                self.last_saved[target]=now;self.start_seen[target]=-1
                            except Exception as exc:self.info['record_error']=str(exc)
                await asyncio.sleep(.01)
        except asyncio.CancelledError:raise
        except Exception as exc:self.info.update(detector_error=f'{type(exc).__name__}: {exc}',detecting=False)
        finally:self.info['detecting']=False

    def capture(self,kind='manual',title='수동 촬영',targets=None,detections=None):
        with self.lock:
            if self.raw_jpeg is None or time.monotonic()-self.info['last_frame']>2:raise ValueError('최신 영상이 없습니다. 로봇 영상 연결을 먼저 확인하세요.')
            if len(self.pending)>=4:raise ValueError('영상 저장 처리 중입니다. 잠시 후 다시 시도하세요.')
            now=time.monotonic();frames=[f for f in self.buffer if f[0]>=now-self.config['pre']]
            if not frames:frames=[(now,self.raw_jpeg)]
            targets=targets or sorted({d['target'] for d in self.info['detections']})
            meta=dict(requested_pre_seconds=self.config['pre'],requested_post_seconds=self.config['post'],detections=detections or list(self.info['detections']),position=None,position_status='미수신',event_kind='class_appearance' if kind=='auto' else 'manual',tracking='클래스 단위 등장, 개별 인물 식별·추적 아님')
            rid=self.store.create(self.session,title,kind,targets,self.raw_jpeg,meta)
            self.pending.append(dict(id=rid,at=now,end=now+self.config['post'],frames=frames,meta=meta))
            return rid

    def write_clip(self,job,interrupted=False):
        import cv2
        import numpy as np
        frames=job['frames'];meta=job['meta'];rid=job['id']
        meta.update(actual_pre_seconds=round(max(0,job['at']-frames[0][0]),2),actual_post_seconds=round(max(0,frames[-1][0]-job['at']),2),frame_count=len(frames),interrupted=interrupted)
        avi=self.store.media/(rid+'.avi')
        try:
            first=cv2.imdecode(np.frombuffer(frames[0][1],np.uint8),cv2.IMREAD_COLOR)
            if first is None:raise RuntimeError('저장 프레임 해독 실패')
            h,w=first.shape[:2];writer=cv2.VideoWriter(str(avi),cv2.VideoWriter_fourcc(*'MJPG'),10,(w,h))
            if not writer.isOpened():raise RuntimeError('영상 인코더를 열 수 없습니다.')
            try:
                # Resample by arrival time to maintain clip duration when frames drop.
                idx=0;stamp=frames[0][0];last=frames[-1][0]
                while stamp<=last+.001:
                    while idx+1<len(frames) and frames[idx+1][0]<=stamp:idx+=1
                    im=cv2.imdecode(np.frombuffer(frames[idx][1],np.uint8),cv2.IMREAD_COLOR)
                    if im is not None:writer.write(cv2.resize(im,(w,h)))
                    stamp+=.1
            finally:writer.release()
            clip=avi.name
            if shutil.which('ffmpeg'):
                mp4=avi.with_suffix('.mp4')
                proc=subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(avi),'-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(mp4)],capture_output=True,timeout=45)
                if proc.returncode==0:clip=mp4.name;avi.unlink(missing_ok=True)
                else:meta['conversion_warning']='MP4 변환 실패: AVI 다운로드 사용';mp4.unlink(missing_ok=True)
            else:meta['conversion_warning']='ffmpeg 미설치: AVI 다운로드 사용'
            meta['pre_roll_complete']=meta['actual_pre_seconds']>=meta['requested_pre_seconds']-.3
            meta['post_roll_complete']=meta['actual_post_seconds']>=meta['requested_post_seconds']-.3
            self.store.finish(rid,clip,meta,'partial' if interrupted or not meta['pre_roll_complete'] or not meta['post_roll_complete'] else 'ready')
        except Exception as exc:
            meta['error']=f'{type(exc).__name__}: {exc}'
            self.store.finish(rid,'',meta,'error')
            self.info['record_error']=meta['error']

    async def finalize_loop(self):
        while self.running:
            with self.lock:
                now=time.monotonic();ready=[j for j in self.pending if j['end']<=now]
                self.pending=[j for j in self.pending if j['end']>now]
            for job in ready:await asyncio.to_thread(self.write_clip,job)
            await asyncio.sleep(.1)

    async def close(self):
        self.running=False
        tasks=self.tasks+list(self.tracks)
        for task in tasks:task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        self.tasks=[]
        with self.lock:jobs=self.pending;self.pending=[]
        for job in jobs:await asyncio.to_thread(self.write_clip,job,True)
        self.info.update(video=False,detecting=False,detections=[])
        self.latest=None;self.jpeg=None;self.raw_jpeg=None
