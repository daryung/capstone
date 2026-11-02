import asyncio
import json
import threading
import time


class Controller:
    def __init__(self,vision):
        self.vision=vision
        self.lock=threading.Lock()
        self.info=dict(connected=False,connecting=False,mode='',error='',command='정지',ip='')
        self.vector=(0.,0.,0.);self.deadline=0.;self.sequence=-1
        self.conn=None;self.commands=None;self.pose_busy=False;self.audio=None;self.speech=None
        self.loop=asyncio.new_event_loop()
        threading.Thread(target=self.run,daemon=True).start()

    def run(self):
        asyncio.set_event_loop(self.loop);self.loop.create_task(self.pump());self.loop.run_forever()

    def update(self,**kw):
        with self.lock:self.info.update(kw)

    def status(self):
        with self.lock:return dict(self.info,pose_busy=self.pose_busy)

    def command(self,vector,sequence):
        with self.lock:
            if sequence<=self.sequence:return False
            self.sequence=sequence
            if not self.info['connected'] or self.pose_busy:return False
            self.vector=tuple(vector);self.deadline=time.monotonic()+.4
            return True

    def stop(self):
        with self.lock:self.vector=(0.,0.,0.);self.deadline=0.

    def send(self,vector):
        from unitree_webrtc_connect.constants import RTC_TOPIC,DATA_CHANNEL_TYPE
        pub=self.conn.datachannel.pub_sub
        if pub.channel.readyState!='open':raise ConnectionError('DataChannel 닫힘: 정지 전달을 확인할 수 없습니다.')
        moving=any(vector)
        payload={'header':{'identity':{'id':time.time_ns()%2147483647,'api_id':self.commands['Move' if moving else 'StopMove']}},'parameter':json.dumps(dict(zip(('x','y','z'),vector))) if moving else ''}
        pub.publish_without_callback(RTC_TOPIC['SPORT_MOD'],payload,DATA_CHANNEL_TYPE['REQUEST'])

    async def pump(self):
        while True:
            if self.status()['connected'] and not self.pose_busy:
                with self.lock:v=self.vector if time.monotonic()<self.deadline else (0.,0.,0.)
                try:
                    self.send(v);self.update(command='이동 명령 전송' if any(v) else '정지 요청')
                except Exception as exc:
                    self.stop();self.update(connected=False,error=f'{exc} / 리모컨 정지를 확인하세요.')
            await asyncio.sleep(.05)

    def begin_connect(self,ip):
        with self.lock:
            if self.info['connecting'] or self.info['connected']:return False
            self.info.update(connecting=True,error='',ip=ip)
        asyncio.run_coroutine_threadsafe(self._connect(ip),self.loop)
        return True

    async def _connect(self,ip):
        try:
            await self._cleanup();self.update(connecting=True)
            from robot.connection import Go2Connection
            from unitree_webrtc_connect import constants
            wrapper=Go2Connection(ip=ip)
            try:self.conn=await asyncio.wait_for(wrapper.connect(),30)
            except BaseException:self.conn=wrapper.conn;raise
            r=await asyncio.wait_for(self.conn.datachannel.pub_sub.publish_request_new(constants.RTC_TOPIC['MOTION_SWITCHER'],{'api_id':1001}),5)
            if r['data']['header']['status']['code']!=0:raise RuntimeError('제어 모드 조회 실패')
            data=r['data']['data'];mode=(json.loads(data) if isinstance(data,str) else data)['name']
            self.commands=getattr(constants,'SPORT_CMD_MCF',None) if mode=='mcf' else constants.SPORT_CMD if mode=='normal' else None
            if self.commands is None:raise RuntimeError(f'지원하지 않는 모드/명령표: {mode}')
            self.stop();self.vision.attach(self.conn)
            self.update(connected=True,connecting=False,mode=mode,error='')
            if self.speech:await self.speech.attach(self.conn)
        except Exception as exc:
            await self._cleanup();self.update(connecting=False,connected=False,error=f'{type(exc).__name__}: {exc}')

    async def _cleanup(self):
        if self.speech:await self.speech.detach()
        if self.audio:self.audio.cancel()
        self.stop();self.update(connected=False)
        if self.conn is not None:
            try:
                if self.commands:
                    for _ in range(3):self.send((0.,0.,0.));await asyncio.sleep(.03)
            except Exception:pass
        await self.vision.close()
        if self.conn is not None:
            try:await asyncio.wait_for(self.conn.disconnect(),3)
            except Exception as exc:self.update(error=f'연결 해제 확인 필요: {exc}')
        self.conn=None;self.commands=None
        self.update(mode='',command='정지')

    def disconnect(self):
        if self.status()['connecting']:raise ValueError('연결 진행 중입니다. 완료 후 다시 시도하세요.')
        self.update(connecting=True)
        try:asyncio.run_coroutine_threadsafe(self._cleanup(),self.loop).result(60)
        finally:self.update(connecting=False)

    def capture(self):
        async def do():return self.vision.capture()
        return asyncio.run_coroutine_threadsafe(do(),self.loop).result(5)

    def pose(self,name):
        if name not in ('StandUp','StandDown'):raise ValueError('지원하지 않는 자세')
        if not self.status()['connected'] or self.pose_busy:raise ValueError('연결/자세 전환 상태를 확인하세요.')
        self.stop();self.pose_busy=True
        async def do():
            try:
                self.send((0.,0.,0.))
                from unitree_webrtc_connect.constants import RTC_TOPIC
                r=await asyncio.wait_for(self.conn.datachannel.pub_sub.publish_request_new(RTC_TOPIC['SPORT_MOD'],{'api_id':self.commands[name]}),5)
                if r['data']['header']['status']['code']!=0:raise RuntimeError('자세 명령 거부')
                await asyncio.sleep(2)
            except Exception as exc:self.update(error=f'자세 명령: {exc}')
            finally:self.stop();self.pose_busy=False
        asyncio.run_coroutine_threadsafe(do(),self.loop)
