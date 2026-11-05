"""Manual robot microphone recording. Stop -> persist WAV -> local Korean STT.
Never starts recording on launch, reconnect, model change, or speaker pause.
"""
import asyncio
from datetime import datetime, timezone
import json
import queue
import threading
import time
import wave

# Bound an accidentally forgotten recording to ~19 MB of mono PCM.
MAX_SECONDS = 600


class SpeechRecorder:
    def __init__(self, store, robot, vision):
        self.store, self.robot, self.vision = store, robot, vision
        self.lock = threading.RLock()
        self.info = dict(recording=False, receiving=False, busy=False, message='녹음 대기',
                         error='', model='small', last_text='', queued=0, saved=0, elapsed_seconds=0,
                         max_seconds=MAX_SECONDS)
        self.config_path = store.root / 'speech_config.json'
        try:
            config=json.loads(self.config_path.read_text(encoding='utf-8'))
            if config.get('model') in ('base','small','medium'):self.info['model']=config['model']
        except (OSError,ValueError):pass
        self.conn = None
        self.generation = 0
        self.jobs = queue.Queue(maxsize=1)
        self.worker = None
        self.model = None
        self.model_name = None
        self.resampler = None
        self.callback = None
        self.pcm = bytearray()
        self.started = 0.
        self.started_at = ''
        self.last_received = 0.
        self.watchdog = None
        self.sample_error = ''
        for record in store.list():
            if record['metadata'].get('event_kind')=='speech' and record['state'] in ('queued','transcribing','recording'):
                meta=record['metadata'];meta['error']='이전 실행 중 처리 중단 · 저장된 원본 음성 확인 필요'
                store.finish(record['id'],record['clip'],meta,'error')

    def update(self, **kw):
        with self.lock:self.info.update(kw)

    def status(self):
        with self.lock:
            s=dict(self.info)
            s['queued']=self.jobs.qsize()
            s['elapsed_seconds']=round(time.monotonic()-self.started,1) if s['recording'] else 0
            s['receiving']=bool(s['recording'] and self.last_received and time.monotonic()-self.last_received<3)
            return s

    def configure(self, data):
        model=data.get('model','small')
        if model not in ('base','small','medium'):raise ValueError('STT 모델을 확인하세요.')
        if self.status()['recording'] or self.status()['busy']:raise ValueError('녹음·변환 종료 후 모델을 변경하세요.')
        self.config_path.write_text(json.dumps(dict(model=model,mode='manual')),encoding='utf-8')
        self.update(model=model,error='')

    async def attach(self, conn):
        self.conn=conn
        self.generation+=1
        generation=self.generation
        async def receive(frame):
            if generation!=self.generation or not self.info['recording']:return
            try:
                self.last_received=time.monotonic()
                for chunk in self.resampler.resample(frame):
                    remaining=MAX_SECONDS*32000-len(self.pcm)
                    if remaining>0:self.pcm.extend(bytes(chunk.planes[0])[:min(chunk.samples*2,remaining)])
            except Exception as exc:
                self.sample_error=f'마이크 처리 오류: {exc}'
                self.update(error=self.sample_error)
        self.callback=receive
        conn.audio.add_track_callback(receive)
        conn.audio.switchAudioChannel(False)
        self.update(recording=False,message='녹음 대기')

    def start(self):
        return asyncio.run_coroutine_threadsafe(self._start(),self.robot.loop).result(8)

    async def _start(self):
        if not self.conn or not self.robot.status()['connected']:raise ValueError('로봇을 먼저 연결하세요.')
        if self.info['recording']:raise ValueError('이미 녹음 중입니다.')
        if self.info['busy'] or self.jobs.unfinished_tasks:raise ValueError('이전 녹음의 저장·변환이 끝난 후 시작하세요.')
        from av.audio.resampler import AudioResampler
        self.resampler=AudioResampler(format='s16',layout='mono',rate=16000)
        self.pcm=bytearray();self.sample_error='';self.last_received=0.
        self.started=time.monotonic();self.started_at=datetime.now(timezone.utc).isoformat()
        self.update(recording=True,error='',message='녹음 중')
        try:self.conn.audio.switchAudioChannel(True)
        except Exception:
            self.update(recording=False,message='녹음 시작 실패');self.pcm.clear();raise
        self.watchdog=asyncio.create_task(self._watch())
        return {'ok':True}

    async def _watch(self):
        while self.info['recording']:
            await asyncio.sleep(.5)
            if time.monotonic()-self.started>=MAX_SECONDS or len(self.pcm)>=MAX_SECONDS*32000:
                await self._stop('최대 10분 도달');return
            if time.monotonic()-self.started>10 and (not self.last_received or time.monotonic()-self.last_received>10):
                await self._stop('마이크 수신 중단');return

    def stop(self):
        return asyncio.run_coroutine_threadsafe(self._stop('사용자 종료'),self.robot.loop).result(8)

    async def _stop(self, reason):
        if not self.info['recording']:raise ValueError('녹음 중이 아닙니다.')
        self.update(recording=False,busy=True)
        if self.watchdog and self.watchdog is not asyncio.current_task():
            self.watchdog.cancel();await asyncio.gather(self.watchdog,return_exceptions=True)
        self.watchdog=None
        try:
            if self.conn:self.conn.audio.switchAudioChannel(False)
        except Exception as exc:
            self.sample_error=f'오디오 종료 요청 실패: {exc}'
        try:
            for chunk in self.resampler.resample(None):
                remaining=MAX_SECONDS*32000-len(self.pcm)
                if remaining>0:self.pcm.extend(bytes(chunk.planes[0])[:min(chunk.samples*2,remaining)])
        except Exception as exc:self.sample_error=f'오디오 마무리 오류: {exc}'
        if not self.pcm:
            self.update(busy=False,message='녹음 대기',error='수신된 음성이 없습니다. 로봇 마이크 연결을 확인하세요.')
            self.resampler=None
            return {'ok':False}
        with self.vision.lock:
            image=self.vision.raw_jpeg if time.monotonic()-self.vision.info['last_frame']<2 else None
            session=self.vision.session
        meta=dict(event_kind='speech',source='robot_microphone',record_mode='manual',
                  started_at=self.started_at,ended_at=datetime.now(timezone.utc).isoformat(),
                  model=self.info['model'],duration_seconds=round(len(self.pcm)/32000,3),
                  stop_reason=reason,transcript='',segments=[],speaker='미확인',position_status='미수신',
                  timestamp_basis='PC reception estimate, not robot capture clock',snapshot_timing='녹음 종료 시점')
        if self.sample_error:meta['capture_warning']=self.sample_error
        # Capture is disabled before taking this immutable snapshot. No disk/model
        # work runs on the robot control loop.
        job=(bytes(self.pcm),image,session,meta)
        self.pcm.clear();self.resampler=None
        self.update(busy=True,message='음성 저장·STT 변환 중',error=self.sample_error)
        if self.worker is None:
            self.worker=threading.Thread(target=self._worker,name='manual-stt',daemon=True);self.worker.start()
        self.jobs.put_nowait(job)
        return {'ok':True}

    async def detach(self):
        if self.info['recording']:await self._stop('로봇 연결 해제')
        self.generation+=1
        if self.conn:
            try:
                self.conn.audio.switchAudioChannel(False)
                callbacks=getattr(self.conn.audio,'track_callbacks',[])
                if self.callback in callbacks:callbacks.remove(self.callback)
            except Exception:pass
        self.conn=None;self.callback=None
        self.update(recording=False,receiving=False)

    def suspend_for_playback(self, value):
        # Speaker playback must not silently stop or restart manual recording.
        if value and self.info['recording']:raise ValueError('안내 음성 재생 전에 녹음을 종료하세요.')

    def drain_saved_audio(self, timeout=5):
        deadline=time.monotonic()+timeout
        while self.jobs.unfinished_tasks and time.monotonic()<deadline:time.sleep(.05)

    def _process(self, pcm, image, session, meta):
        # Persist original WAV BEFORE importing/downloading/invoking STT.
        rid=self.store.create(session,'수동 음성 기록','manual',['speech'],image,meta)
        path=self.store.media/(rid+'.wav')
        try:
            with wave.open(str(path),'wb') as out:
                out.setnchannels(1);out.setsampwidth(2);out.setframerate(16000);out.writeframes(pcm)
            self.store.finish(rid,path.name,meta,'transcribing')
            self.update(saved=self.info['saved']+1,message='원본 저장 완료 · STT 변환 중')
            from faster_whisper import WhisperModel
            if self.model_name!=meta['model']:
                self.update(message=f"STT {meta['model']} 모델 준비 · 첫 실행 다운로드")
                self.model=None;self.model_name=None
                self.model=WhisperModel(meta['model'],device='cpu',compute_type='int8',cpu_threads=2)
                self.model_name=meta['model']
            segments,_=self.model.transcribe(str(path),language='ko',task='transcribe',
                                            beam_size=5,vad_filter=True,condition_on_previous_text=False)
            meta['segments']=[dict(start_seconds=round(s.start,3),end_seconds=round(s.end,3),text=s.text.strip())
                              for s in segments if s.text.strip()]
            meta['transcript']='\n'.join(s['text'] for s in meta['segments'])
            meta.update(review_required=True,note='수동 녹음의 자동 STT 결과. 원본 음성으로 확인하세요.')
            txt=self.store.media/(rid+'.txt');txt.write_text(meta['transcript'],encoding='utf-8-sig')
            meta['text_file']=txt.name
            (self.store.media/(rid+'.json')).write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
            self.store.finish(rid,path.name,meta,'ready')
            self.update(last_text=meta['transcript'] or '인식된 문장 없음',message='변환·저장 완료',error=meta.get('capture_warning',''))
        except Exception as exc:
            meta['error']=f'{type(exc).__name__}: {exc} · 저장된 원본 WAV는 유지됩니다.'
            self.store.finish(rid,path.name if path.exists() else '',meta,'error')
            raise

    def _worker(self):
        while True:
            job=self.jobs.get()
            try:self._process(*job)
            except Exception as exc:self.update(error=f'저장/STT 오류: {type(exc).__name__}: {exc}',message='기록의 원본·오류를 확인하세요.')
            finally:self.update(busy=False);self.jobs.task_done()
