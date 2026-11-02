"""Every 30 seconds, listen for 5 seconds; save only STT keyword matches.
Nonmatching audio and text stay in memory and are discarded, not written to disk.
"""
import asyncio
from datetime import datetime, timezone
import json
import queue
import re
import threading
import time
import unicodedata
import wave

INTERVAL = 30
WINDOW = 5
KEYWORD = '살려주세요'


def keyword_matches(text):
    normalized = re.sub(r'[\W_]+', '', unicodedata.normalize('NFKC', text))
    return KEYWORD in normalized


class SpeechRecorder:
    def __init__(self, store, robot, vision):
        self.store, self.robot, self.vision = store, robot, vision
        self.lock = threading.RLock()
        self.info = dict(enabled=False, receiving=False, listening=False, busy=False,
                         message='꺼짐', error='', model='small', last_text='',
                         queued=0, saved=0, dropped=0, interval=INTERVAL,
                         window_seconds=WINDOW, keyword=KEYWORD, checked=0)
        self.config_path = store.root / 'speech_config.json'
        try:
            saved = json.loads(self.config_path.read_text(encoding='utf-8'))
            if saved.get('model') in ('base', 'small', 'medium'):
                self.info['model'] = saved['model']
            self.info['enabled'] = saved.get('enabled') is True
        except (OSError, ValueError):
            pass
        self.conn = None
        self.generation = 0
        self.jobs = queue.Queue(maxsize=2)
        self.worker = None
        self.model = None
        self.model_name = None
        self.resampler = None
        self.callback = None
        self.last_received = 0.
        self.suspended = False
        self.schedule = None
        self.next_listen = 0.
        self.pcm = bytearray()
        self.accepting = False
        self.sample_error = ''

    def status(self):
        with self.lock:
            s = dict(self.info)
            s['queued'] = self.jobs.qsize()
            s['receiving'] = bool(self.accepting and time.monotonic()-self.last_received < 2)
            s['listening'] = self.accepting
            s['next_seconds'] = max(0, int(self.next_listen-time.monotonic()+.999)) if self.next_listen else 0
            if s['enabled'] and not self.conn:
                s['message'] = '로봇 연결 대기'
            elif self.suspended:
                s['message'] = '스피커 재생 후 일시중지 · 다시 듣기 필요'
            elif s['enabled'] and self.conn:
                s['message'] = '5초 듣는 중' if self.accepting else f"다음 듣기 {s['next_seconds']}초 후"
            return s

    def update(self, **kw):
        with self.lock:
            self.info.update(kw)

    def configure(self, data):
        enabled, model = data.get('enabled'), data.get('model', 'small')
        if not isinstance(enabled, bool) or model not in ('base', 'small', 'medium'):
            raise ValueError('음성 기록 설정을 확인하세요.')
        async def apply():
            await self._stop_capture()
            self.update(enabled=enabled, model=model, error='', message='연결 대기' if enabled else '꺼짐')
            if enabled and self.robot.conn and self.robot.status()['connected']:
                await self.attach(self.robot.conn)
            self.config_path.write_text(json.dumps(dict(enabled=enabled, model=model, interval=INTERVAL,
                                                        window_seconds=WINDOW, keyword=KEYWORD)), encoding='utf-8')
        asyncio.run_coroutine_threadsafe(apply(), self.robot.loop).result(8)

    async def attach(self, conn):
        self.conn = conn
        if not self.info['enabled']:
            return
        try:
            from av.audio.resampler import AudioResampler
            # Import check is light; actual model load stays on a worker.
            if self.worker is None:
                self.worker = threading.Thread(target=self._worker, name='speech-stt', daemon=True)
                self.worker.start()
            self.generation += 1
            generation = self.generation
            self.suspended = False
            async def receive(frame):
                if generation != self.generation or not self.accepting or self.suspended:
                    return
                try:
                    self.last_received = time.monotonic()
                    for chunk in self.resampler.resample(frame):
                        remaining = int(WINDOW*32000)-len(self.pcm)
                        if remaining > 0:
                            self.pcm.extend(bytes(chunk.planes[0])[:min(chunk.samples*2, remaining)])
                except Exception as exc:
                    self.sample_error = f'마이크 처리 오류: {exc}'
            self.callback = receive
            conn.audio.add_track_callback(receive)
            conn.audio.switchAudioChannel(False)
            self.schedule = asyncio.create_task(self._listen_loop(generation))
            self.update(error='', message='30초 주기 듣기 시작')
        except Exception as exc:
            self.update(error=f'음성 기록 시작 실패: {exc}. requirements.txt를 설치하세요.')

    async def _listen_loop(self, generation):
        try:
            while generation == self.generation:
                if self.suspended:
                    self.next_listen = 0.
                    await asyncio.sleep(.2)
                    continue
                started = time.monotonic()
                self.next_listen = started + INTERVAL
                await self._listen_once(generation)
                await asyncio.sleep(max(0, self.next_listen-time.monotonic()))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.update(error=f'주기적 듣기 실패: {exc}. 음성 설정을 다시 적용하세요.')
        finally:
            self.accepting = False
            self.pcm.clear()
            if self.conn:
                try:self.conn.audio.switchAudioChannel(False)
                except Exception:pass

    async def _listen_once(self, generation):
        from av.audio.resampler import AudioResampler
        self.resampler = AudioResampler(format='s16', layout='mono', rate=16000)
        self.pcm = bytearray()
        self.sample_error = ''
        self.last_received = 0.
        started_at = datetime.now(timezone.utc).isoformat()
        self.accepting = True
        try:
            self.conn.audio.switchAudioChannel(True)
            await asyncio.sleep(WINDOW)
        finally:
            self.accepting = False
            try:self.conn.audio.switchAudioChannel(False)
            except Exception:pass
        if generation != self.generation or self.suspended:
            self.pcm.clear()
            return
        if self.sample_error:
            self.update(error=self.sample_error)
            self.pcm.clear()
            return
        if len(self.pcm) < 6400:
            self.update(error='5초 듣기 구간에 충분한 오디오가 수신되지 않았습니다.')
            self.pcm.clear()
            return
        # Flush resampling tail, retaining the strict 5-second memory bound.
        for chunk in self.resampler.resample(None):
            remaining=int(WINDOW*32000)-len(self.pcm)
            if remaining > 0:self.pcm.extend(bytes(chunk.planes[0])[:min(chunk.samples*2, remaining)])
        with self.vision.lock:
            image = self.vision.raw_jpeg if time.monotonic()-self.vision.info['last_frame'] < 2 else None
            session = self.vision.session
        meta = dict(event_kind='speech', source='robot_microphone', started_at=started_at,
                    ended_at=datetime.now(timezone.utc).isoformat(), model=self.info['model'],
                    duration_seconds=round(len(self.pcm)/32000,3), interval_seconds=INTERVAL,
                    listening_window_seconds=WINDOW, trigger_keyword=KEYWORD,
                    timestamp_basis='PC reception estimate, not robot capture clock',
                    speaker='미확인', position_status='미수신', snapshot_timing='듣기 구간 종료 시점')
        try:
            self.jobs.put_nowait((bytes(self.pcm), image, session, meta))
            self.update(error='')
        except queue.Full:
            self.update(error='STT 처리 지연으로 이번 듣기 구간을 건너뛰었습니다.', dropped=self.info['dropped']+1)
        finally:
            self.pcm.clear()

    async def _stop_capture(self):
        self.generation += 1
        self.accepting = False
        if self.schedule:
            self.schedule.cancel()
            await asyncio.gather(self.schedule, return_exceptions=True)
            self.schedule = None
        self.pcm.clear()
        self.next_listen = self.last_received = 0.
        if self.conn:
            try:
                self.conn.audio.switchAudioChannel(False)
                callbacks = getattr(self.conn.audio, 'track_callbacks', [])
                if self.callback in callbacks:callbacks.remove(self.callback)
            except Exception:pass
        self.resampler = self.callback = None

    async def detach(self):
        await self._stop_capture()
        self.conn = None
        self.update(receiving=False)

    def suspend_for_playback(self, value):
        self.suspended = value
        if value:
            self.accepting = False
            self.pcm.clear()
            if self.conn:
                try:self.conn.audio.switchAudioChannel(False)
                except Exception:pass

    def resume(self):
        async def apply():
            self.suspend_for_playback(False)
        asyncio.run_coroutine_threadsafe(apply(), self.robot.loop).result(5)

    def drain_saved_audio(self, timeout=5):
        # Samples awaiting STT have no disk file; wait briefly during clean shutdown.
        deadline = time.monotonic()+timeout
        while self.jobs.unfinished_tasks and time.monotonic()<deadline:
            time.sleep(.05)

    def _process(self, pcm, image, session, meta):
        import numpy as np
        from faster_whisper import WhisperModel
        if self.model_name != meta['model']:
            self.update(message=f"STT {meta['model']} 모델 준비 · 첫 실행 다운로드")
            self.model = None
            self.model_name = None
            self.model = WhisperModel(meta['model'], device='cpu', compute_type='int8', cpu_threads=2)
            self.model_name = meta['model']
        samples = np.frombuffer(pcm, dtype='<i2').astype(np.float32)/32768.0
        segments, _ = self.model.transcribe(samples, language='ko', task='transcribe',
                    beam_size=5, vad_filter=True, condition_on_previous_text=False)
        rows = [dict(start_seconds=round(s.start,3),end_seconds=round(s.end,3),text=s.text.strip())
                for s in segments if s.text.strip()]
        transcript = '\n'.join(s['text'] for s in rows)
        self.update(checked=self.info['checked']+1)
        if not keyword_matches(transcript):
            # Do not store the recording, screenshot, or transcript for other speech.
            return False
        meta.update(transcript=transcript, segments=rows, review_required=True,
                    note='도움 요청 키워드 자동 인식. 발화자·실제 위급 여부는 미확인.')
        rid = self.store.create(session, '도움 요청 음성 후보', 'auto', ['speech'], image, meta)
        path = self.store.media/(rid+'.wav')
        try:
            with wave.open(str(path), 'wb') as out:
                out.setnchannels(1);out.setsampwidth(2);out.setframerate(16000);out.writeframes(pcm)
            txt = self.store.media/(rid+'.txt')
            txt.write_text(transcript, encoding='utf-8-sig')
            meta['text_file'] = txt.name
            (self.store.media/(rid+'.json')).write_text(json.dumps(meta,ensure_ascii=False,indent=2), encoding='utf-8')
            self.store.finish(rid, path.name, meta, 'ready')
        except Exception as exc:
            meta['error']=str(exc)
            self.store.finish(rid, path.name if path.exists() else '', meta, 'error')
            raise
        self.update(last_text=transcript, saved=self.info['saved']+1)
        return True

    def _worker(self):
        while True:
            pcm, image, session, meta = self.jobs.get()
            self.update(busy=True)
            try:
                self._process(pcm, image, session, meta)
            except Exception as exc:
                self.update(error=f'STT/저장 실패: {type(exc).__name__}: {exc} · 미확인 듣기 구간은 파일로 남기지 않음')
            finally:
                self.update(busy=False)
                self.jobs.task_done()
