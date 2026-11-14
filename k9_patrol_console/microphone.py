"""Browser PCM to the existing Go2 WebRTC audio sender."""
import asyncio
from fractions import Fraction
import secrets

from aiortc import AudioStreamTrack
from av import AudioFrame

RATE = 48000
SAMPLES = 960
FRAME_BYTES = SAMPLES * 2


class MicrophoneTrack(AudioStreamTrack):
    def __init__(self):
        super().__init__()
        self.pending = asyncio.Queue(maxsize=4)
        self.samples = 0
        self.next_frame = None

    def feed(self, pcm):
        for offset in range(0, len(pcm), FRAME_BYTES):
            if self.pending.full():
                self.pending.get_nowait()
            self.pending.put_nowait(pcm[offset:offset + FRAME_BYTES])

    async def recv(self):
        loop = asyncio.get_running_loop()
        now = loop.time()
        self.next_frame = max(now, self.next_frame or now)
        await asyncio.sleep(max(0, self.next_frame - now))
        self.next_frame += SAMPLES / RATE
        frame = AudioFrame(format="s16", layout="mono", samples=SAMPLES)
        frame.planes[0].update(
            self.pending.get_nowait() if not self.pending.empty() else bytes(FRAME_BYTES))
        frame.sample_rate = RATE
        frame.pts = self.samples
        frame.time_base = Fraction(1, RATE)
        self.samples += SAMPLES
        return frame


class Microphone:
    def __init__(self, robot):
        self.robot = robot
        self.track = self.sender = self.session = self.watchdog = None

    def call(self, action, payload=None, session=None):
        async def run():
            if action == "start":
                if not self.robot.status()["connected"] or self.robot.conn is None:
                    raise ValueError("Go2를 먼저 연결하세요.")
                if self.session:
                    raise ValueError("이미 마이크 방송 중입니다.")
                if self.robot.speech.status()["recording"] or self.robot.audio.status()["busy"]:
                    raise ValueError("음성 녹음 또는 파일 작업이 끝난 뒤 방송하세요.")
                pc = self.robot.conn.pc
                tx = next((t for t in pc.getTransceivers() if t.kind == "audio"
                           and t.currentDirection in ("sendrecv", "sendonly")), None)
                if tx is None or tx.sender.track is not None:
                    raise ValueError("사용 가능한 Go2 오디오 송신 채널이 없습니다.")
                self.track = MicrophoneTrack()
                self.sender = tx.sender
                self.session = secrets.token_urlsafe(18)
                try:
                    self.sender.replaceTrack(self.track)
                    self.arm_watchdog(5)
                except Exception:
                    self.stop()
                    raise
                return {"session": self.session}
            if not self.session or self.session != session:
                if action == "stop":
                    return {"ok": True}
                raise ValueError("마이크 방송 세션이 종료되었습니다.")
            if action == "stop":
                self.stop()
            elif action == "pcm":
                if not self.robot.status()["connected"]:
                    self.stop()
                    raise ValueError("Go2 연결이 끊겼습니다.")
                self.track.feed(payload)
                self.arm_watchdog(1.5)
            return {"ok": True}
        future = asyncio.run_coroutine_threadsafe(run(), self.robot.loop)
        try:
            return future.result(3)
        except Exception:
            future.cancel()
            raise

    def arm_watchdog(self, seconds):
        if self.watchdog:
            self.watchdog.cancel()
        self.watchdog = asyncio.get_running_loop().call_later(seconds, self.stop)

    def stop(self):
        if self.watchdog:
            self.watchdog.cancel()
        if self.sender and self.sender.track is self.track:
            self.sender.replaceTrack(None)
        if self.track:
            self.track.stop()
        self.track = self.sender = self.session = self.watchdog = None
