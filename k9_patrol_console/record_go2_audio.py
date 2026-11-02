#!/usr/bin/env python3
"""Go2 robot microphone -> WAV (16 kHz, mono, signed 16-bit PCM).

Run in your existing WebRTC venv:
    python record_go2_audio.py --ip 192.168.0.101 --seconds 30
Use --seconds 0 to record until Ctrl+C. Files go to ./recordings/.
Requires unitree_webrtc_connect and av (installed with aiortc).
No robot movement commands are sent.
"""
import argparse
import asyncio
import math
import os
from pathlib import Path
import sys
import time
import wave
from datetime import datetime


async def record(args):
    from av.audio.resampler import AudioResampler
    from unitree_webrtc_connect.webrtc_driver import (
        UnitreeWebRTCConnection, WebRTCConnectionMethod,
    )

    output = args.output or Path('recordings') / (
        datetime.now().strftime('go2_%Y%m%d_%H%M%S_%f') + '.wav'
    )
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    options = {'ip': args.ip}
    if os.getenv('GO2_AES_KEY'):
        options['aes_128_key'] = os.environ['GO2_AES_KEY']
    conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, **options)
    resampler = AudioResampler(format='s16', layout='mono', rate=16000)
    samples = 0
    limit = round(args.seconds * 16000) if args.seconds else None
    first = asyncio.Event()
    done = asyncio.Event()
    failure = None
    accepting = False
    last_frame = None
    writer = None

    def write_frames(frames):
        nonlocal samples
        for frame in frames:
            available = frame.samples
            count = min(available, max(0, limit - samples)) if limit else available
            if count:
                # Plane buffers may contain alignment padding: write samples only.
                writer.writeframesraw(bytes(frame.planes[0])[:count * 2])
                samples += count
            if limit and samples >= limit:
                done.set()
                break

    async def on_audio(frame):
        nonlocal failure, last_frame
        if not accepting or done.is_set():
            return
        try:
            last_frame = time.monotonic()
            first.set()
            write_frames(resampler.resample(frame))
        except Exception as exc:
            failure = exc
            done.set()

    try:
        print(f'[GO2] Connecting to {args.ip}...', flush=True)
        await asyncio.wait_for(conn.connect(), timeout=40)
        # Exclusive creation prevents accidental overwrite of an earlier recording.
        with output.open('xb') as raw:
            with wave.open(raw, 'wb') as wav:
                writer = wav
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                conn.audio.add_track_callback(on_audio)
                accepting = True
                try:
                    conn.audio.switchAudioChannel(True)
                    print('[REC] Waiting for microphone audio (15 seconds)...', flush=True)
                    await asyncio.wait_for(first.wait(), timeout=15)
                    print('[REC] Recording. Ctrl+C to stop and save.', flush=True)
                    while not done.is_set():
                        try:
                            await asyncio.wait_for(done.wait(), timeout=0.5)
                        except asyncio.TimeoutError:
                            pass
                        if last_frame and time.monotonic() - last_frame > 10:
                            raise RuntimeError('Audio stopped for 10 seconds; partial recording saved.')
                    if failure:
                        raise RuntimeError(f'Audio processing failed: {failure}')
                finally:
                    accepting = False
                    if samples:
                        write_frames(resampler.resample(None))
    finally:
        accepting = False
        try:
            if getattr(conn, 'audio', None):
                conn.audio.switchAudioChannel(False)
        except Exception:
            pass
        try:
            await asyncio.wait_for(conn.disconnect(), timeout=5)
        except Exception:
            pass
        if samples:
            print(f'[SAVED] {output}\n[INFO] {samples / 16000:.2f}s | 16000 Hz | mono | PCM16', flush=True)
        elif writer is not None:
            output.unlink(missing_ok=True)
            print('[INFO] No audio received. Empty file removed.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ip', default='192.168.0.101', help='Go2 Wi-Fi IP')
    parser.add_argument('--seconds', type=float, default=30, help='Audio duration; 0 = until Ctrl+C (default: 30)')
    parser.add_argument('--output', type=Path, help='WAV output path (must not already exist)')
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds < 0:
        parser.error('--seconds must be a finite number >= 0')
    if 0 < args.seconds < 1 / 16000:
        parser.error('--seconds is too small')
    try:
        asyncio.run(record(args))
    except KeyboardInterrupt:
        print('[STOP] Recording stopped.')
    except ModuleNotFoundError as exc:
        print(f'[ERROR] Missing dependency: {exc.name}\nRun in the venv used for your Go2 WebRTC connection.', file=sys.stderr)
        return 1
    except asyncio.TimeoutError:
        print('[ERROR] Connection or audio reception timed out. Check Wi-Fi and robot microphone stream.', file=sys.stderr)
        return 1
    except Exception as exc:
        print(f'[ERROR] {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
