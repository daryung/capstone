"""Compare Go2 WAV transcription with and without VAD."""
import argparse
from pathlib import Path
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", nargs="+", type=Path)
    parser.add_argument("--model", default="small", choices=("base", "small", "medium"))
    args = parser.parse_args()
    for path in args.wav:
        if not path.is_file() or path.suffix.lower() != ".wav":
            parser.error(f"WAV 파일을 확인하세요: {path}")
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        parser.exit(1, "현재 가상환경에 faster-whisper를 설치하세요.\n")

    model = WhisperModel(args.model, device="cpu", compute_type="int8", cpu_threads=2)
    for path in args.wav:
        print(f"\n=== {path.name} / {args.model} ===", flush=True)
        for vad in (True, False):
            started = perf_counter()
            segments, info = model.transcribe(
                str(path), language="ko", task="transcribe",
                beam_size=5, vad_filter=vad, condition_on_previous_text=False,
            )
            rows = [(s.start, s.end, s.text.strip()) for s in segments if s.text.strip()]
            elapsed = perf_counter() - started
            print(f"\nVAD {'ON' if vad else 'OFF'} | 처리 {elapsed:.2f}초")
            retained = getattr(info, "duration_after_vad", None)
            if vad and retained is not None:
                print(f"VAD 통과 오디오: {retained:.2f}초")
            for start, end, text in rows:
                print(f"[{start:.2f}–{end:.2f}초] {text}")
            if not rows:
                print("(인식된 발화 없음)")


if __name__ == "__main__":
    main()
