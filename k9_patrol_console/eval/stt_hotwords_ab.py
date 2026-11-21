"""STT 구조 요청 단어 힌트(hotwords) 적용 전후 비교.

같은 녹음을 콘솔과 같은 설정(faster-whisper, cpu int8, beam 5, VAD 켬)으로
hotwords 없이 / 있게 두 번 전사하고, 도움 요청 판정(help_request.py) 결과를 비교한다.

정답은 labels CSV(file, truth, said)로 준다. truth: help(구조 요청을 말함) / daily(일상 발화) / unknown.
  - 구조 요청 인식: truth=help 녹음 중 판정이 help 인 수
  - 오탐: truth=daily 녹음 중 판정이 help 또는 suspect 인 수 (일상 발화를 도움 요청으로 잘못 잡음)
사용: python eval/stt_hotwords_ab.py WAV_DIR LABELS.csv [--model small] [--out result.md]
"""
import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from help_request import find_help_request  # noqa: E402

HOTWORDS = '살려주세요 도와주세요 구해주세요 살려줘 도와줘'


def load(path):
    import wave
    import numpy as np
    with wave.open(str(path), 'rb') as w:
        if w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getframerate() == 16000:
            return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    return str(path)


def run(model, audio, hot):
    t = time.time()
    segs, _ = model.transcribe(audio, language='ko', task='transcribe', beam_size=5, vad_filter=True,
                               condition_on_previous_text=False, hotwords=hot)
    text = '\n'.join(s.text.strip() for s in segs if s.text.strip())
    return text, time.time() - t


def level(text):
    r = find_help_request(text)
    return r.get('code', 'none')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('wav_dir'); ap.add_argument('labels')
    ap.add_argument('--model', default='small'); ap.add_argument('--hotwords', default=HOTWORDS)
    ap.add_argument('--out', default='stt_hotwords_result.md')
    a = ap.parse_args()
    from faster_whisper import WhisperModel
    model = WhisperModel(a.model, device='cpu', compute_type='int8', cpu_threads=2)
    rows = list(csv.DictReader(open(a.labels, encoding='utf-8-sig')))
    out = []
    for r in rows:
        p = Path(a.wav_dir) / r['file']
        if not p.exists():
            print('없음', p); continue
        audio = load(p)
        t0, s0 = run(model, audio, None)
        t1, s1 = run(model, audio, a.hotwords)
        out.append(dict(file=r['file'], truth=r['truth'], said=r.get('said', ''), before=t0, after=t1,
                        lv0=level(t0), lv1=level(t1), sec0=s0, sec1=s1))
        print(r['file'][:8], r['truth'], '|', t0.replace('\n', ' / '), '→', t1.replace('\n', ' / '))

    def count(truth, levels):
        return sum(1 for o in out if o['truth'] == truth and o[levels] in ('help',)), \
               sum(1 for o in out if o['truth'] == truth and o[levels] in ('help', 'suspect'))
    nh = sum(o['truth'] == 'help' for o in out); nd = sum(o['truth'] == 'daily' for o in out)
    lines = ['# STT 단어 힌트 적용 전후 비교', '',
             f'- 모델 {a.model}, hotwords: `{a.hotwords}`', f'- 녹음 {len(out)}개 (구조 요청 {nh}, 일상 발화 {nd}, 미분류 {len(out)-nh-nd})', '',
             '| 항목 | 적용 전 | 적용 후 |', '|---|---|---|',
             f'| 구조 요청 녹음 중 도움 요청 판정 | {count("help","lv0")[0]} / {nh} | {count("help","lv1")[0]} / {nh} |',
             f'| 일상 발화를 도움 요청·의심으로 잘못 잡음 | {count("daily","lv0")[1]} / {nd} | {count("daily","lv1")[1]} / {nd} |',
             f'| 평균 전사 시간 (초) | {sum(o["sec0"] for o in out)/max(len(out),1):.2f} | {sum(o["sec1"] for o in out)/max(len(out),1):.2f} |',
             '', '| 녹음 | 정답 | 실제 발화 | 적용 전 | 판정 | 적용 후 | 판정 |', '|---|---|---|---|---|---|---|']
    for o in out:
        lines.append(f"| {o['file'][:8]} | {o['truth']} | {o['said']} | {o['before'].replace(chr(10),' / ')} | {o['lv0']} | {o['after'].replace(chr(10),' / ')} | {o['lv1']} |")
    Path(a.out).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines[:10]))


if __name__ == '__main__':
    main()
