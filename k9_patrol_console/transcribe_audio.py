#!/usr/bin/env python3
""" python transcribe_audio.py recordings/test.wav --model small
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time


def select_audio(audio, latest, recordings):
    if latest:
        files = [p for p in recordings.glob('*.wav') if p.is_file()]
        if not files:
            raise FileNotFoundError(f'WAV 파일이 없습니다: {recordings.resolve()}')
        return max(files, key=lambda p: p.stat().st_mtime).resolve()
    path = audio.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'음성 파일을 찾을 수 없습니다: {path}')
    if path.stat().st_size == 0:
        raise ValueError('빈 음성 파일입니다.')
    return path


def transcribe(audio, model_name, use_vad):
    from faster_whisper import WhisperModel

    print(f'[STT] 모델 준비: {model_name} / CPU INT8', flush=True)
    print('[안내] 첫 실행은 모델 다운로드에 인터넷이 필요합니다.', flush=True)
    model = WhisperModel(model_name, device='cpu', compute_type='int8')
    started = time.perf_counter()
    print(f'[STT] 변환 중: {audio.name}', flush=True)
    segments, info = model.transcribe(
        str(audio), language='ko', task='transcribe', beam_size=5,
        vad_filter=use_vad, condition_on_previous_text=False,
    )
    rows = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            row = {
                'start_seconds': round(segment.start, 3),
                'end_seconds': round(segment.end, 3),
                'text': text,
            }
            rows.append(row)
            print(f"[{segment.start:.1f} ~ {segment.end:.1f}초] {text}", flush=True)
    return {
        'audio_path': str(audio),
        'created_at': datetime.now().astimezone().isoformat(),
        'model': model_name,
        'device': 'cpu',
        'compute_type': 'int8',
        'language': info.language,
        'duration_seconds': info.duration,
        'processing_seconds': round(time.perf_counter() - started, 3),
        'vad_enabled': use_vad,
        'text': '\n'.join(row['text'] for row in rows),
        'segments': rows,
        'note': '자동 음성 인식 결과입니다. 원본 음성으로 확인하세요. 시간은 파일 시작 기준입니다.',
    }


def save_result(result, audio, output_dir=None):
    directory = output_dir or audio.parent
    directory.mkdir(parents=True, exist_ok=True)
    stem = audio.stem + '_stt_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    txt_path, json_path = directory / (stem + '.txt'), directory / (stem + '.json')
    # UTF-8 BOM makes Korean text easy to open with Windows editors.
    with txt_path.open('x', encoding='utf-8-sig') as stream:
        stream.write(result['text'] + ('\n' if result['text'] else ''))
    with json_path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    return txt_path, json_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('audio', nargs='?', type=Path, help='변환할 녹음 파일 경로')
    group.add_argument('--latest', action='store_true', help='recordings 폴더에서 가장 최근 수정된 WAV 사용')
    parser.add_argument('--recordings', type=Path, default=Path('recordings'), help='--latest의 검색 폴더')
    parser.add_argument('--model', choices=['tiny', 'base', 'small', 'medium', 'large-v3'], default='base')
    parser.add_argument('--output-dir', type=Path, help='저장 폴더 (기본: 음성 파일과 같은 폴더)')
    parser.add_argument('--no-vad', action='store_true', help='음성 구간 필터 해제 (작은 목소리가 누락될 때 비교)')
    args = parser.parse_args()
    try:
        audio = select_audio(args.audio, args.latest, args.recordings.expanduser())
        result = transcribe(audio, args.model, not args.no_vad)
        txt, metadata = save_result(result, audio, args.output_dir.expanduser() if args.output_dir else None)
        if not result['text']:
            print('[안내] 인식된 발화가 없습니다. 원본 녹음을 확인하거나 --no-vad로 비교하세요.')
        print(f'[저장] {txt}\n[저장] {metadata}')
        return 0
    except ModuleNotFoundError as exc:
        print(f'[오류] 패키지 누락: {exc.name}\n설치: python -m pip install faster-whisper', file=sys.stderr)
    except KeyboardInterrupt:
        print('\n[중지] 변환을 중단했습니다. 원본 녹음은 유지됩니다.', file=sys.stderr)
        return 130
    except Exception as exc:
        print(f'[오류] {type(exc).__name__}: {exc}', file=sys.stderr)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
