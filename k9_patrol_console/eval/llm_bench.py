"""Time the important-record selection (highlights.HighlightsService) end to end.

Runs the same code path as the 선정 button, against a COPY of the data folder,
and logs Ollama's own timing fields for every LLM call.

Usage (from k9_patrol_console/, Ollama running):
    python eval/llm_bench.py --data /path/to/copy_of_data [--model qwen3.5:4b] [--limit 5]
    python eval/llm_bench.py --data ... --before 2026-11-05T10:01:38   # records saved before this UTC time
Never point --data at the live data folder: the result is written to its highlights table.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import highlights  # noqa: E402
import ollama_connection  # noqa: E402
from storage import Store  # noqa: E402

calls = []
_chat = ollama_connection.chat


def timed_chat(payload):
    t = time.monotonic()
    raw = _chat(payload)
    ns = lambda k: round(raw.get(k, 0) / 1e9, 1)
    calls.append(dict(wall=round(time.monotonic() - t, 1), load=ns('load_duration'),
                      prompt_tokens=raw.get('prompt_eval_count'), prompt_s=ns('prompt_eval_duration'),
                      output_tokens=raw.get('eval_count'), output_s=ns('eval_duration'),
                      input_chars=len(payload['messages'][-1]['content'])))
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--model', default='qwen3.5:4b')
    ap.add_argument('--limit', type=int, default=5)
    ap.add_argument('--before', default='')
    args = ap.parse_args()
    if Path(args.data).resolve() == (ROOT / 'data').resolve():
        sys.exit('데이터 사본을 지정하세요 (선정 결과가 DB에 저장됩니다).')

    highlights.chat = timed_chat
    ollama_connection.start_background()
    store = Store(args.data)
    records = [r for r in store.list() if not args.before or r['time'] < args.before]
    svc = highlights.HighlightsService(store)
    t = time.monotonic()
    svc.generate(records, args.model, args.limit)
    total = round(time.monotonic() - t, 1)
    info = svc.status()
    for i, c in enumerate(calls, 1):
        print(f'call {i}: {c}')
    print(f'records {len(records)}, LLM calls {len(calls)}, total {total}s')
    if info['error']:
        print('ERROR', info['error'])
    else:
        for s in info['result']['content']['selections']:
            print('-', s['record_id'][:8], s['reason'])
    print(json.dumps(dict(records=len(records), calls=calls, total_s=total, error=info['error'],
                          selected=[s['record_id'][:8] for s in (info['result'] or {}).get('content', {}).get('selections', [])]),
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
