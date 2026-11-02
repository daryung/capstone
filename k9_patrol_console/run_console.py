"""Windows launcher: start local Ollama when needed, then run app.py.

Place beside app.py. Run with your Windows virtual environment:
    .\.venv-win\Scripts\python.exe run_console.py
Ollama must already be installed. Models are not downloaded automatically.
"""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

URL = 'http://127.0.0.1:11434/api/tags'
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def ollama_ready():
    try:
        with OPENER.open(URL, timeout=1) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def find_ollama():
    executable = shutil.which('ollama')
    if executable:
        return executable
    local = os.environ.get('LOCALAPPDATA')
    if local:
        candidate = Path(local) / 'Programs' / 'Ollama' / 'ollama.exe'
        if candidate.is_file():
            return str(candidate)
    return None


def ensure_ollama(root):
    if ollama_ready():
        print('[Ollama] 실행 중입니다.', flush=True)
        return
    executable = find_ollama()
    if not executable:
        raise RuntimeError('Ollama를 찾지 못했습니다. Windows에 Ollama를 먼저 설치하세요.')
    log_dir = root / 'logs'
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / 'ollama.log'
    env = os.environ.copy()
    env['OLLAMA_HOST'] = '127.0.0.1:11434'
    flags = 0
    if os.name == 'nt':
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    print('[Ollama] 자동으로 시작합니다...', flush=True)
    with log_path.open('ab') as log:
        process = subprocess.Popen(
            [executable, 'serve'], env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, creationflags=flags,
        )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if ollama_ready():
            print('[Ollama] 준비 완료.', flush=True)
            return
        if process.poll() is not None:
            # Another instance may have won the port while we were starting.
            if ollama_ready():
                return
            raise RuntimeError(f'Ollama 실행 실패. 로그를 확인하세요: {log_path}')
        time.sleep(0.5)
    raise RuntimeError(f'Ollama 응답 대기 시간 초과. 로그: {log_path}')


def main():
    root = Path(__file__).resolve().parent
    app = root / 'app.py'
    if not app.is_file():
        print('[오류] run_console.py를 app.py와 같은 폴더에 넣으세요.', file=sys.stderr)
        return 1
    try:
        ensure_ollama(root)
        print('[Console] 시작합니다. LLM 주소: http://localhost:11434', flush=True)
        return subprocess.call([sys.executable, str(app), *sys.argv[1:]], cwd=root)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f'[오류] {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
