"""Local Ollama startup and Windows bridge for WSL (stdlib only)."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import urllib.request

_URL = os.environ.get('K9_OLLAMA_URL', 'http://127.0.0.1:11434').rstrip('/')
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_BRIDGE = False
_LOCK = threading.Lock()
_READY = threading.Event()
_ERROR = ''
_PS = None


def _powershell(script, payload=None, timeout=35):
    if not _PS:
        raise RuntimeError('Windows PowerShell을 사용할 수 없습니다. Windows에서 Ollama를 실행하세요.')
    prefix = '[Console]::InputEncoding = [Text.UTF8Encoding]::new($false); [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false); $ErrorActionPreference = "Stop"; '
    result = subprocess.run([_PS, '-NoProfile', '-NonInteractive', '-Command', prefix + script],
                            input=json.dumps(payload, ensure_ascii=False) if payload is not None else '',
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            encoding='utf-8', errors='replace', timeout=timeout)
    if result.returncode:
        raise RuntimeError('Windows Ollama 요청 실패: ' + result.stderr.strip()[-600:])
    return json.loads(result.stdout.lstrip('\ufeff').strip())


def _http(path, payload=None, timeout=2):
    request = urllib.request.Request(_URL + path,
        data=json.dumps(payload, ensure_ascii=False).encode('utf-8') if payload is not None else None,
        headers={'Content-Type': 'application/json'})
    with _OPENER.open(request, timeout=timeout) as response:
        return json.load(response)


def _windows_start_script():
    return r'''
$uri = 'http://127.0.0.1:11434/api/tags';
try { $r = Invoke-RestMethod -Uri $uri -TimeoutSec 2; $r | ConvertTo-Json -Depth 20 -Compress; exit 0 } catch {};
$exe = (Get-Command ollama.exe -ErrorAction SilentlyContinue).Source;
if (-not $exe) { $exe = Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe' };
if (-not (Test-Path $exe)) { throw 'Windows Ollama executable not found' };
$env:OLLAMA_HOST = '127.0.0.1:11434';
Start-Process -FilePath $exe -ArgumentList 'serve' -WindowStyle Hidden;
for ($i=0; $i -lt 25; $i++) {
  try { $r = Invoke-RestMethod -Uri $uri -TimeoutSec 1; $r | ConvertTo-Json -Depth 20 -Compress; exit 0 } catch {};
  Start-Sleep -Milliseconds 500;
};
throw 'Ollama startup timed out';
'''


def ensure_connection():
    global _BRIDGE, _ERROR, _PS
    with _LOCK:
        try:
            try:
                _http('/api/tags')
                _BRIDGE = False
                _ERROR = ''
                return
            except Exception:
                pass
            # Explicit URLs remain authoritative, including remote servers.
            if os.environ.get('K9_OLLAMA_URL') and _URL not in (
                'http://127.0.0.1:11434', 'http://localhost:11434'):
                raise RuntimeError('K9_OLLAMA_URL 서버에 연결할 수 없습니다: ' + _URL)
            _PS = shutil.which('powershell.exe')
            if not _PS and os.name != 'nt':
                candidate = Path('/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe')
                if candidate.is_file():
                    _PS = str(candidate)
            if os.name == 'nt' and not _PS:
                _PS = str(Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32/WindowsPowerShell/v1.0/powershell.exe')
            if not _PS:
                raise RuntimeError('로컬 Ollama에 연결되지 않습니다. Ollama를 실행하세요.')
            _powershell(_windows_start_script())
            _BRIDGE = os.name != 'nt'
            _ERROR = ''
            print('[LLM] Windows Ollama 연결 완료' + (' (WSL 브리지)' if _BRIDGE else ''), flush=True)
        except Exception as exc:
            _ERROR = str(exc)
            print('[LLM] ' + _ERROR, file=sys.stderr, flush=True)
            raise
        finally:
            _READY.set()


def start_background():
    def worker():
        try:
            ensure_connection()
        except Exception:
            pass
    print('[LLM] Ollama 확인 중 (필요하면 자동 시작)...', flush=True)
    threading.Thread(target=worker, daemon=True, name='ollama-startup').start()


def chat(payload):
    if not _READY.wait(40):
        raise RuntimeError('Ollama 시작 대기 시간 초과. 잠시 후 다시 시도하세요.')
    if _ERROR:
        ensure_connection()
    if _BRIDGE:
        return _powershell(r'''
$body = [Console]::In.ReadToEnd();
$r = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/chat' -Method Post -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 240;
$r | ConvertTo-Json -Depth 100 -Compress;
''', payload, timeout=250)
    return _http('/api/chat', payload, timeout=240)
