import argparse
import ipaddress
import json
import math
from pathlib import Path
import re
import secrets
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse,parse_qs,unquote
from controller import Controller
from storage import Store
from vision import Vision
from summarizer import Summarizer
from ollama_connection import start_background
from audio_control import AudioControl
from speech import SpeechRecorder
from audio_files import save_audio

ROOT=Path(__file__).parent


def serve(args):
    start_background()
    store=Store(args.data);vision=Vision(store);robot=Controller(vision);llm=Summarizer(store)
    audio=AudioControl(robot);robot.audio=audio
    speech=SpeechRecorder(store,robot,vision);robot.speech=speech
    audio_root=store.root/'audio'
    token=secrets.token_urlsafe(32);origin=f'http://127.0.0.1:{args.port}'
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def authorized(self):
            query=parse_qs(urlparse(self.path).query)
            return self.headers.get('Host')==urlparse(origin).netloc and (self.headers.get('X-Control-Token')==token or query.get('token',[''])[0]==token)
        def reply(self,status,data,kind='application/json; charset=utf-8',download=''):
            raw=data if isinstance(data,bytes) else json.dumps(data,ensure_ascii=False).encode()
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            if download:self.send_header('Content-Disposition',f'attachment; filename="{download}"')
            self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            path=urlparse(self.path).path;q=parse_qs(urlparse(self.path).query)
            if self.headers.get('Host')!=urlparse(origin).netloc:self.reply(403,{'error':'127.0.0.1 주소로 접속하세요.'});return
            if path=='/':
                self.reply(200,(ROOT/'static/index.html').read_text().replace('__TOKEN__',token).encode(),'text/html; charset=utf-8');return
            if not self.authorized():self.reply(403,{'error':'인증되지 않은 로컬 요청'});return
            try:
                if path=='/api/status':self.reply(200,dict(robot=robot.status(),vision=vision.status(),llm=llm.status(),audio=audio.status(),speech=speech.status()))
                elif path in ('/api/records','/api/export'):
                    records=store.list(query=q.get('q',[''])[0],target=q.get('target',[''])[0],session=q.get('session',[''])[0])
                    kind=q.get('kind',['all'])[0]
                    if kind in ('auto','manual'):records=[r for r in records if r['type']==kind]
                    elif kind=='excluded':records=[r for r in records if r['excluded']]
                    self.reply(200,records,download='k9-records.json' if path.endswith('export') else '')
                elif path=='/api/summaries':self.reply(200,store.summaries())
                elif path=='/api/video':
                    self.send_response(200);self.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame');self.send_header('Cache-Control','no-store');self.end_headers()
                    try:
                        previous=None
                        while True:
                            with vision.lock:image=vision.jpeg
                            if image is not None and image is not previous and vision.status()['video']:
                                self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(image)).encode()+b'\r\n\r\n'+image+b'\r\n');self.wfile.flush();previous=image
                            time.sleep(.08)
                    except (BrokenPipeError,ConnectionResetError):pass
                elif path.startswith('/api/media/'):
                    name=path.removeprefix('/api/media/')
                    if not re.fullmatch(r'[a-f0-9]{32}\.(jpg|avi|mp4|wav|txt|json)',name):raise ValueError('잘못된 파일 이름')
                    file=store.media/name
                    if not file.is_file():self.reply(404,{'error':'파일을 찾을 수 없습니다.'});return
                    kind='image/jpeg' if name.endswith('.jpg') else 'video/mp4' if name.endswith('.mp4') else 'audio/wav' if name.endswith('.wav') else 'text/plain; charset=utf-8' if name.endswith('.txt') else 'application/json; charset=utf-8' if name.endswith('.json') else 'video/x-msvideo'
                    # Range support lets browsers seek recorded MP4 clips.
                    total=file.stat().st_size;start=0;end=total-1;partial=False
                    if self.headers.get('Range'):
                        match=re.fullmatch(r'bytes=(\d+)-(\d*)',self.headers['Range'])
                        if not match:raise ValueError('지원하지 않는 Range')
                        start=int(match[1]);end=min(int(match[2]) if match[2] else end,end);partial=True
                        if start>end:self.reply(416,{'error':'범위 오류'});return
                    self.send_response(206 if partial else 200);self.send_header('Content-Type',kind);self.send_header('Accept-Ranges','bytes');self.send_header('Content-Length',str(end-start+1));self.send_header('Cache-Control','no-store')
                    if partial:self.send_header('Content-Range',f'bytes {start}-{end}/{total}')
                    if q.get('download'):self.send_header('Content-Disposition',f'attachment; filename="{name}"')
                    self.end_headers()
                    with file.open('rb') as f:
                        f.seek(start);remaining=end-start+1
                        while remaining:
                            chunk=f.read(min(65536,remaining));self.wfile.write(chunk);remaining-=len(chunk)
                else:self.reply(404,{'error':'Not found'})
            except (BrokenPipeError,ConnectionResetError):pass
            except Exception as exc:self.reply(400,{'error':str(exc)})
        def do_POST(self):
            if not self.authorized() or self.headers.get('Origin')!=origin:self.reply(403,{'error':'동일 출처의 로컬 콘솔 요청만 허용됩니다.'});return
            try:
                size=int(self.headers.get('Content-Length','0'))
                if self.path=='/api/audio/upload':
                    if not 0<size<=8*1024*1024:raise ValueError('오디오 파일은 8MB 이하로 선택하세요.')
                    if not robot.status()['connected']:raise ValueError('로봇을 먼저 연결하세요.')
                    if audio.status()['busy']:raise ValueError('오디오 요청 처리 중입니다.')
                    path,duration=save_audio(self.rfile,size,audio_root,unquote(self.headers.get('X-Audio-Name','announcement')))
                    try:audio.start('upload',path=path)
                    except BaseException:path.unlink(missing_ok=True);raise
                    self.reply(200,{'ok':True,'seconds':round(duration,2)});return
                if not 0<size<=20000:raise ValueError('요청 크기 오류')
                data=json.loads(self.rfile.read(size));out={'ok':True}
                if self.path=='/api/speech/config':speech.configure(data)
                elif self.path=='/api/speech/resume':speech.resume()
                elif self.path=='/api/audio/list':audio.start('list')
                elif self.path=='/api/audio/play':audio.start('play',uid=data.get('id'))
                elif self.path=='/api/audio/pause':audio.start('pause')
                elif self.path=='/api/connect':robot.begin_connect(str(ipaddress.IPv4Address(data['ip'])))
                elif self.path=='/api/disconnect':robot.disconnect()
                elif self.path=='/api/command':
                    v=[float(data[k]) for k in ('x','y','z')]
                    if not all(math.isfinite(x) for x in v) or math.hypot(*v[:2])>.40001 or abs(v[2])>.5:raise ValueError('속도 제한 초과')
                    if not robot.command(v,int(data['seq'])):self.reply(409,{'error':'연결·모드·명령 순서를 확인하세요.'});return
                elif self.path=='/api/pose':robot.pose(data['name'])
                elif self.path=='/api/config':vision.configure(data)
                elif self.path=='/api/capture':
                    rid=robot.capture()
                    if data.get('title') or data.get('note') or data.get('keywords'):
                        store.edit(rid,str(data.get('note','')),str(data.get('keywords','')),False)
                        if data.get('title'):
                            with store.connection() as db:db.execute('UPDATE records SET title=? WHERE id=?',(str(data['title'])[:100],rid))
                    out['id']=rid
                elif self.path=='/api/record':
                    if not isinstance(data.get('excluded',False),bool):raise ValueError('제외 상태 오류')
                    store.edit(data['id'],str(data.get('note','')),str(data.get('keywords','')),data.get('excluded',False))
                elif self.path=='/api/summarize':
                    records=store.list(str(data.get('q','')),str(data.get('target','')),str(data.get('session','')),False)
                    records=[r for r in records if r['state'] in ('ready','partial')]
                    kind=data.get('kind','all')
                    if kind in ('auto','manual'):records=[r for r in records if r['type']==kind]
                    elif kind=='excluded':records=[]
                    llm.start(records,data.get('model','qwen3.5:4b'))
                else:self.reply(404,{'error':'Not found'});return
                self.reply(200,out)
            except Exception as exc:self.reply(400,{'error':str(exc)})
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'K9 Console: {origin}  | 실제 연결 전용 | 자료: {store.root}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        robot.stop()
        try:robot.disconnect()
        except Exception:pass
        speech.drain_saved_audio()
        server.server_close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8000);p.add_argument('--data',default=str(ROOT/'data'))
    serve(p.parse_args())
