"""Go2 AudioHub adapter. RPC fields follow upstream webrtc_audiohub.py.
No motion commands or microphone capture are issued here.
"""
import asyncio
import base64
import hashlib
import json
from pathlib import Path
import threading
import time


class AudioControl:
    def __init__(self,robot):
        self.robot=robot
        self.lock=threading.RLock()
        self.future=None
        self.generation=0
        self.info=dict(busy=False,message='오디오 목록을 불러오세요.',error='',items=[])

    def status(self):
        with self.lock:return dict(self.info,items=list(self.info['items']))

    def cancel(self):
        with self.lock:
            self.generation+=1
            if self.future:self.future.cancel()
            self.info.update(busy=False,items=[],message='로봇 미연결',error='')

    def start(self,action,path=None,uid=None):
        with self.lock:
            if not self.robot.status()['connected']:raise ValueError('로봇을 먼저 연결하세요.')
            if self.info['busy']:raise ValueError('오디오 요청 처리 중입니다.')
            if action not in ('list','upload','play','pause'):raise ValueError('지원하지 않는 오디오 요청')
            if action=='play' and (not isinstance(uid,str) or uid not in {x['id'] for x in self.info['items']}):raise ValueError('목록에서 재생할 파일을 선택하세요.')
            self.info.update(busy=True,error='',message={'list':'목록 조회 중…','upload':'로봇으로 전송 중…','play':'재생 요청 중…','pause':'일시정지 요청 중…'}[action])
            generation=self.generation;conn=self.robot.conn
            self.future=asyncio.run_coroutine_threadsafe(self.execute(action,conn,generation,path,uid),self.robot.loop)

    async def rpc(self,conn,key,parameter):
        from unitree_webrtc_connect.constants import AUDIO_API
        if self.robot.conn is not conn or not self.robot.status()['connected']:raise ConnectionError('로봇 연결이 변경되었습니다.')
        if key not in AUDIO_API:raise RuntimeError('현재 드라이버에 AudioHub API가 없습니다. unitree_webrtc_connect 버전을 확인하세요.')
        response=await asyncio.wait_for(conn.datachannel.pub_sub.publish_request_new('rt/api/audiohub/request',{'api_id':AUDIO_API[key],'parameter':json.dumps(parameter)}),15)
        if not isinstance(response,dict):raise RuntimeError('AudioHub 응답이 없습니다.')
        envelope=response.get('data',{})
        code=envelope.get('header',{}).get('status',{}).get('code')
        if code!=0:raise RuntimeError(f'AudioHub 요청 거부 또는 응답 형식 오류: {code}')
        data=envelope.get('data',{})
        if isinstance(data,str):
            try:data=json.loads(data) if data else {}
            except json.JSONDecodeError:data={}
        return data

    async def listing(self,conn):
        data=await self.rpc(conn,'GET_AUDIO_LIST',{})
        if not isinstance(data,dict) or not isinstance(data.get('audio_list'),list):raise RuntimeError('오디오 목록 응답 형식을 확인하세요.')
        out=[]
        for x in data['audio_list']:
            if isinstance(x,dict) and x.get('UNIQUE_ID'):
                out.append(dict(id=str(x['UNIQUE_ID']),name=str(x.get('CUSTOM_NAME') or x['UNIQUE_ID'])))
        return out

    async def execute(self,action,conn,generation,path,uid):
        try:
            items=None
            if action=='list':
                items=await self.listing(conn);message=f'로봇 저장 파일 {len(items)}개'
            elif action=='upload':
                path=Path(path);raw=await asyncio.to_thread(path.read_bytes)
                encoded=base64.b64encode(raw).decode('ascii');md5=hashlib.md5(raw).hexdigest()
                blocks=[encoded[i:i+4096] for i in range(0,len(encoded),4096)]
                for i,block in enumerate(blocks,1):
                    await self.rpc(conn,'UPLOAD_AUDIO_FILE',dict(file_name=path.stem,file_type='wav',file_size=len(raw),current_block_index=i,total_block_number=len(blocks),block_content=block,current_block_size=len(block),file_md5=md5,create_time=int(time.time()*1000)))
                    with self.lock:
                        if generation!=self.generation:return
                        self.info['message']=f'로봇으로 전송 중 {i}/{len(blocks)}'
                    await asyncio.sleep(.1)
                # Always re-fetch and parse the list after upload; never reuse a stale list.
                for attempt in range(5):
                    items=await self.listing(conn)
                    if any(x['name']==path.stem for x in items):break
                    await asyncio.sleep(.4)
                else:raise RuntimeError('전송 후 로봇 목록에서 파일을 확인하지 못했습니다. 목록을 다시 조회하세요.')
                message='전송 확인 완료 · 목록에서 선택 후 재생하세요.'
            elif action=='play':
                if self.robot.speech:self.robot.speech.suspend_for_playback(True)
                await self.rpc(conn,'SET_PLAY_MODE',{'play_mode':'no_cycle'})
                await self.rpc(conn,'SELECT_START_PLAY',{'unique_id':uid})
                message='재생 요청이 승인되었습니다. 현장에서 소리를 확인하세요.'
            else:
                await self.rpc(conn,'PAUSE',{});message='일시정지 요청이 승인되었습니다.'
                if self.robot.speech:self.robot.speech.suspend_for_playback(False)
            with self.lock:
                if generation!=self.generation:return
                self.info.update(busy=False,message=message,error='')
                if items is not None:self.info['items']=items
        except asyncio.CancelledError:raise
        except Exception as exc:
            with self.lock:
                if generation==self.generation:self.info.update(busy=False,error=f'{type(exc).__name__}: {exc}',message='오디오 요청 실패')
