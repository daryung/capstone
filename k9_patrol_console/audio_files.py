"""Validate and normalize user-selected announcement files for AudioHub."""
from pathlib import Path
import shutil
import re
import subprocess
import uuid
import wave


def save_audio(stream,size,root,name="announcement"):
    label=re.sub(r"[^\w가-힣-]", "_", Path(name).stem)[:36] or "announcement"
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    stem=label+'_'+uuid.uuid4().hex[:12]
    source=root/(stem+'.input');target=root/(stem+'.wav')
    try:
        with source.open('wb') as out:
            remaining=size
            while remaining:
                block=stream.read(min(65536,remaining))
                if not block:raise ValueError('파일 전송이 중단되었습니다.')
                out.write(block);remaining-=len(block)
        # PCM WAV is supported without ffmpeg. Other formats require conversion.
        direct=False
        try:
            with wave.open(str(source),'rb') as w:
                duration=w.getnframes()/w.getframerate()
                if duration>30:raise ValueError('안내 음성은 30초 이하로 선택하세요.')
                direct=w.getframerate()==44100 and w.getsampwidth()==2 and w.getnchannels() in (1,2) and w.getcomptype()=='NONE'
        except (wave.Error,EOFError):pass
        if direct:source.replace(target)
        else:
            if not shutil.which('ffmpeg'):raise ValueError('MP3 및 WAV 변환에는 ffmpeg가 필요합니다. 또는 44.1kHz·16bit PCM WAV를 선택하세요.')
            proc=subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',str(source),'-t','31','-vn','-ar','44100','-ac','1','-c:a','pcm_s16le',str(target)],capture_output=True,timeout=30)
            if proc.returncode:raise ValueError('오디오 파일을 변환할 수 없습니다.')
        with wave.open(str(target),'rb') as w:
            duration=w.getnframes()/w.getframerate()
            if not 0<duration<=30:raise ValueError('안내 음성은 0초 초과 30초 이하로 선택하세요.')
        return target,duration
    except BaseException:
        target.unlink(missing_ok=True);raise
    finally:source.unlink(missing_ok=True)
