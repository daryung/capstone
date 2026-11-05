import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.media = self.root / 'media'
        self.media.mkdir(exist_ok=True)
        self.db = self.root / 'records.sqlite3'
        with self.connection() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS records (
            id TEXT PRIMARY KEY, session TEXT, time TEXT, title TEXT, type TEXT,
            targets TEXT, note TEXT DEFAULT '', keywords TEXT DEFAULT '',
            excluded INTEGER DEFAULT 0, state TEXT, image TEXT, clip TEXT,
            metadata TEXT)''')
            db.execute('''CREATE TABLE IF NOT EXISTS summaries (
            id TEXT PRIMARY KEY, time TEXT, model TEXT, source_ids TEXT, content TEXT)''')

            db.execute('''CREATE TABLE IF NOT EXISTS highlights (
            id TEXT PRIMARY KEY, time TEXT, model TEXT, source_ids TEXT, content TEXT)''')

    def connection(self):
        return sqlite3.connect(self.db, timeout=10)

    def create(self, session, title, kind, targets, image, metadata=None):
        rid = uuid.uuid4().hex
        image_name = rid + '.jpg' if image else ''
        if image_name:(self.media/image_name).write_bytes(image)
        try:
            with self.connection() as db:
                db.execute('INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)', (
                    rid, session, utc_now(), title, kind, json.dumps(targets), '', '',
                    0, 'recording', image_name, '', json.dumps(metadata or {})))
        except Exception:
            if image_name:(self.media/image_name).unlink(missing_ok=True)
            raise
        return rid

    def finish(self, rid, clip, metadata, state='ready'):
        with self.connection() as db:
            db.execute('UPDATE records SET clip=?,metadata=?,state=? WHERE id=?',
                       (clip, json.dumps(metadata), state, rid))

    def list(self, query='', target='', session='', include_excluded=True):
        with self.connection() as db:
            db.row_factory = sqlite3.Row
            rows = db.execute('SELECT * FROM records ORDER BY time DESC').fetchall()
        out = []
        for row in rows:
            r = dict(row)
            r['targets'] = json.loads(r['targets'])
            r['metadata'] = json.loads(r['metadata'])
            r['excluded'] = bool(r['excluded'])
            if session and r['session'] != session:continue
            if target and target not in r['targets']:continue
            if not include_excluded and r['excluded']:continue
            hay = ' '.join([r['title'], r['note'], r['keywords'], ' '.join(r['targets']), str(r['metadata'].get('transcript','')), ' '.join(d.get('posture',{}).get('label','') for d in r['metadata'].get('detections',[]))]).casefold()
            if query.casefold() not in hay:continue
            out.append(r)
        return out

    def get(self, rid):
        return next((r for r in self.list() if r['id'] == rid), None)

    def edit(self, rid, note, keywords, excluded):
        with self.connection() as db:
            c = db.execute('UPDATE records SET note=?, keywords=?, excluded=? WHERE id=?',
                           (note[:5000], keywords[:500], int(excluded), rid))
            if c.rowcount != 1:raise ValueError('기록을 찾을 수 없습니다.')

    def save_summary(self, model, ids, content):
        rid = uuid.uuid4().hex
        with self.connection() as db:
            db.execute('INSERT INTO summaries VALUES (?,?,?,?,?)',
                       (rid, utc_now(), model, json.dumps(ids), content))
        return rid

    def summaries(self):
        with self.connection() as db:
            db.row_factory = sqlite3.Row
            return [dict(r) for r in db.execute('SELECT * FROM summaries ORDER BY time DESC LIMIT 30')]


    def save_highlights(self, model, ids, content):
        rid=uuid.uuid4().hex
        with self.connection() as db:
            db.execute('INSERT INTO highlights VALUES (?,?,?,?,?)',
                       (rid,utc_now(),model,json.dumps(ids),json.dumps(content,ensure_ascii=False)))
        return rid

    def highlights(self):
        with self.connection() as db:
            db.row_factory=sqlite3.Row
            rows=[dict(r) for r in db.execute('SELECT * FROM highlights ORDER BY time DESC LIMIT 30')]
        for r in rows:
            r['source_ids']=json.loads(r['source_ids']);r['content']=json.loads(r['content'])
        return rows
