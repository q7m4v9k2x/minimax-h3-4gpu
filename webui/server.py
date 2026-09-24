"""HTTP interface for the H3 queue, progress stream and seekable video files."""
from __future__ import annotations

import json
import mimetypes
import os
from pathlib import Path
import queue
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

from jobs import JOB_ID, TERMINAL, Jobs, validate_request


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'H3Gateway/1'

    def _json(self, code, value):
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(payload)

    def _route(self):
        path = unquote(urlsplit(self.path).path)
        prefix = self.server.jobs.args.public_prefix.rstrip('/')
        if prefix and path.startswith(prefix + '/'):
            path = path[len(prefix):]
        return path.strip('/').split('/')

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        route = self._route()
        jobs = self.server.jobs
        try:
            if route in (['health'], ['healthz']):
                health = jobs.health()
                return self._json(200 if health['ready'] else 503, health)
            if route == ['jobs']:
                return self._json(200, {'data': jobs.recent()})
            if len(route) in {2, 3} and route[0] == 'jobs' and JOB_ID.fullmatch(route[1]):
                record = jobs.snapshot(route[1])
                if len(route) == 2:
                    return self._json(200, record)
                if route[2] == 'events' and self.command == 'GET':
                    return self._events(record['id'])
            if len(route) == 3 and route[0] == 'files' and JOB_ID.fullmatch(route[1]):
                record = jobs.snapshot(route[1])
                if record['status'] != 'completed' or route[2] not in [x['filename'] for x in record['data']]:
                    return self._json(404, {'error': '文件不存在'})
                return self._file(jobs.root / route[1] / route[2])
            if route in ([''], ['minimax-h3'], ['minimax-h3', '']):
                return self._file(Path(__file__).parent / 'h3.html')
            return self._json(404, {'error': '接口不存在'})
        except KeyError:
            return self._json(404, {'error': '任务不存在'})
        except (ConnectionResetError, BrokenPipeError):
            return

    def do_POST(self):
        jobs = self.server.jobs
        route = self._route()
        try:
            # Do not permit cross-site browser form submissions to a private gateway.
            origin = self.headers.get('Origin')
            if origin and urlsplit(origin).netloc != self.headers.get('Host'):
                self.close_connection = True
                return self._json(403, {'error': '不允许跨站提交'})
            length = int(self.headers.get('Content-Length', '0'))
            if length <= 0 or length > 65536:
                self.close_connection = True
                return self._json(413, {'error': '请求正文过大或为空'})
            if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                self.close_connection = True
                return self._json(415, {'error': '请使用 application/json'})
            raw = self.rfile.read(length)
            data = json.loads(raw)
            if route == ['v1', 'videos', 'generations']:
                request = validate_request(data)
                health = jobs.health()
                if not health['ready']:
                    return self._json(503, {'error': '生成后端尚未就绪', **health})
                return self._json(202, jobs.create(request))
            if len(route) == 3 and route[0] == 'jobs' and JOB_ID.fullmatch(route[1]) and route[2] == 'cancel':
                return self._json(200, jobs.cancel(route[1]))
            return self._json(404, {'error': '接口不存在'})
        except queue.Full:
            return self._json(429, {'error': '等待队列已满，请稍后重试'})
        except KeyError:
            return self._json(404, {'error': '任务不存在'})
        except (ValueError, UnicodeError) as error:
            return self._json(400, {'error': str(error)[:300]})
        except (ConnectionResetError, BrokenPipeError):
            return

    def _events(self, job_id):
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('X-Accel-Buffering', 'no')
        self.send_header('Connection', 'close')
        self.end_headers()
        self.close_connection = True
        jobs = self.server.jobs
        revision = -1
        while not jobs.stopping:
            record = jobs.snapshot(job_id)
            if record['revision'] != revision:
                payload = json.dumps(record, ensure_ascii=False)
                self.wfile.write(f'id: {record["revision"]}\nevent: job\ndata: {payload}\n\n'.encode())
                revision = record['revision']
            else:
                self.wfile.write(b': heartbeat\n\n')
            self.wfile.flush()
            if record['status'] in TERMINAL:
                return
            with jobs.wake:
                jobs.wake.wait(timeout=1)

    def _file(self, path):
        if not path.is_file():
            return self._json(404, {'error': '文件不存在'})
        size = path.stat().st_size
        start, end, code = 0, size - 1, 200
        requested = self.headers.get('Range')
        if requested:
            match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested)
            if not match or not any(match.groups()):
                return self._json(416, {'error': 'Range 无效'})
            left, right = match.groups()
            if left:
                start = int(left)
                end = min(int(right), size - 1) if right else size - 1
            else:
                start = max(0, size - int(right))
            if start > end or start >= size:
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{size}')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            code = 206
        self.send_response(code)
        self.send_header('Content-Type', mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
        self.send_header('Content-Length', str(end - start + 1))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('X-Content-Type-Options', 'nosniff')
        if path.suffix == '.html':
            self.send_header('Cache-Control', 'no-store')
        else:
            self.send_header('Cache-Control', 'private, max-age=31536000, immutable')
        if code == 206:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        if self.command == 'HEAD':
            return
        with path.open('rb') as stream:
            stream.seek(start)
            remaining = end - start + 1
            while remaining:
                chunk = stream.read(min(256 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def serve(args):
    jobs = Jobs(args)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.jobs = jobs
    print(json.dumps({'listening': f'{args.host}:{args.port}', 'verified': args.verified}), flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        jobs.close()
