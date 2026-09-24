"""Persistent bounded queue. One pipeline owns all four GPUs at a time."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import queue
import re
import secrets
import signal
import subprocess
import threading
import time

TERMINAL = {'completed', 'failed', 'cancelled', 'interrupted'}
STAGES = {'conditioning', 'loading', 'sampling', 'decode', 'encode', 'starting', 'queued'}
JOB_ID = re.compile(r'^[0-9a-f]{24}$')


def save_json(path: Path, data: dict) -> None:
    temp = path.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def validate_request(data: object) -> dict:
    if not isinstance(data, dict):
        raise ValueError('请求必须是 JSON 对象')
    prompt = data.get('prompt')
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 8000:
        raise ValueError('提示词长度须为 1 至 8000 个字符')
    values = {'width': 864, 'height': 480, 'frames': 124, 'steps': 30, 'seed': -1}
    for name, default in values.items():
        value = data.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f'{name} 必须为整数')
        values[name] = value
    if (values['width'], values['height']) not in {(864, 480), (480, 864), (1152, 640), (640, 1152)}:
        raise ValueError('当前支持 864×480、480×864、1152×640、640×1152')
    if values['frames'] != 124 or values['steps'] not in {20, 30}:
        raise ValueError('当前帧数固定 124，采样评估次数为 20 或 30')
    if not -1 <= values['seed'] <= 2**32 - 1:
        raise ValueError('seed 超出范围')
    if values['seed'] == -1:
        values['seed'] = secrets.randbits(32)
    lossless = data.get('lossless', False)
    if not isinstance(lossless, bool):
        raise ValueError('lossless 必须为布尔值')
    return {'prompt': prompt.strip(), **values, 'fps': 24, 'lossless': lossless}


class Jobs:
    def __init__(self, args):
        self.args = args
        self.root = args.data_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.wake = threading.Condition(self.lock)
        self.records = {}
        self.pending = queue.Queue(maxsize=args.queue_size)
        self.process = None
        self.active = None
        self.stopping = False
        self.health_lock = threading.Lock()
        self.health_cache = None
        self.health_time = 0
        for path in sorted(self.root.glob('*/job.json'), key=lambda p: p.stat().st_mtime):
            try:
                record = json.loads(path.read_text(encoding='utf-8'))
                if not JOB_ID.fullmatch(record['id']) or path.parent.name != record['id']:
                    continue
                if record['status'] == 'running':
                    record.update(status='interrupted', error='服务重启中断了此任务，请重新提交', updated=time.time())
                    save_json(path, record)
                self.records[record['id']] = record
                if record['status'] == 'queued':
                    try:
                        self.pending.put_nowait(record['id'])
                    except queue.Full:
                        record.update(status='interrupted', error='重启后队列容量不足，请重新提交')
                        save_json(path, record)
            except (OSError, ValueError, KeyError):
                continue
        self.worker = threading.Thread(target=self._work, name='h3-pipeline', daemon=True)
        self.worker.start()

    def health(self):
        with self.health_lock:
            if self.health_cache is None or time.monotonic() - self.health_time > 30:
                reasons = []
                script = Path(self.args.pipeline_script)
                if not script.is_file():
                    reasons.append('生成流水线尚未安装')
                else:
                    try:
                        result = subprocess.run([self.args.pipeline_python, str(script), '--check'],
                                                capture_output=True, text=True, timeout=15)
                        checks = json.loads(result.stdout.strip().splitlines()[-1])
                        if result.returncode or checks.get('ready') is not True:
                            reasons.extend(str(x) for x in checks.get('reasons', ['模型资源检查未通过']))
                    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
                        reasons.append('模型资源检查暂未通过')
                if not self.args.verified:
                    reasons.append('尚未完成真实提示词端到端验收')
                self.health_cache = {'ready': not reasons, 'reasons': reasons, 'model': 'minimax-h3',
                                     'fps': 24, 'frames': 124, 'max_queued': self.args.queue_size}
                self.health_time = time.monotonic()
            value = copy.deepcopy(self.health_cache)
        with self.lock:
            value.update(active_job=self.active,
                         queued=sum(r['status'] == 'queued' for r in self.records.values()))
        return value

    def _persist(self, record):
        record['updated'] = time.time()
        record['revision'] = record.get('revision', 0) + 1
        save_json(self.root / record['id'] / 'job.json', record)
        self.wake.notify_all()

    def create(self, request):
        with self.lock:
            if self.pending.full():
                raise queue.Full
            job_id = secrets.token_hex(12)
            directory = self.root / job_id
            directory.mkdir()
            now = time.time()
            record = {'id': job_id, 'status': 'queued', 'created': now, 'updated': now,
                      'request': request, 'progress': {'stage': 'queued', 'current': 0, 'total': 0},
                      'data': [], 'revision': 0}
            save_json(directory / 'request.json', request)
            self.records[job_id] = record
            self._persist(record)
            self.pending.put_nowait(job_id)
            return self.snapshot(job_id)

    def snapshot(self, job_id):
        with self.lock:
            record = copy.deepcopy(self.records[job_id])
            queued = [r['id'] for r in self.records.values() if r['status'] == 'queued']
            record['queue_position'] = queued.index(job_id) + 1 if job_id in queued else 0
            record['status_url'] = f'{self.args.public_prefix}/jobs/{job_id}'
            record['events_url'] = f'{self.args.public_prefix}/jobs/{job_id}/events'
            record['server_time'] = time.time()
            return record

    def recent(self, limit=50):
        with self.lock:
            ordered = sorted(self.records.values(), key=lambda r: r['created'], reverse=True)[:limit]
            return [self.snapshot(r['id']) for r in ordered]

    def cancel(self, job_id):
        with self.lock:
            record = self.records[job_id]
            if record['status'] in TERMINAL:
                return self.snapshot(job_id)
            record.update(status='cancelled', finished=time.time())
            if self.active == job_id and self.process:
                self._terminate(self.process)
            self._persist(record)
            return self.snapshot(job_id)

    @staticmethod
    def _terminate(process):
        try:
            if os.name == 'posix':
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            pass

    def _event(self, record, event):
        if event.get('event') != 'progress' or event.get('stage') not in STAGES:
            return
        current, total = event.get('current', 0), event.get('total', 0)
        if not isinstance(current, (int, float)) or not isinstance(total, (int, float)):
            return
        if not (0 <= current <= 10**9 and 0 <= total <= 10**9):
            return
        with self.lock:
            if record['status'] != 'running':
                return
            old_stage = record['progress'].get('stage')
            record['progress'] = {'stage': event['stage'], 'current': current, 'total': total,
                                  'unit': str(event.get('unit', ''))[:40],
                                  'detail': str(event.get('detail', ''))[:300],
                                  'stage_started': (record['progress'].get('stage_started', time.time())
                                                    if old_stage == event['stage'] else time.time())}
            self._persist(record)

    def _result(self, job_id):
        directory = self.root / job_id
        result = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
        output = []
        for key in ('video', 'lossless', 'poster'):
            if not result.get(key):
                continue
            path = (directory / result[key]).resolve()
            if path.parent != directory or not path.is_file() or path.stat().st_size == 0:
                raise ValueError('输出文件不存在或位置无效')
            if key == 'video' and path.suffix.lower() != '.mp4':
                raise ValueError('预览输出必须为 MP4')
            output.append({'kind': key, 'url': f'{self.args.public_prefix}/files/{job_id}/{path.name}',
                           'filename': path.name, 'bytes': path.stat().st_size})
        if not any(item['kind'] == 'video' for item in output):
            raise ValueError('流水线没有生成视频文件')
        return output

    def _work(self):
        while not self.stopping:
            try:
                job_id = self.pending.get(timeout=1)
            except queue.Empty:
                continue
            record = self.records[job_id]
            try:
                with self.lock:
                    if record['status'] != 'queued':
                        continue
                    record.update(status='running', started=time.time(),
                                  progress={'stage': 'starting', 'current': 0, 'total': 0})
                    self.active = job_id
                    self._persist(record)
                directory = self.root / job_id
                command = [self.args.pipeline_python, self.args.pipeline_script,
                           '--request-json', str(directory / 'request.json'), '--output-dir', str(directory)]
                with (directory / 'pipeline.log').open('w', encoding='utf-8') as log:
                    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                               text=True, encoding='utf-8', errors='replace', bufsize=1,
                                               start_new_session=os.name == 'posix',
                                               env={**os.environ, 'PYTHONUNBUFFERED': '1'})
                    with self.lock:
                        self.process = process
                        if record['status'] == 'cancelled':
                            self._terminate(process)
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        try:
                            event = json.loads(line)
                            if isinstance(event, dict):
                                self._event(record, event)
                        except (ValueError, TypeError):
                            pass
                    code = process.wait()
                with self.lock:
                    if record['status'] == 'cancelled':
                        continue
                if code:
                    raise RuntimeError(f'生成程序退出（代码 {code}），请检查工作站任务日志')
                output = self._result(job_id)
                with self.lock:
                    record.update(status='completed', data=output, finished=time.time(),
                                  progress={'stage': 'encode', 'current': 1, 'total': 1})
                    self._persist(record)
            except Exception as error:
                with self.lock:
                    if record['status'] != 'cancelled':
                        record.update(status='failed', error=str(error)[:500], finished=time.time())
                        self._persist(record)
            finally:
                with self.lock:
                    self.process = None
                    self.active = None
                    self.wake.notify_all()
                self.pending.task_done()

    def close(self):
        self.stopping = True
        with self.lock:
            if self.process:
                self._terminate(self.process)
        self.worker.join(timeout=10)
