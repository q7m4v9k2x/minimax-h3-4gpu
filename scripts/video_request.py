"""Shared HTTP/CLI contract for bounded H3 generation requests."""
from __future__ import annotations

import secrets


def normalize_request(data: object) -> dict:
    if not isinstance(data, dict):
        raise ValueError('请求必须是 JSON 对象')
    prompt = data.get('prompt')
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 8000:
        raise ValueError('提示词长度须为 1 至 8000 个字符')
    values = {'width': 864, 'height': 480, 'frames': 124, 'steps': 20, 'seed': -1}
    for name, default in values.items():
        value = data.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f'{name} 必须为整数')
        values[name] = value
    if (values['width'], values['height']) not in {(864, 480), (480, 864), (1152, 640), (640, 1152)}:
        raise ValueError('当前支持 864×480、480×864、1152×640、640×1152')
    if values['frames'] != 124 or values['steps'] not in {20, 30}:
        raise ValueError('每段固定 124 帧，采样评估次数为 20 或 30；15 秒请传 duration=15')
    if not -1 <= values['seed'] <= 2**32 - 1:
        raise ValueError('seed 超出范围')
    if values['seed'] == -1:
        values['seed'] = secrets.randbits(32)
    duration = data.get('duration', 15 if data.get('segments') == 3 else 5)
    segments = data.get('segments', 3 if duration == 15 else 1)
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration not in {5, 15}:
        raise ValueError('duration 必须为 5 或 15 秒')
    if isinstance(segments, bool) or not isinstance(segments, int) or segments not in {1, 3}:
        raise ValueError('segments 必须为 1 或 3')
    if (duration, segments) not in {(5, 1), (15, 3)}:
        raise ValueError('5 秒对应 1 段，15 秒对应 3 段')
    lossless = data.get('lossless', False)
    if not isinstance(lossless, bool):
        raise ValueError('lossless 必须为布尔值')
    return {'prompt': prompt.strip(), **values, 'fps': 24, 'lossless': lossless,
            'duration': int(duration), 'segments': segments}
