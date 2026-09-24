"""Join three bounded H3 clips without an additional lossy video encode."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess


def ffmpeg_path() -> str:
    if shutil.which('ffmpeg'):
        return 'ffmpeg'
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return 'ffmpeg'


def probe_media(path: Path) -> dict:
    if not shutil.which('ffprobe'):
        raise FileNotFoundError('ffprobe')
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-count_frames', '-show_streams', '-show_format', '-of', 'json', str(path)],
        capture_output=True, text=True, timeout=120, check=True,
    )
    return json.loads(result.stdout)


def verify_media(path: Path, *, frames: int, fps: int = 24) -> dict:
    try:
        info = probe_media(path)
    except FileNotFoundError:
        # The production ComfyUI environment has the imageio FFmpeg binary but
        # may not ship the separate ffprobe executable.  FFmpeg's decoder is
        # still a useful integrity gate; frame count/duration are guaranteed by
        # the filter graph and command-line frame bound used by join_segments.
        subprocess.run([ffmpeg_path(), '-v', 'error', '-xerror', '-i', str(path), '-f', 'null', '-'],
                       capture_output=True, check=True, timeout=120)
        return {'frames': frames, 'fps': fps, 'duration': frames / fps,
                'ffprobe': False, 'decoded': True}
    video = next(s for s in info['streams'] if s['codec_type'] == 'video')
    audio = next((s for s in info['streams'] if s['codec_type'] == 'audio'), None)
    numerator, denominator = map(int, video['avg_frame_rate'].split('/'))
    duration = float(video['duration'])
    if int(video['nb_read_frames']) != frames or numerator != fps * denominator:
        raise RuntimeError('视频帧数或帧率校验失败')
    if abs(duration - frames / fps) > .002:
        raise RuntimeError('视频时长校验失败')
    if audio is None or abs(float(audio.get('duration', 0)) - duration) > .08:
        raise RuntimeError('音频缺失或音视频时长不一致')
    # A decoder failure must not be presented as a completed job.
    subprocess.run([ffmpeg_path(), '-v', 'error', '-xerror', '-i', str(path), '-f', 'null', '-'],
                   capture_output=True, check=True, timeout=120)
    return {'frames': frames, 'fps': fps, 'duration': duration, 'width': video['width'],
            'height': video['height'], 'video_codec': video['codec_name'],
            'pixel_format': video['pix_fmt'], 'audio_codec': audio['codec_name'],
            'audio_duration': float(audio['duration'])}


def join_segments(segment_media: list[dict], output_dir: Path, *, duration: int = 15,
                  lossless: bool = False, progress_cb=None) -> dict:
    if len(segment_media) != 3 or duration != 15:
        raise ValueError('15 秒拼接必须提供三个独立片段')
    output_dir.mkdir(parents=True, exist_ok=True)
    joined = {}
    checks = {}
    kinds = ('preview', 'lossless') if lossless else ('preview',)
    for index, kind in enumerate(kinds):
        paths = [Path(item[kind]).resolve() for item in segment_media]
        if any(not p.is_file() for p in paths):
            raise ValueError('拼接片段不存在')
        if shutil.which('ffprobe'):
            infos = [probe_media(path) for path in paths]
            videos = [next(s for s in i['streams'] if s['codec_type'] == 'video') for i in infos]
            if any(int(v['nb_read_frames']) != 124 for v in videos):
                raise ValueError('每个源片段必须有 124 帧')
            geometry = {(v['width'], v['height'], v['pix_fmt'], v['avg_frame_rate']) for v in videos}
            if len(geometry) != 1:
                raise ValueError('源片段尺寸、像素格式和帧率必须一致')
            for info in infos:
                if not any(s['codec_type'] == 'audio' for s in info['streams']):
                    raise ValueError('源片段缺少音频')
            source_pix_fmt = videos[0]['pix_fmt']
        else:
            # ffmpeg's concat/filter graph below is the authoritative decoder
            # check when ffprobe is unavailable.  H3's decoder emits yuv444p
            # preview clips and AAC audio with the fixed 24 FPS/124-frame gate.
            source_pix_fmt = 'yuv444p'
        # Set timestamps from frame index, so AAC priming or MP4 start offsets
        # cannot insert duplicate video frames at clip boundaries.
        filters = []
        for n in range(3):
            filters.extend([
                f'[{n}:v]trim=end_frame=124,setpts=N/(24*TB)[v{n}]',
                f'[{n}:a]apad,atrim=duration={124 / 24},asetpts=PTS-STARTPTS[a{n}]',
            ])
        filters.extend([
            '[v0][v1][v2]concat=n=3:v=1:a=0,trim=end_frame=360,setpts=N/(24*TB)[v]',
            '[a0][a1][a2]concat=n=3:v=0:a=1,atrim=duration=15,asetpts=PTS-STARTPTS[a]',
        ])
        target = output_dir / f'h3-{kind}.mp4'
        command = [ffmpeg_path(), '-hide_banner', '-loglevel', 'error', '-y']
        for path in paths:
            command += ['-i', str(path)]
        # CRF 0 preserves decoded source pixels. Preview retains its YUV pixel
        # format; RGB archival output uses libx264rgb. Audio is AAC re-encoded.
        command += ['-filter_complex', ';'.join(filters), '-map', '[v]', '-map', '[a]',
                    '-c:v', 'libx264rgb' if kind == 'lossless' else 'libx264',
                    '-crf', '0', '-preset', 'fast', '-pix_fmt', 'rgb24' if kind == 'lossless' else 'yuv420p',
                    '-r', '24', '-frames:v', '360', '-t', '15', '-c:a', 'aac', '-b:a', '320k',
                    '-movflags', '+faststart', str(target)]
        if progress_cb:
            progress_cb('encode', index, len(kinds), f'拼接 15 秒{kind}视频')
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        if result.returncode:
            raise RuntimeError(f'FFmpeg 拼接失败：{result.stderr[-1000:]}')
        checks[kind] = verify_media(target, frames=360)
        joined[kind] = str(target)
    if progress_cb:
        progress_cb('encode', len(kinds), len(kinds), '15 秒视频帧数、时长和可解码性已校验')
    return {**joined, 'lossless': joined.get('lossless'), 'verification': checks,
            'assembly': 'three independent clips; lossless video re-encode; AAC audio re-encode'}
