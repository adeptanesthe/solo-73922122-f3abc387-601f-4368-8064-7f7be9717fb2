"""WAV 解析与冲击（冲次）检测。

输入要求：16 kHz、16-bit PCM、单声道 WAV，时长不超过 8 秒。
检测基于短时能量包络的自适应阈值峰值拾取，并用非极大值抑制
剔除 40 ms 内的回声（回声不算多冲）。
"""
from __future__ import annotations

import io
import wave
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2  # 16-bit PCM
CHANNELS = 1
MAX_DURATION_SEC = 8.0

# 能量帧参数
FRAME_SEC = 0.010       # 10 ms 帧长
HOP_SEC = 0.005         # 5 ms 帧移
# 峰值拾取参数
MIN_IMPACT_INTERVAL_SEC = 0.040   # 40 ms 内回声不算多冲
ABS_RATIO = 0.15        # 相对全段最大能量的最低能量门限
PROMINENCE_RATIO = 2.0  # 相对局部背景能量的突出倍数
BACKGROUND_SEC = 0.030  # 局部背景统计窗口
REF_MIN_IMPACTS = 4
REF_MAX_IMPACTS = 8
REF_MIN_GAP_SEC = 0.150


class AudioValidationError(ValueError):
    """音频文件不符合约定格式或长度。"""


@dataclass
class ImpactDetection:
    times: list[float]            # 冲击时刻（秒）
    suppressed_echoes: list[float]  # 被 40ms 回声规则抑制的时刻
    duration: float
    peak_energy: float


def parse_wav(data: bytes) -> np.ndarray:
    """校验并解析 WAV，返回 float32、范围 [-1, 1] 的单声道波形。"""
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            n_frames = wf.getnframes()
            frames = wf.readframes(n_frames)
    except (wave.Error, EOFError, OSError) as exc:
        raise AudioValidationError(f"无法解析 WAV 文件：{exc}") from exc

    if n_channels != CHANNELS:
        raise AudioValidationError(f"只支持单声道 WAV，当前声道数：{n_channels}")
    if sample_width != SAMPLE_WIDTH:
        raise AudioValidationError(
            f"只支持 16-bit PCM，当前采样位深：{sample_width * 8}-bit"
        )
    if frame_rate != SAMPLE_RATE:
        raise AudioValidationError(
            f"只支持 {SAMPLE_RATE} Hz 采样率，当前：{frame_rate} Hz"
        )

    duration = n_frames / frame_rate
    if duration > MAX_DURATION_SEC + 1e-9:
        raise AudioValidationError(
            f"录音时长 {duration:.2f}s 超过上限 {MAX_DURATION_SEC:.0f}s"
        )
    if n_frames == 0:
        raise AudioValidationError("录音为空，没有任何采样点")

    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if samples.shape[0] != n_frames:
        raise AudioValidationError("WAV 数据块长度与声明帧数不一致，文件可能损坏")
    return samples


def _energy_envelope(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame_len = int(round(FRAME_SEC * SAMPLE_RATE))
    hop = int(round(HOP_SEC * SAMPLE_RATE))
    if len(samples) < frame_len:
        # 极短录音也给单帧
        frame_len = len(samples)
        hop = max(1, frame_len)
    n = 1 + (len(samples) - frame_len) // hop
    energies = np.empty(n, dtype=np.float32)
    centers = np.empty(n, dtype=np.float32)
    for k in range(n):
        start = k * hop
        frame = samples[start : start + frame_len]
        energies[k] = float(np.mean(frame * frame))
        centers[k] = (start + frame_len / 2) / SAMPLE_RATE
    return centers, energies


def detect_impacts(samples: np.ndarray) -> ImpactDetection:
    """从波形提取冲击时刻。

    步骤：短时能量包络 -> 自适应门限（兼顾响度整体变化）->
    非极大值峰值拾取 -> 40ms 回声抑制。
    """
    centers, energies = _energy_envelope(samples)
    if len(energies) == 0:
        return ImpactDetection([], [], len(samples) / SAMPLE_RATE, 0.0)

    peak_energy = float(np.max(energies))
    if peak_energy <= 0.0:
        return ImpactDetection([], [], len(samples) / SAMPLE_RATE, 0.0)

    # 自适应门限：绝对门限跟随录音自身最大能量，对整体响度变化不敏感
    abs_threshold = ABS_RATIO * peak_energy
    # 背景能量取较低分位，避免静音段把门槛压得过低
    bg = float(np.median(energies))
    threshold = max(abs_threshold, PROMINENCE_RATIO * bg)

    bg_frames = max(1, int(BACKGROUND_SEC / HOP_SEC))
    threshold_frames: list[int] = []
    for i in range(len(energies)):
        lo = max(0, i - bg_frames)
        hi = min(len(energies), i + bg_frames + 1)
        local_bg = float(np.median(energies[lo:hi]))
        e = float(energies[i])
        if e < threshold:
            continue
        if e < PROMINENCE_RATIO * max(local_bg, 1e-12):
            continue
        threshold_frames.append(i)

    # 局部极大值：只保留峰点，冲击衰减拖尾不会产生伪峰；
    # 平台区取最早帧，保证冲击时刻不向后偏
    peak_frames: list[int] = []
    local_win = max(1, int(0.010 / HOP_SEC))  # ±10ms
    for i in threshold_frames:
        lo = max(0, i - local_win)
        hi = min(len(energies), i + local_win + 1)
        if i == lo + int(np.argmax(energies[lo:hi])):
            peak_frames.append(i)

    # 40ms 不应期（回声抑制）：窗口内只保留能量最大的峰，
    # 其余局部极大值记为被抑制的回声
    min_interval_frames = max(1, int(round(MIN_IMPACT_INTERVAL_SEC / HOP_SEC)))
    kept_idx: list[int] = []
    suppressed: list[float] = []
    for i in peak_frames:
        if kept_idx and i - kept_idx[-1] < min_interval_frames:
            last = kept_idx[-1]
            if energies[i] > energies[last]:
                # 新峰更强：替换已保留峰，旧峰改记为回声
                suppressed.append(float(centers[last]))
                kept_idx[-1] = i
            else:
                suppressed.append(float(centers[i]))
            continue
        kept_idx.append(i)
    candidate_times = [float(centers[i]) for i in kept_idx]

    return ImpactDetection(
        times=candidate_times,
        suppressed_echoes=suppressed,
        duration=len(samples) / SAMPLE_RATE,
        peak_energy=peak_energy,
    )


def validate_reference(det: ImpactDetection) -> None:
    """参考录音必须含 4~8 次清晰冲击，且相邻间隔至少 150ms。"""
    n = len(det.times)
    if not (REF_MIN_IMPACTS <= n <= REF_MAX_IMPACTS):
        raise AudioValidationError(
            f"参考录音应包含 {REF_MIN_IMPACTS}~{REF_MAX_IMPACTS} 次清晰冲击，"
            f"当前检测到 {n} 次，请更换参考录音"
        )
    gaps = np.diff(np.asarray(det.times))
    if len(gaps) and float(np.min(gaps)) < REF_MIN_GAP_SEC - 1e-9:
        raise AudioValidationError(
            "参考录音中存在相邻冲击间隔小于 150ms 的情况，无法作为节奏基准"
        )
