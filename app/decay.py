"""维修前后冲击尾音衰减速度对比。

每份录音由核对员圈定两段：
- 冲击+尾音时段：其中最强的一帧作为 0 dB 峰值；
- 无冲击背景段：段内频带能量的中位数作为底噪。

分析只使用「短时频带能量」：10 ms Hann 帧、5 ms 帧移，把待测频带内
各 FFT 频点的功率求和作为该帧的频带能量。峰后测量截止到冲击时段窗末，
或首个高出底噪不足 6 dB 的帧之前。

过线规则：连续 3 帧不高于目标电平才算过线，时刻取这 3 帧中的第一帧。
两段录音都从各自 -5 dB 过线时刻起计时；在 -15 ~ -25 dB 之间从深到浅
寻找双方都能测到的最深整 dB 电平，作为共用终点，且终点须晚于各自起点。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import SAMPLE_RATE

FRAME_SEC = 0.010
HOP_SEC = 0.005
HOLD_FRAMES = 3              # 连续 3 帧不高于目标电平才算过线
START_LEVEL_DB = -5.0        # 计时起点电平
END_LEVEL_DEEP_DB = -25      # 共用终点的最深候选
END_LEVEL_SHALLOW_DB = -15   # 共用终点的最浅候选
GATE_SNR_DB = 6.0            # 高出底噪不足 6 dB 即截止测量
BG_MAX_ABOVE_PEAK_DB = -6.0  # 背景段峰值至少比冲击峰值低 6 dB
MIN_REGION_FRAMES = HOLD_FRAMES + 1


class DecayValidationError(ValueError):
    """圈定条件不足，无法完成衰减对比。"""


@dataclass(frozen=True)
class TimeWindow:
    start: float
    end: float


@dataclass
class TailMeasurement:
    peak_time: float
    noise_floor_db: float          # 底噪相对峰值电平（dB）
    gate_time: float | None        # 因高出底噪不足 6 dB 而截止的帧时刻
    start_time: float              # -5 dB 过线时刻
    deepest_level_db: int | None   # 该录音在 -15~-25 dB 内最深可测整 dB
    end_times: dict[int, float]    # 各候选电平的终点（须晚于起点）

    def end_time(self, level_db: int) -> float | None:
        return self.end_times.get(level_db)


def _validate_band(low_hz: float, high_hz: float) -> tuple[int, int]:
    if not (np.isfinite(low_hz) and np.isfinite(high_hz)):
        raise DecayValidationError("待测频带必须是有限数值")
    if low_hz <= 0 or high_hz <= low_hz:
        raise DecayValidationError(
            f"待测频带无效：低端 {low_hz:g} Hz 应大于 0 且小于高端 {high_hz:g} Hz"
        )
    nyquist = SAMPLE_RATE / 2
    if high_hz > nyquist:
        raise DecayValidationError(
            f"待测频带上限 {high_hz:g} Hz 超过 {SAMPLE_RATE} Hz 采样的"
            f"奈奎斯特频率 {nyquist:g} Hz"
        )
    if round(high_hz) <= round(low_hz):
        raise DecayValidationError(
            f"待测频带过窄：{low_hz:g}~{high_hz:g} Hz 取整后没有可用频点，请加宽频带"
        )
    return int(round(low_hz)), int(round(high_hz))


def _validate_window(label: str, win: TimeWindow, duration: float) -> None:
    if not (np.isfinite(win.start) and np.isfinite(win.end)):
        raise DecayValidationError(f"{label}的时刻必须是有限数值")
    if win.start < 0 or win.end <= win.start:
        raise DecayValidationError(
            f"{label}无效：起点 {win.start:.3f}s 应不小于 0 且早于终点 {win.end:.3f}s"
        )
    if win.end > duration + 1e-9:
        raise DecayValidationError(
            f"{label}终点 {win.end:.3f}s 超出录音时长 {duration:.3f}s"
        )


def band_energies(
    samples: np.ndarray, low_hz: float, high_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    """返回帧中心时刻与待测频带内的短时能量。"""
    low, high = _validate_band(low_hz, high_hz)
    frame_len = int(round(FRAME_SEC * SAMPLE_RATE))
    hop = int(round(HOP_SEC * SAMPLE_RATE))
    if len(samples) < frame_len:
        raise DecayValidationError(
            f"录音过短，不足 {FRAME_SEC * 1000:.0f} ms 的一帧"
        )

    freqs = np.fft.rfftfreq(frame_len, 1 / SAMPLE_RATE)
    band = (freqs >= low) & (freqs <= high)
    if not np.any(band):
        raise DecayValidationError(
            f"待测频带 {low}~{high} Hz 内没有可用频点，请加宽频带"
        )

    window = np.hanning(frame_len).astype(np.float32)
    n_frames = 1 + (len(samples) - frame_len) // hop
    energies = np.empty(n_frames, dtype=np.float64)
    times = np.empty(n_frames, dtype=np.float64)
    for k in range(n_frames):
        start = k * hop
        spectrum = np.fft.rfft(samples[start : start + frame_len] * window)
        energies[k] = float(np.sum(np.abs(spectrum[band]) ** 2))
        times[k] = (start + frame_len / 2) / SAMPLE_RATE
    return times, energies


def _frames_in_window(times: np.ndarray, win: TimeWindow) -> np.ndarray:
    return np.flatnonzero((times >= win.start - 1e-9) & (times <= win.end + 1e-9))


def _crossing_runs(
    times: np.ndarray, levels: np.ndarray, target_db: float
) -> list[float]:
    """返回所有「连续不高于目标电平」帧组的首帧时刻。"""
    runs: list[float] = []
    run_start: float | None = None
    run_len = 0
    for t, level in zip(times, levels):
        if level <= target_db + 1e-9:
            if run_len == 0:
                run_start = float(t)
            run_len += 1
        else:
            if run_len >= HOLD_FRAMES:
                runs.append(run_start)
            run_start = None
            run_len = 0
    if run_len >= HOLD_FRAMES:
        runs.append(run_start)
    return runs


def _first_hold_crossing(
    times: np.ndarray,
    levels: np.ndarray,
    target_db: float,
    after_time: float | None = None,
) -> float | None:
    """连续 HOLD_FRAMES 帧不高于 target 时，返回这组帧的首帧时刻。

    after_time 给定时，只接受整组帧都严格晚于该时刻的过线：
    若尾部在起点之前就已低于目标电平，不算起点之后的新过线。
    """
    for run_start in _crossing_runs(times, levels, target_db):
        if after_time is None or run_start > after_time + 1e-9:
            return run_start
    return None


def measure_tail(
    samples: np.ndarray,
    low_hz: float,
    high_hz: float,
    impact_window: TimeWindow,
    background_window: TimeWindow,
    label: str,
) -> TailMeasurement:
    low, high = _validate_band(low_hz, high_hz)
    duration = len(samples) / SAMPLE_RATE
    _validate_window(f"{label}冲击时段", impact_window, duration)
    _validate_window(f"{label}背景时段", background_window, duration)
    if (
        background_window.start < impact_window.end - 1e-9
        and background_window.end > impact_window.start + 1e-9
    ):
        raise DecayValidationError(
            f"{label}的冲击时段与背景时段重叠，请圈定互不重叠的两段"
        )

    times, energies = band_energies(samples, low, high)
    impact_idx = _frames_in_window(times, impact_window)
    bg_idx = _frames_in_window(times, background_window)
    if len(impact_idx) < MIN_REGION_FRAMES:
        raise DecayValidationError(
            f"{label}冲击时段过短：仅覆盖 {len(impact_idx)} 帧，"
            f"至少需要 {MIN_REGION_FRAMES} 帧（约 "
            f"{(MIN_REGION_FRAMES - 1) * HOP_SEC + FRAME_SEC:.2f}s）"
        )
    if len(bg_idx) < MIN_REGION_FRAMES:
        raise DecayValidationError(
            f"{label}背景时段过短：仅覆盖 {len(bg_idx)} 帧，"
            f"至少需要 {MIN_REGION_FRAMES} 帧（约 "
            f"{(MIN_REGION_FRAMES - 1) * HOP_SEC + FRAME_SEC:.2f}s）"
        )

    local_peak = float(np.max(energies[impact_idx]))
    if local_peak <= 0.0:
        raise DecayValidationError(f"{label}冲击时段内没有可测能量，请重新圈定")

    noise_floor = float(np.median(energies[bg_idx]))
    if noise_floor <= 0.0:
        raise DecayValidationError(f"{label}背景时段能量为 0，无法估计底噪")

    bg_max = float(np.max(energies[bg_idx]))
    bg_max_db = 10 * np.log10(max(bg_max, 1e-300) / local_peak)
    if bg_max_db > BG_MAX_ABOVE_PEAK_DB + 1e-9:
        raise DecayValidationError(
            f"{label}的背景时段疑似包含冲击：段内峰值仅比冲击峰值低 "
            f"{-bg_max_db:.1f} dB，应至少低 {-BG_MAX_ABOVE_PEAK_DB:.0f} dB，"
            "请改圈一段无冲击背景声"
        )

    noise_floor_db = 10 * np.log10(max(noise_floor, 1e-300) / local_peak)
    if noise_floor_db > -GATE_SNR_DB:
        raise DecayValidationError(
            f"{label}冲击峰值仅高出底噪 {-noise_floor_db:.1f} dB，"
            f"不足 {GATE_SNR_DB:.0f} dB，无法分辨尾音衰减，请更换录音"
        )
    # 帧级电平，以冲击时段峰值为 0 dB
    levels = 10 * np.log10(np.maximum(energies, 1e-300) / local_peak)

    peak_pos = int(np.argmax(energies[impact_idx]))
    peak_idx = int(impact_idx[peak_pos])
    peak_time = float(times[peak_idx])

    # 峰后有效测量区：冲击窗末或首个高出底噪不足 6 dB 的帧之前
    gate_level_db = noise_floor_db + GATE_SNR_DB
    gate_idx: int | None = None
    for i in range(peak_idx + 1, int(impact_idx[-1]) + 1):
        if levels[i] < gate_level_db - 1e-9:
            gate_idx = i
            break
    meas_end = int(impact_idx[-1]) if gate_idx is None else gate_idx - 1
    if meas_end - peak_idx < HOLD_FRAMES:
        raise DecayValidationError(
            f"{label}冲击后有效测量帧不足 {HOLD_FRAMES} 帧："
            "尾音在底噪 6 dB 以内过早截止，无法测量衰减"
        )

    meas_idx = np.arange(peak_idx, meas_end + 1)
    meas_times = times[meas_idx]
    meas_levels = levels[meas_idx]

    start_time = _first_hold_crossing(meas_times, meas_levels, START_LEVEL_DB)
    if start_time is None:
        raise DecayValidationError(
            f"{label}冲击后未见 −5 dB 过线：峰后尾音没有衰减到 −5 dB，"
            "请检查冲击/尾音时段或更换录音"
        )

    end_times: dict[int, float] = {}
    deepest: int | None = None
    for level_db in range(END_LEVEL_DEEP_DB, END_LEVEL_SHALLOW_DB - 1, 1):
        t_end = _first_hold_crossing(
            meas_times, meas_levels, float(level_db), after_time=start_time
        )
        if t_end is not None and t_end > start_time + 1e-9:
            end_times[level_db] = t_end
            if deepest is None:
                deepest = level_db

    return TailMeasurement(
        peak_time=peak_time,
        noise_floor_db=float(noise_floor_db),
        gate_time=None if gate_idx is None else float(times[gate_idx]),
        start_time=start_time,
        deepest_level_db=deepest,
        end_times=end_times,
    )


def _fmt_level(level: int | None) -> str:
    return "无" if level is None else f"{level} dB"


def compare_tails(
    before_samples: np.ndarray,
    after_samples: np.ndarray,
    low_hz: float,
    high_hz: float,
    before_impact: TimeWindow,
    before_background: TimeWindow,
    after_impact: TimeWindow,
    after_background: TimeWindow,
) -> tuple[TailMeasurement, TailMeasurement, int]:
    """分析两份录音并选出双方都能测到的最深整 dB 共用终点。"""
    low, high = _validate_band(low_hz, high_hz)
    before = measure_tail(
        before_samples, low, high, before_impact, before_background, "维修前录音"
    )
    after = measure_tail(
        after_samples, low, high, after_impact, after_background, "维修后录音"
    )

    common_level: int | None = None
    for level_db in range(END_LEVEL_DEEP_DB, END_LEVEL_SHALLOW_DB - 1, 1):
        if (
            before.end_time(level_db) is not None
            and after.end_time(level_db) is not None
        ):
            common_level = level_db
            break

    if common_level is None:
        raise DecayValidationError(
            "两份录音在 −15～−25 dB 内没有共同可测的整 dB 终点"
            f"（维修前最深可测 {_fmt_level(before.deepest_level_db)}，"
            f"维修后最深可测 {_fmt_level(after.deepest_level_db)}）；"
            "终点须晚于各自的 −5 dB 起点，衰减过快或尾音在 6 dB 截止前"
            "未能走到 −15 dB 都无法计时，请延长冲击+尾音时段或更换录音"
        )
    return before, after, common_level
