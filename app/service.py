"""检查服务与结果存储。

无效音频（格式错误、参考不达标等）直接抛错，由 API 层返回 400，
且完全不触碰已保存的上一次有效结果。
"""
from __future__ import annotations

import hashlib
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .audio import (
    SAMPLE_RATE,
    detect_impacts,
    parse_wav,
    validate_reference,
)
from .decay import (
    END_LEVEL_DEEP_DB,
    END_LEVEL_SHALLOW_DB,
    TimeWindow,
    compare_tails,
)
from .matching import align_impacts

MAX_UPLOAD_BYTES = 16_000 * 2 * 8 + 4096  # 8 秒 16k/16bit/单声道 + WAV 头余量


class Store:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.uploads_dir = data_dir / "uploads"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last_result: dict[str, Any] | None = None
        self._last_comparison: dict[str, Any] | None = None

    def _save_wav(self, raw: bytes, prefix: str) -> str:
        digest = hashlib.sha256(raw).hexdigest()[:16]
        name = f"{prefix}_{digest}.wav"
        path = self.uploads_dir / name
        if not path.exists():
            path.write_bytes(raw)
        return f"/data/uploads/{name}"

    def inspect(self, reference_raw: bytes, test_raw: bytes) -> dict[str, Any]:
        # 先完整解析与检测，任何一步抛错都不会写入状态
        ref_samples = parse_wav(reference_raw)
        test_samples = parse_wav(test_raw)

        ref_det = detect_impacts(ref_samples)
        validate_reference(ref_det)  # 参考不达标 -> AudioValidationError
        test_det = detect_impacts(test_samples)

        alignment = align_impacts(ref_det.times, test_det.times)

        matched_ref = {i for i, _ in alignment.matched}
        matched_test = {j for _, j in alignment.matched}
        ref_url = self._save_wav(reference_raw, "ref")
        test_url = self._save_wav(test_raw, "test")

        reference_impacts = [
            {
                "index": i,
                "time": round(t, 4),
                "matched": i in matched_ref,
            }
            for i, t in enumerate(ref_det.times)
        ]
        test_impacts = [
            {
                "index": j,
                "time": round(t, 4),
                "matched": j in matched_test,
                "is_extra": j in alignment.extra_test_indices,
            }
            for j, t in enumerate(test_det.times)
        ]

        missed = [
            {
                "ref_index": i,
                "ref_time": round(ref_det.times[i], 4),
                "expected_time": round(
                    alignment.expected_missed_times.get(i, ref_det.times[i]), 4
                ),
            }
            for i in alignment.missed_ref_indices
        ]
        extras = [
            {
                "test_index": j,
                "time": round(test_det.times[j], 4),
            }
            for j in alignment.extra_test_indices
        ]
        matches = [
            {
                "ref_index": i,
                "test_index": j,
                "ref_time": round(ref_det.times[i], 4),
                "test_time": round(test_det.times[j], 4),
            }
            for i, j in alignment.matched
        ]

        verdict = "ok"
        if alignment.ambiguous:
            verdict = "ambiguous"
        elif alignment.miss_count or alignment.extra_count:
            verdict = "defect"

        warnings: list[str] = []
        if not test_det.times:
            warnings.append("待检录音未检测到任何冲击，全部参考冲次按漏冲处理")
        if test_det.suppressed_echoes:
            warnings.append(
                f"已抑制 {len(test_det.suppressed_echoes)} 个 40ms 内回声，不计多冲"
            )
        if alignment.ambiguous:
            warnings.append("存在多种同样符合节奏与最少漏/多冲规则的归属，请人工复听")

        result = {
            "id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "verdict": verdict,
            "ambiguous": alignment.ambiguous,
            "optimal_way_count": alignment.optimal_way_count,
            "summary": {
                "reference_impact_count": len(ref_det.times),
                "test_impact_count": len(test_det.times),
                "matched_count": len(alignment.matched),
                "miss_count": alignment.miss_count,
                "extra_count": alignment.extra_count,
                "suppressed_echo_count": len(test_det.suppressed_echoes),
            },
            "reference": {
                "url": ref_url,
                "duration": round(ref_det.duration, 4),
                "impacts": reference_impacts,
            },
            "test": {
                "url": test_url,
                "duration": round(test_det.duration, 4),
                "impacts": test_impacts,
            },
            "matches": matches,
            "missed": missed,
            "extras": extras,
            "timing_ratios": [round(r, 4) for r in alignment.timing_ratios],
            "warnings": warnings,
        }

        with self._lock:
            self._last_result = result
        return result

    def get_last(self) -> dict[str, Any] | None:
        with self._lock:
            return None if self._last_result is None else dict(self._last_result)

    def clear(self) -> None:
        with self._lock:
            self._last_result = None

    def compare(
        self,
        before_raw: bytes,
        after_raw: bytes,
        low_hz: float,
        high_hz: float,
        before_impact: TimeWindow,
        before_background: TimeWindow,
        after_impact: TimeWindow,
        after_background: TimeWindow,
    ) -> dict[str, Any]:
        """对比维修前后冲击尾音的衰减速度。

        任一份录音或圈定条件不合格都会抛错，且不写入新的对比结果，
        由 API 层撤下旧对比并返回 400。
        """
        before_samples = parse_wav(before_raw)
        after_samples = parse_wav(after_raw)
        before_measure, after_measure, common_level = compare_tails(
            before_samples,
            after_samples,
            low_hz,
            high_hz,
            before_impact,
            before_background,
            after_impact,
            after_background,
        )

        before_url = self._save_wav(before_raw, "before")
        after_url = self._save_wav(after_raw, "after")

        def side(url: str, samples: np.ndarray, measure) -> dict[str, Any]:
            end_time = measure.end_time(common_level)
            if end_time is None:  # compare_tails 已保证双方可测，此处仅防御
                raise RuntimeError(
                    f"{common_level} dB 终点在服务层不可用，对比结果未写入"
                )
            return {
                "url": url,
                "duration": round(len(samples) / SAMPLE_RATE, 4),
                "peak_time": round(measure.peak_time, 4),
                "noise_floor_db": round(measure.noise_floor_db, 2),
                "gate_time": (
                    None if measure.gate_time is None else round(measure.gate_time, 4)
                ),
                "start_time": round(measure.start_time, 4),
                "end_time": round(end_time, 4),
                "elapsed": round(end_time - measure.start_time, 4),
                "deepest_level_db": measure.deepest_level_db,
            }

        before_elapsed = before_measure.end_time(common_level) - before_measure.start_time
        after_elapsed = after_measure.end_time(common_level) - after_measure.start_time
        if abs(before_elapsed - after_elapsed) <= 1e-9:
            slower = None
        else:
            slower = "before" if before_elapsed > after_elapsed else "after"

        result = {
            "id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "band": {"low_hz": low_hz, "high_hz": high_hz},
            "common_level_db": common_level,
            "level_range_db": [END_LEVEL_DEEP_DB, END_LEVEL_SHALLOW_DB],
            "before": side(before_url, before_samples, before_measure),
            "after": side(after_url, after_samples, after_measure),
            "slower": slower,
            "warnings": [
                "测量使用 10ms 帧、5ms 帧移的待测频带能量；"
                "以各冲击时段峰值为 0 dB、背景段能量中位数为底噪",
                "峰后只测量到冲击时段窗末或首个高出底噪不足 6 dB 的帧之前",
                "连续 3 帧不高于目标电平才算过线，时间取这 3 帧的第一帧；"
                "两份录音均从 −5 dB 过线时刻计时",
                "共用终点取 −15～−25 dB 之间双方都能测到的最深整 dB 电平，"
                "且终点须晚于各自的计时起点",
            ],
        }

        with self._lock:
            self._last_comparison = result
        return result

    def get_last_comparison(self) -> dict[str, Any] | None:
        with self._lock:
            return (
                None
                if self._last_comparison is None
                else dict(self._last_comparison)
            )

    def clear_comparison(self) -> bool:
        """撤下旧对比；返回此前是否存在旧结果。"""
        with self._lock:
            existed = self._last_comparison is not None
            self._last_comparison = None
            return existed
