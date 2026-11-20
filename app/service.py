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

from .audio import (
    detect_impacts,
    parse_wav,
    validate_reference,
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
