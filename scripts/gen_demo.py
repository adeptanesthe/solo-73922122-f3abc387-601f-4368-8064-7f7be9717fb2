"""生成演示用 WAV（16kHz/16bit/单声道），供手工冒烟测试。

用法：python -I scripts/gen_demo.py <输出目录>
生成参考录音及四种待检录音（正常/漏冲/多冲/歧义）。
"""
from __future__ import annotations

import io
import sys
import wave
from pathlib import Path

import numpy as np

SR = 16_000


def synth(times, duration, gain=0.8, noise=0.002, seed=7, echoes=None):
    rng = np.random.default_rng(seed)
    n = int(duration * SR)
    x = rng.standard_normal(n).astype(np.float32) * noise
    burst = int(0.012 * SR)
    decay = np.exp(-np.linspace(0, 5, burst)).astype(np.float32)

    def put(t, amp):
        s = int(round(t * SR))
        e = min(n, s + burst)
        if s < n:
            x[s:e] += amp * rng.standard_normal(e - s).astype(np.float32) * decay[: e - s]

    for t in times:
        put(t, gain)
    for t, d in (echoes or {}).items():
        put(t + d, gain * 0.7)
    return np.clip(x, -1, 1)


def save(path, samples):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((samples * 32767).astype("<i2").tobytes())


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ref_t = [0.6, 1.1, 1.6, 2.1, 2.6]
    save(out / "ref.wav", synth(ref_t, 3.2))
    save(out / "test_ok.wav", synth([1.3, 1.8, 2.3, 2.8, 3.3], 4.0))  # 起始静音
    save(out / "test_miss.wav", synth([0.6, 1.1, 2.1, 2.6], 3.2))      # 漏第 3 冲
    save(
        out / "test_extra.wav",
        synth([0.6, 1.1, 1.35, 1.6, 2.1, 2.6], 3.2, echoes={0.6: 0.035}),
    )
    save(out / "test_ambiguous.wav", synth([0.6, 1.1, 1.6, 2.1, 2.6, 3.1], 3.7))
    print("written to", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
