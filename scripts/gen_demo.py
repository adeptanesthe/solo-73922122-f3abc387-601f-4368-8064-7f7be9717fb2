"""生成演示用 WAV（16kHz/16bit/单声道），供手工冒烟测试。

用法：python -I scripts/gen_demo.py <输出目录>
生成参考录音、四种待检录音（正常/漏冲/多冲/歧义），
以及维修前后两份带衰减尾音的录音（after 尾音更长）。
"""
from __future__ import annotations

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


def decaying_tail(tau, impact_t=1.0, duration=3.0, amp=0.6, noise=0.002, seed=7):
    """冲击 + 指数衰减尾音（若干带内正弦叠加），用于衰减对比演示。"""
    rng = np.random.default_rng(seed)
    n = int(duration * SR)
    t = np.arange(n) / SR
    x = rng.standard_normal(n).astype(np.float32) * noise
    tone = np.zeros(n, dtype=np.float32)
    for freq in (700, 1100, 1700, 2500, 3300):
        tone += np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi)).astype(np.float32)
    tone /= 5
    ramp = np.clip((t - impact_t) / 0.003, 0, 1)
    env = np.where(t >= impact_t, np.exp(-(t - impact_t) / tau), 0) * ramp
    x += amp * tone * env
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
    # 维修前后衰减对比：冲击在 1.0s，冲击+尾音时段 0.98~2.6s，背景段 0.1~0.7s
    save(out / "tail_before.wav", decaying_tail(0.045, seed=11))
    save(out / "tail_after.wav", decaying_tail(0.090, seed=12))
    print("written to", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
