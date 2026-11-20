"""合成测试 WAV 与 TestClient 夹具。"""
from __future__ import annotations

import io
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.service import Store

SR = 16_000


def build_wav(
    impact_times: list[float],
    duration: float | None = None,
    sr: int = SR,
    gain: float = 0.8,
    burst_ms: int = 12,
    echoes: dict[float, float] | None = None,
    echo_gain: float = 0.7,
    noise_floor: float = 0.0,
    sample_width: int = 2,
    channels: int = 1,
    seed: int = 7,
) -> bytes:
    """生成带若干冲击噪声脉冲的 PCM WAV。

    echoes: {主冲击时刻: 回声延迟秒}，回声幅度约为主冲击的 40%。
    """
    rng = np.random.default_rng(seed)
    if duration is None:
        duration = max(impact_times) + 0.4 if impact_times else 1.0
    n = int(round(duration * sr))
    x = rng.standard_normal(n).astype(np.float32) * noise_floor
    burst_n = int(burst_ms / 1000 * sr)
    decay = np.exp(-np.linspace(0, 5, burst_n)).astype(np.float32)

    def put(t: float, amp: float) -> None:
        start = int(round(t * sr))
        end = min(n, start + burst_n)
        if start >= n or end <= 0:
            return
        seg = rng.standard_normal(end - start).astype(np.float32)
        x[start:end] += amp * seg * decay[: end - start]

    for t in impact_times:
        put(t, gain)
    for t, delay in (echoes or {}).items():
        put(t + delay, gain * echo_gain)

    x = np.clip(x, -1.0, 1.0)

    if sample_width == 2:
        pcm = (x * 32767).astype("<i2").tobytes()
        sampwidth = 2
    else:  # 8-bit unsigned PCM
        pcm = ((x + 1.0) * 127.5).clip(0, 255).astype(np.uint8).tobytes()
        sampwidth = 1
    if channels == 2:
        pcm = b"".join(bytes([b, b]) for b in pcm) if sampwidth == 1 else b""
        if sampwidth == 2:
            stereo = np.repeat((x * 32767).astype("<i2"), 2)
            pcm = stereo.tobytes()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sr)
        wf.writeframes(pcm)
    return buf.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("app.main.store", Store(tmp_path))
    return TestClient(app)


def post_inspect(client: TestClient, ref: bytes, test: bytes):
    return client.post(
        "/api/inspect",
        files={
            "reference": ("ref.wav", ref, "audio/wav"),
            "test": ("test.wav", test, "audio/wav"),
        },
    )
