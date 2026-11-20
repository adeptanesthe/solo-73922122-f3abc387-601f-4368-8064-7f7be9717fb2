"""维修前后尾音衰减对比：时序规则与 API 流程测试。"""
from __future__ import annotations

import numpy as np
import pytest

from app.decay import (
    DecayValidationError,
    TimeWindow,
    _first_hold_crossing,
    compare_tails,
    measure_tail,
)
from app.audio import parse_wav
from conftest import build_wav

SR = 16_000

# 冲击在 1.0 s；冲击+尾音时段与背景段互不重叠
IMPACT_WIN = TimeWindow(0.98, 2.6)
BG_WIN = TimeWindow(0.1, 0.7)
BAND_LOW, BAND_HIGH = 500, 4000


def decaying_tail(
    tau: float,
    impact_t: float = 1.0,
    duration: float = 3.0,
    amp: float = 0.6,
    bg: float = 0.0005,
    seed: int = 1,
) -> np.ndarray:
    """生成带指数衰减尾音的 16 kHz 波形。

    尾音由若干待测频带内的正弦叠加而成，包络为 exp(-(t-impact)/tau)，
    起点有 3 ms 上升沿，保证峰后可稳定测到 −25 dB 以下。
    """
    rng = np.random.default_rng(seed)
    n = int(duration * SR)
    t = np.arange(n) / SR
    x = rng.standard_normal(n).astype(np.float32) * bg
    tone = np.zeros(n, dtype=np.float32)
    for freq in (700, 1100, 1700, 2500, 3300):
        phase = rng.uniform(0, 2 * np.pi)
        tone += np.sin(2 * np.pi * freq * t + phase).astype(np.float32)
    tone /= 5.0
    ramp = np.clip((t - impact_t) / 0.003, 0.0, 1.0)
    env = np.where(t >= impact_t, np.exp(-(t - impact_t) / tau), 0.0) * ramp
    x += amp * tone * env
    return np.clip(x, -1.0, 1.0)


def _wav_bytes(samples: np.ndarray) -> bytes:
    pcm = (samples * 32767).astype("<i2").tobytes()
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm)
    return buf.getvalue()


def post_compare(client, before: bytes, after: bytes, **overrides):
    data = {
        "band_low_hz": str(BAND_LOW),
        "band_high_hz": str(BAND_HIGH),
        "before_impact_start": str(IMPACT_WIN.start),
        "before_impact_end": str(IMPACT_WIN.end),
        "before_background_start": str(BG_WIN.start),
        "before_background_end": str(BG_WIN.end),
        "after_impact_start": str(IMPACT_WIN.start),
        "after_impact_end": str(IMPACT_WIN.end),
        "after_background_start": str(BG_WIN.start),
        "after_background_end": str(BG_WIN.end),
    }
    data.update({k: str(v) for k, v in overrides.items()})
    return client.post(
        "/api/compare",
        data=data,
        files={
            "before": ("before.wav", before, "audio/wav"),
            "after": ("after.wav", after, "audio/wav"),
        },
    )


def test_slower_tail_is_identified_and_common_level_is_deepest_shared():
    before = decaying_tail(0.045, seed=1)
    after = decaying_tail(0.090, seed=2)
    b, a, level = compare_tails(
        before, after, BAND_LOW, BAND_HIGH, IMPACT_WIN, BG_WIN, IMPACT_WIN, BG_WIN
    )
    assert -25 <= level <= -15
    # 用合成尾音时双方都能测到 −25 dB
    assert level == -25
    assert b.end_time(level) is not None and a.end_time(level) is not None
    # 终点必须晚于各自 −5 dB 起点
    for m in (b, a):
        assert m.start_time > m.peak_time
        assert m.end_time(level) > m.start_time
    # 衰减更慢的一份耗时更长
    assert a.end_time(level) - a.start_time > b.end_time(level) - b.start_time


def test_swapping_before_after_swaps_result():
    b, a, level = compare_tails(
        decaying_tail(0.09, seed=1),
        decaying_tail(0.045, seed=2),
        BAND_LOW,
        BAND_HIGH,
        IMPACT_WIN,
        BG_WIN,
        IMPACT_WIN,
        BG_WIN,
    )
    assert b.end_time(level) - b.start_time > a.end_time(level) - a.start_time


def test_hold_rule_uses_first_of_three_frames():
    times = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
    # 0.02 帧 −7 dB 后立刻反弹，只有两帧不高于 −5，不构成 3 帧
    assert _first_hold_crossing(times, np.array([0, -5, -7, -1, -8, -1.0]), -5) is None
    # 0.01、0.02、0.03 连续 3 帧，取第一帧 0.01
    assert (
        _first_hold_crossing(times, np.array([0, -5, -6, -7, -1, -8.0]), -5) == 0.01
    )
    # 恰好等于目标电平也算不高于
    assert _first_hold_crossing(times, np.array([-5, -5, -5, 0, 0, 0.0]), -5) == 0.0


def test_end_crossing_must_start_again_after_start():
    """起点前已低于目标的拖尾不算起点后的新过线。"""
    times = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
    # 0.00 仍高于 −5，0.01 起连续 −5/−6/−7 三帧 -> 起点 0.01
    levels = np.array([-2, -5, -6, -7, -1, -5, -6, -7.0])
    assert _first_hold_crossing(times, levels, -5) == 0.01
    # 整段都低于 −30 dB 的拖尾早于起点出现，不算新的终点
    early = np.array([-30, -31, -32, -5, -6, -7, 0.0, 0.0])
    assert _first_hold_crossing(times, early, -30, after_time=0.01) is None
    # 起点之后才出现新的 −30 dB 连续段 -> 取该段首帧
    levels2 = np.array([0, -5, -6, -7, -30, -31, -32, 0.0])
    assert _first_hold_crossing(times, levels2, -30, after_time=0.01) == 0.04


def test_end_must_be_later_than_start():
    times = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    levels = np.array([0, -6, -30, -30, -30, -6, -30])
    start = _first_hold_crossing(times, levels, -5)
    assert start == 0.01
    # −30 dB 的过线在起点之前/之内，不能算作终点
    assert _first_hold_crossing(times, levels, -30, after_time=start) is not None
    assert _first_hold_crossing(
        np.array([0.0, 0.01, 0.02]), np.array([0.0, -30.0, -30.0]), -30,
        after_time=0.05,
    ) is None


def test_background_containing_impact_is_rejected():
    # 冲击在 1.0s 与 2.0s；背景段落在第二个冲击上（与冲击时段不重叠）
    samples = parse_wav(
        build_wav([1.0, 2.0], duration=3.0, noise_floor=0.002)
    )
    with pytest.raises(DecayValidationError, match="背景时段疑似包含冲击"):
        measure_tail(
            samples,
            BAND_LOW,
            BAND_HIGH,
            TimeWindow(0.98, 1.5),
            TimeWindow(1.8, 2.2),
            "维修前录音",
        )


def test_overlapping_windows_rejected():
    samples = decaying_tail(0.045)
    with pytest.raises(DecayValidationError, match="重叠"):
        measure_tail(
            samples, BAND_LOW, BAND_HIGH, IMPACT_WIN, TimeWindow(1.0, 1.5), "维修前录音"
        )


def test_invalid_band_rejected():
    samples = decaying_tail(0.045)
    for low, high, msg in [
        (0, 4000, "待测频带无效"),
        (2000, 1000, "待测频带无效"),
        (500, 9000, "奈奎斯特"),
        (1000.2, 1000.4, "过窄"),
    ]:
        with pytest.raises(DecayValidationError, match=msg):
            measure_tail(samples, low, high, IMPACT_WIN, BG_WIN, "维修前录音")


def test_window_out_of_range_rejected():
    samples = decaying_tail(0.045, duration=3.0)
    with pytest.raises(DecayValidationError, match="超出录音时长"):
        measure_tail(
            samples,
            BAND_LOW,
            BAND_HIGH,
            TimeWindow(0.98, 3.5),
            BG_WIN,
            "维修前录音",
        )


def test_noisy_floor_forces_shallower_common_level():
    """底噪抬高时，6 dB 截止会让深电平不可测，共用终点须退到双方都能测到的电平。"""
    noisy_before = decaying_tail(0.045, bg=0.01, seed=1)
    noisy_after = decaying_tail(0.09, bg=0.01, seed=2)
    b, a, level = compare_tails(
        noisy_before,
        noisy_after,
        BAND_LOW,
        BAND_HIGH,
        IMPACT_WIN,
        BG_WIN,
        IMPACT_WIN,
        BG_WIN,
    )
    assert -25 < level <= -15
    assert b.gate_time is not None and a.gate_time is not None
    assert b.end_time(level) is not None and a.end_time(level) is not None
    assert -25 not in b.end_times and -25 not in a.end_times


def test_no_common_level_explains_reason():
    before = decaying_tail(0.045)
    after = decaying_tail(0.09, seed=2)
    with pytest.raises(DecayValidationError, match="没有共同可测的整 dB 终点"):
        compare_tails(
            before,
            after,
            BAND_LOW,
            BAND_HIGH,
            TimeWindow(0.98, 1.08),  # 窗太短，双方都到不了 −15 dB
            BG_WIN,
            TimeWindow(0.98, 1.08),
            BG_WIN,
        )


def test_flat_tail_without_minus5_crossing_rejected():
    # 尾音几乎不衰减（τ=5s），测量区末尾仍高于 −5 dB，无法取计时起点
    samples = decaying_tail(5.0)
    with pytest.raises(DecayValidationError, match="−5 dB 过线"):
        measure_tail(samples, BAND_LOW, BAND_HIGH, IMPACT_WIN, BG_WIN, "维修前录音")


def test_api_compare_success_and_last_retrieval(client):
    resp = post_compare(
        client, _wav_bytes(decaying_tail(0.045, seed=1)),
        _wav_bytes(decaying_tail(0.09, seed=2)),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["common_level_db"] == -25
    assert body["slower"] == "after"
    assert body["before"]["elapsed"] < body["after"]["elapsed"]
    assert body["before"]["end_time"] > body["before"]["start_time"]
    assert body["after"]["end_time"] > body["after"]["start_time"]

    last = client.get("/api/comparison/last").json()
    assert last["id"] == body["id"]
    # 回传录音可播放
    assert client.get(body["before"]["url"]).status_code == 200


def test_api_invalid_input_clears_previous_comparison(client):
    ok = post_compare(client, _wav_bytes(decaying_tail(0.045)), _wav_bytes(decaying_tail(0.09, seed=2)))
    assert ok.status_code == 200
    assert client.get("/api/comparison/last").status_code == 200

    # 频带非法 -> 400 且旧对比被撤下
    bad = post_compare(
        client,
        _wav_bytes(decaying_tail(0.045)),
        _wav_bytes(decaying_tail(0.09, seed=2)),
        band_low_hz=0,
    )
    assert bad.status_code == 400
    assert "待测频带" in bad.json()["detail"]
    assert client.get("/api/comparison/last").status_code == 404

    # 上一份核对结果不受对比失败影响
    assert client.get("/api/result/last").status_code == 404


def test_api_missing_field_clears_previous_comparison(client):
    ok = post_compare(
        client, _wav_bytes(decaying_tail(0.045)), _wav_bytes(decaying_tail(0.09, seed=2))
    )
    assert ok.status_code == 200
    assert client.get("/api/comparison/last").status_code == 200

    resp = client.post(
        "/api/compare",
        data={"band_low_hz": "500", "band_high_hz": "4000"},
        files={
            "before": ("b.wav", _wav_bytes(decaying_tail(0.045)), "audio/wav"),
            "after": ("a.wav", _wav_bytes(decaying_tail(0.09, seed=2)), "audio/wav"),
        },
    )
    assert resp.status_code == 400
    assert "参数不完整" in resp.json()["detail"]
    assert client.get("/api/comparison/last").status_code == 404


def test_api_compare_rejects_bad_wav(client):
    resp = post_compare(client, b"not a wav", _wav_bytes(decaying_tail(0.09)))
    assert resp.status_code == 400
    assert "WAV" in resp.json()["detail"]


def test_api_compare_rejects_non_wav_extension(client):
    resp = client.post(
        "/api/compare",
        data={
            "band_low_hz": "500",
            "band_high_hz": "4000",
            "before_impact_start": "0.98",
            "before_impact_end": "2.6",
            "before_background_start": "0.1",
            "before_background_end": "0.7",
            "after_impact_start": "0.98",
            "after_impact_end": "2.6",
            "after_background_start": "0.1",
            "after_background_end": "0.7",
        },
        files={
            "before": ("before.txt", b"hello", "text/plain"),
            "after": ("after.wav", _wav_bytes(decaying_tail(0.09)), "audio/wav"),
        },
    )
    assert resp.status_code == 400
    assert ".wav" in resp.json()["detail"]
