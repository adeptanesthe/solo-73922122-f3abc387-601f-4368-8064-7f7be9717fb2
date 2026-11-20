"""节奏匹配动态规划测试。"""
from app.matching import TIMING_TOLERANCE, align_impacts


def test_perfect_match():
    r = [0.5, 1.0, 1.5, 2.0]
    t = [1.2, 1.7, 2.2, 2.7]  # 起始静音平移，节奏完全一致
    res = align_impacts(r, t)
    assert res.miss_count == 0
    assert res.extra_count == 0
    assert res.matched == [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert res.ambiguous is False


def test_miss_does_not_shift_later_impacts():
    """一次漏冲后，后续冲次必须仍按参考序号对齐，不错位。"""
    r = [0.5, 1.0, 1.5, 2.0, 2.5]
    t = [0.5, 1.0, 2.0, 2.5]  # 第 3 冲（1.5s）漏冲
    res = align_impacts(r, t)
    assert res.miss_count == 1
    assert res.extra_count == 0
    assert res.missed_ref_indices == [2]
    # 后续冲次没有整体错位
    assert res.matched == [(0, 0), (1, 1), (3, 2), (4, 3)]


def test_double_miss():
    r = [0.5, 1.0, 1.5, 2.0, 2.5]
    t = [0.5, 1.0, 2.5]  # 连漏两冲
    res = align_impacts(r, t)
    assert res.missed_ref_indices == [2, 3]
    assert res.matched == [(0, 0), (1, 1), (4, 2)]


def test_extra_impact_reported_with_time():
    r = [0.5, 1.0, 1.5, 2.0]
    t = [0.5, 1.0, 1.25, 1.5, 2.0]  # 1.25s 多冲
    res = align_impacts(r, t)
    assert res.miss_count == 0
    assert res.extra_count == 1
    assert res.extra_test_indices == [2]


def test_leading_silence_extra_is_extra():
    r = [0.5, 1.0, 1.5, 2.0]
    t = [0.1, 0.5, 1.0, 1.5, 2.0]  # 开头多一个
    res = align_impacts(r, t)
    assert res.extra_test_indices == [0]
    assert res.matched[0] == (0, 1)


def test_tempo_change_within_15_percent_ok():
    r = [0.0, 0.5, 1.0, 1.5]
    scale = 1 + TIMING_TOLERANCE  # +15%
    t = [x * scale for x in r]
    res = align_impacts(r, t)
    assert res.miss_count == 0 and res.extra_count == 0

    t2 = [x * (1 - TIMING_TOLERANCE) for x in r]  # -15%
    res2 = align_impacts(r, t2)
    assert res2.miss_count == 0 and res2.extra_count == 0


def test_tempo_change_beyond_tolerance_forces_miss():
    r = [0.0, 0.5, 1.0, 1.5]
    t = [x * 1.30 for x in r]  # +30%，超出窗口
    res = align_impacts(r, t)
    # 无法整段匹配，最优解中必然出现漏冲/多冲
    assert res.miss_count + res.extra_count > 0


def test_ambiguous_equal_cost_alignments_flagged():
    """参考节奏均匀时，末尾多一个等间隔冲次可对应两种等价归属。"""
    r = [0.0, 0.5, 1.0, 1.5]
    t = [0.0, 0.5, 1.0, 1.5, 2.0]
    res = align_impacts(r, t)
    assert res.ambiguous is True
    assert res.optimal_way_count >= 2
    # 两种归属都是 1 次多冲，不应被当成正常
    assert res.extra_count + res.miss_count >= 1


def test_empty_test_all_missed():
    r = [0.5, 1.0, 1.5, 2.0]
    res = align_impacts(r, [])
    assert res.miss_count == 4
    assert res.matched == []
    assert res.ambiguous is False
    assert set(res.expected_missed_times) == {0, 1, 2, 3}


def test_expected_missed_time_estimation():
    r = [0.0, 0.5, 1.0, 1.5]
    t = [2.0, 2.5, 3.5]  # 参考第 3 冲漏，整体平移 2s、节奏不变
    res = align_impacts(r, t)
    assert res.missed_ref_indices == [2]
    # 漏冲预计时刻 ≈ 2.0 + 1.0 = 3.0
    assert res.expected_missed_times[2] == 3.0
