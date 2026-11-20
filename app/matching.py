"""参考冲次与待检冲次的节奏匹配。

用动态规划在「参考冲次序列 R」和「待检冲次序列 T」之间找最优对齐：
- 匹配：参考冲次 i 与待检冲次 j 对应，相邻两次匹配之间的总时长相对
  参考节奏允许 ±15% 偏差（累计跨度判定，单个间隔的偏差不会被漏冲掩盖）；
- 漏冲：参考冲次没有对应的待检冲次，计 1 次漏冲；
- 多冲：待检冲次无法归属到任何参考冲次，计 1 次多冲。

关键性质：
1. 漏冲只增加漏冲计数，不会把后续冲次整体错位（对齐始终以参考序号为准）；
2. 起始静音通过「首对匹配前的待检冲次记多冲、参考冲次记漏冲」自然处理；
3. 40ms 回声已在检测阶段抑制，不进入本模块；
4. 若存在多条成本相同但归属不同的最优对齐，置 ambiguous=True，
   前端提示人工复听，而不是武断放行。
"""
from __future__ import annotations

from dataclasses import dataclass, field

TIMING_TOLERANCE = 0.15
MISS_COST = 1
EXTRA_COST = 1
MAX_WAY_COUNT = 2  # 只需区分「唯一最优」还是「多个等价最优」


@dataclass
class MatchResult:
    matched: list[tuple[int, int]] = field(default_factory=list)
    missed_ref_indices: list[int] = field(default_factory=list)
    extra_test_indices: list[int] = field(default_factory=list)
    miss_count: int = 0
    extra_count: int = 0
    ambiguous: bool = False
    optimal_cost: int = 0
    optimal_way_count: int = 1
    timing_ratios: list[float] = field(default_factory=list)
    expected_missed_times: dict[int, float] = field(default_factory=dict)


def _timing_ok(t_j: float, t_prev: float, r_i: float, r_prev: float) -> bool:
    """相邻两次匹配之间的累计时长是否落在 ±15% 节奏窗口内。"""
    ref_span = r_i - r_prev
    test_span = t_j - t_prev
    if ref_span <= 0:
        return False
    if test_span < 0:
        return False
    ratio = test_span / ref_span
    return abs(ratio - 1.0) <= TIMING_TOLERANCE + 1e-12


def align_impacts(ref_times: list[float], test_times: list[float]) -> MatchResult:
    n = len(ref_times)
    m = len(test_times)
    result = MatchResult()

    if n == 0:
        # 参考为空（正常流程下会在 API 层拦截，这里做防御处理）
        result.extra_test_indices = list(range(m))
        result.extra_count = m
        result.optimal_cost = m * EXTRA_COST
        result.ambiguous = m > 0
        return result

    # cost[i][j]：参考前 i+1 个、待检前 j+1 个冲次，且 (i,j) 构成一对匹配
    # ways[i][j]：达到该最小成本的不同对齐路径数（封顶 MAX_WAY_COUNT）
    cost: list[list[int]] = [[0] * m for _ in range(n)]
    ways: list[list[int]] = [[0] * m for _ in range(n)]
    # back[i][j]：最优来源，元素为 (pi, pj)；锚点为 None；多来源记 "multi"
    back: list[list] = [[None] * m for _ in range(n)]

    for i in range(n):
        for j in range(m):
            anchor_cost = i * MISS_COST + j * EXTRA_COST  # 首对匹配
            best = anchor_cost
            best_ways = 1
            best_src = None
            for pi in range(i):
                for pj in range(j):
                    if ways[pi][pj] == 0:
                        continue
                    if not _timing_ok(
                        test_times[j], test_times[pj],
                        ref_times[i], ref_times[pi],
                    ):
                        continue
                    between_miss = (i - pi - 1) * MISS_COST
                    between_extra = (j - pj - 1) * EXTRA_COST
                    c = cost[pi][pj] + between_miss + between_extra
                    if c < best:
                        best = c
                        best_ways = ways[pi][pj]
                        best_src = (pi, pj)
                    elif c == best:
                        best_ways = min(
                            MAX_WAY_COUNT, best_ways + ways[pi][pj]
                        )
                        if best_src != (pi, pj):
                            best_src = "multi"
            cost[i][j] = best
            ways[i][j] = best_ways
            back[i][j] = best_src

    # 终止：匹配到 (i,j) 后，剩余参考全漏冲、剩余待检全多冲
    terminal_costs: list[tuple[int, int, int]] = []
    for i in range(n):
        for j in range(m):
            if ways[i][j] == 0:
                continue
            tail = (n - 1 - i) * MISS_COST + (m - 1 - j) * EXTRA_COST
            terminal_costs.append((cost[i][j] + tail, i, j))

    if not terminal_costs:
        # m == 0：待检一个冲次都没检测到，全部漏冲
        result.missed_ref_indices = list(range(n))
        result.miss_count = n
        result.optimal_cost = n * MISS_COST
        result.optimal_way_count = 1
        for i, r in enumerate(ref_times):
            result.expected_missed_times[i] = r
        return result

    global_best = min(tc[0] for tc in terminal_costs)
    best_terminals = [(i, j) for c, i, j in terminal_costs if c == global_best]
    total_ways = 0
    for i, j in best_terminals:
        total_ways = min(MAX_WAY_COUNT, total_ways + ways[i][j])
    # 不同最优终止点本身就是不同的冲次归属，同样构成歧义
    if len(best_terminals) > 1:
        total_ways = MAX_WAY_COUNT

    # 回溯其中一条最优路径用于展示
    end_i, end_j = best_terminals[0]
    matched: list[tuple[int, int]] = []
    i, j = end_i, end_j
    while True:
        matched.append((i, j))
        src = back[i][j]
        if src is None:
            break
        if src == "multi":  # 找到任一具体前驱即可
            found = None
            for pi in range(i):
                for pj in range(j):
                    if ways[pi][pj] == 0:
                        continue
                    if not _timing_ok(
                        test_times[j], test_times[pj],
                        ref_times[i], ref_times[pi],
                    ):
                        continue
                    between = (i - pi - 1) + (j - pj - 1)
                    if cost[pi][pj] + between * MISS_COST == cost[i][j]:
                        found = (pi, pj)
                        break
                if found:
                    break
            if found is None:  # pragma: no cover - 防御性分支
                break
            i, j = found
        else:
            i, j = src
    matched.reverse()

    matched_ref = {i for i, _ in matched}
    matched_test = {j for _, j in matched}
    missed = [i for i in range(n) if i not in matched_ref]
    extras = [j for j in range(m) if j not in matched_test]

    result.matched = matched
    result.missed_ref_indices = missed
    result.extra_test_indices = extras
    result.miss_count = len(missed)
    result.extra_count = len(extras)
    result.optimal_cost = global_best
    result.optimal_way_count = total_ways
    result.ambiguous = total_ways > 1 or len(best_terminals) > 1

    ratios: list[float] = []
    for (i1, j1), (i2, j2) in zip(matched, matched[1:]):
        ref_span = ref_times[i2] - ref_times[i1]
        if ref_span > 0:
            ratios.append((test_times[j2] - test_times[j1]) / ref_span)
    result.timing_ratios = ratios

    result.expected_missed_times = _estimate_missed_times(
        ref_times, test_times, matched, missed
    )
    return result


def _estimate_missed_times(
    ref_times: list[float],
    test_times: list[float],
    matched: list[tuple[int, int]],
    missed: list[int],
) -> dict[int, float]:
    """按匹配段估计每个漏冲参考冲次「本应出现」的时刻。"""
    expected: dict[int, float] = {}
    if not matched:
        for i in missed:
            expected[i] = ref_times[i]
        return expected

    ratios = []
    for (i1, j1), (i2, j2) in zip(matched, matched[1:]):
        ref_span = ref_times[i2] - ref_times[i1]
        if ref_span > 0:
            ratios.append((test_times[j2] - test_times[j1]) / ref_span)
    scale = sorted(ratios)[len(ratios) // 2] if ratios else 1.0

    first_i, first_j = matched[0]
    r0, t0 = ref_times[first_i], test_times[first_j]
    for i in missed:
        expected[i] = max(0.0, t0 + (ref_times[i] - r0) * scale)
    return expected
