"""Compare cash repair with the original heuristic and exhaustive enumeration."""

import argparse
from collections import Counter
from itertools import product
from statistics import mean
from time import perf_counter

import numpy as np

from portfolio import Portfolio


def run():
    fees = (0, 0.0001, 0.0005, 0.002, 0.005)
    tolerances = (0.0001, 0.0005, 0.001)
    rows = []
    for fee, cash_floor, volatility in product(fees, (0, 0.25), (0.08, 0.12, 0.16)):
        model = Portfolio(
            [0.20, 0.20, 0.15, 0.45], [1.12, 1.07, 1.02, 1.0],
            np.diag([0.04, 0.0225, 0.01, 0]),
            [-0.20, -0.20, -0.15, -0.45], [0.80, 0.80, 0.85, 0.55],
            buy_rate=[0.001, 0.001, 0.001, 0],
            sell_rate=[0.001, 0.001, 0.001, 0],
            buy_fee=[fee, fee, fee, 0], sell_fee=[fee, fee, fee, 0],
            min_position=[0, 0, 0, cash_floor], volatility=volatility,
        )
        baseline = model.heuristic()
        exact = model.exact()
        runs = [model.solve_with_cash_correction(3, epsilon) for epsilon in tolerances]
        for epsilon, result in zip(tolerances, runs):
            assert result.upper_bound >= exact.wealth - 1e-8
            if result.corrected is not None:
                assert result.corrected.budget_residual <= 1e-8
                assert result.corrected.wealth <= exact.wealth + 1e-8
            if result.stopped_early:
                assert exact.wealth - result.candidate.wealth <= epsilon + 1e-8
                assert result.solves == 1
            else:
                assert result.solves == baseline.solves
            assert result.candidate.budget_residual <= 1e-8
        rows.append((fee, cash_floor, volatility, baseline, exact, runs))
    return rows


def print_small():
    rows = run()
    print(f"Instances: {len(rows)}; baseline mean solves: {mean(row[3].solves for row in rows):.2f}")
    print(f"Baseline max gap (bp): {1e4 * max(row[4].wealth - row[3].candidate.wealth for row in rows):.4f}")
    print(f"Cash repairs feasible: {sum(row[5][0].corrected is not None for row in rows)}")
    for i, epsilon_bp in enumerate((1, 5, 10)):
        results = [row[5][i] for row in rows]
        early = [(row[4], row[5][i]) for row in rows if row[5][i].stopped_early]
        early_gap = max((exact.wealth - result.candidate.wealth for exact, result in early), default=0)
        max_gap = max(row[4].wealth - row[5][i].candidate.wealth for row in rows)
        by_fee = Counter(row[0] for row in rows if row[5][i].stopped_early)
        print(f"{epsilon_bp:2} bp: early={len(early):2}, mean solves={mean(r.solves for r in results):.2f}, "
              f"early max gap={early_gap * 1e4:.4f} bp, overall max gap={max_gap * 1e4:.4f} bp, "
              f"early by fee={dict(by_fee)}")
    for fee, floor, volatility, baseline, exact, results in rows:
        result = results[1]
        if not result.stopped_early and result.corrected is None:
            print(f"Rejected repair example: fee={fee}, cash floor={floor}, volatility={volatility}, "
                  f"deficit={result.deficit * 1e4:.4f} bp")
            break


def print_large():
    rows = []
    epsilon = 0.0005
    for n, seed, fee_bp, cash_floor in product(
        (20, 50, 100), (12, 49), (0.2, 1, 5), (0, 0.25),
    ):
        rng = np.random.default_rng(seed)
        initial = np.full(n, 0.75 / n)
        loadings = rng.uniform(0.75, 1.25, n)
        risky_covariance = 0.04 * np.outer(loadings, loadings)
        risky_covariance += np.diag(rng.uniform(0.02, 0.07, n))
        covariance = np.zeros((n + 1, n + 1))
        covariance[:n, :n] = risky_covariance
        model = Portfolio(
            np.r_[initial, 0.25], np.r_[rng.uniform(1.01, 1.13, n), 1], covariance,
            np.r_[-initial, cash_floor - 0.25], np.r_[0.15 - initial, 0.75],
            buy_rate=np.r_[np.full(n, 0.001), 0], sell_rate=np.r_[np.full(n, 0.001), 0],
            buy_fee=np.r_[np.full(n, fee_bp / 1e4), 0],
            sell_fee=np.r_[np.full(n, fee_bp / 1e4), 0],
            min_position=np.r_[np.zeros(n), cash_floor], max_position=np.r_[np.full(n, 0.15), 1],
            volatility=1.05 * np.sqrt(initial @ risky_covariance @ initial),
        )
        start = perf_counter()
        baseline = model.heuristic()
        baseline_time = perf_counter() - start
        start = perf_counter()
        result = model.solve_with_cash_correction(n, epsilon)
        result_time = perf_counter() - start
        assert result.candidate is not None and result.candidate.budget_residual <= 1e-8
        if baseline.candidate is not None:
            assert result.upper_bound >= baseline.candidate.wealth - 1e-8
            if result.stopped_early:
                assert baseline.candidate.wealth - result.candidate.wealth <= epsilon + 1e-8
        rows.append((n, fee_bp, cash_floor, baseline, result, baseline_time, result_time))

    for label, index in (("risky assets", 0), ("fee per order (bp)", 1), ("cash floor", 2)):
        for value in sorted({row[index] for row in rows}):
            group = [row for row in rows if row[index] == value]
            baseline_solves = mean(row[3].solves for row in group)
            corrected_solves = mean(row[4].solves for row in group)
            feasible = sum(row[4].corrected is not None for row in group)
            early = sum(row[4].stopped_early for row in group)
            print(f"{label}={value}: feasible={feasible}/{len(group)}, early={early}/{len(group)}, "
                  f"mean solves={baseline_solves:.2f}->{corrected_solves:.2f}")
    print(f"Total time (s), baseline / corrected: {sum(r[5] for r in rows):.2f} / "
          f"{sum(r[6] for r in rows):.2f}")
    print(f"Baseline candidates missing: {sum(r[3].candidate is None for r in rows)}")
    early = [r for r in rows if r[4].stopped_early]
    if early:
        print(f"Largest early gap bound: {1e4 * max(r[4].gap_bound for r in early):.3f} bp")
    comparable = [r for r in rows if r[3].candidate is not None]
    if comparable:
        gains = [1e4 * (r[4].candidate.wealth - r[3].candidate.wealth) for r in comparable]
        print(f"Wealth difference vs baseline (bp): min={min(gains):.3f}, max={max(gains):.3f}")


def print_screening():
    for label, fee, cash, asymmetric in (
        ("low fees", .0001, .3, False),
        ("higher fees", .005, .3, False),
        ("asymmetric fees", .003, .1, True),
    ):
        n = 8
        initial = np.full(n, (1 - cash) / n)
        sell_fees = np.full(n, fee)
        if asymmetric:
            sell_fees[0] *= 2
        model = Portfolio(
            np.r_[initial, cash], [1.18, 1.12, 1.07, 1.04, 1.01, 1, .99, .98, 1],
            np.diag(np.r_[np.linspace(.03, .09, n), 0]),
            -np.r_[initial, cash], np.r_[.25 - initial, 1 - cash],
            buy_rate=np.r_[np.full(n, .001), 0], sell_rate=np.r_[np.full(n, .001), 0],
            buy_fee=np.r_[np.full(n, fee), 0], sell_fee=np.r_[sell_fees, 0],
            min_position=0, volatility=.1,
        )
        start = perf_counter()
        exact = model.exact()
        baseline_time = perf_counter() - start
        start = perf_counter()
        result = model.screened_exact()
        screened_time = perf_counter() - start
        assert model._is_feasible(result.candidate, 1e-8)
        assert abs(exact.wealth - result.candidate.wealth) <= 1e-8
        print(f"{label}: frozen={result.frozen}, patterns={result.patterns_before}->{result.patterns_after}, "
              f"setup={result.setup_solves}, optimum={exact.wealth:.8f}, "
              f"seconds={baseline_time:.2f}->{screened_time:.2f}", flush=True)


def screening_model(n, fee_bp, sell_multiplier=1, asymmetric_rates=False):
    cash = .25
    initial = np.full(n, (1 - cash) / n)
    covariance = np.zeros((n + 1, n + 1))
    covariance[:n, :n] = .008 + np.diag(np.linspace(.03, .09, n))
    buy_fee = np.full(n, fee_bp / 1e4)
    sell_fee = buy_fee.copy()
    sell_fee[:2] *= sell_multiplier
    buy_rate = np.full(n, .001)
    sell_rate = np.linspace(.0005, .003, n) if asymmetric_rates else buy_rate
    return Portfolio(
        np.r_[initial, cash], np.r_[1 + .18 * np.exp(-np.arange(n) / 2.5) - .002 * np.arange(n), 1],
        covariance, -np.r_[initial, cash], np.r_[.22 - initial, 1 - cash],
        buy_rate=np.r_[buy_rate, 0], sell_rate=np.r_[sell_rate, 0],
        buy_fee=np.r_[buy_fee, 0], sell_fee=np.r_[sell_fee, 0],
        min_position=0, volatility=.1,
    )


def print_screening_timing():
    profiles = (
        (10, "proportional only", 0, 1, False),
        (10, "low symmetric fixed", 1, 1, False),
        (10, "high symmetric fixed", 50, 1, False),
        (10, "asymmetric fixed and proportional", 30, 2, True),
        (12, "high symmetric fixed", 50, 1, False),
    )
    for n, label, fee_bp, sell_multiplier, asymmetric_rates in profiles:
        model = screening_model(n, fee_bp, sell_multiplier, asymmetric_rates)
        model.relaxation()  # Warm up the solver before timing either method.
        start = perf_counter()
        baseline = model.exact()
        baseline_time = perf_counter() - start
        start = perf_counter()
        result = model.screened_exact()
        filtered_time = perf_counter() - start
        gap = abs(result.candidate.wealth - baseline.wealth)
        assert model._is_feasible(result.candidate, 1e-8)
        assert gap <= 1e-8
        print(f"n={n}, {label}: frozen={len(result.frozen)}, "
              f"solves={result.patterns_before}->{result.patterns_after}+{result.setup_solves}, "
              f"time={baseline_time:.3f}->{filtered_time:.3f}s, "
              f"ratio={baseline_time / filtered_time:.2f}x, wealth_difference={gap:.2e}", flush=True)


def print_screening_scale():
    profiles = [(n, 50, 1) for n in (10, 12)]
    profiles += [(n, fee, multiplier) for n in (50, 80) for fee, multiplier in ((1, 1), (50, 1), (30, 2))]
    for n, fee, multiplier in profiles:
        model = screening_model(n, fee, multiplier, asymmetric_rates=multiplier != 1)
        start = perf_counter()
        screen = model.screen_trades()
        elapsed = perf_counter() - start
        assert screen.incumbent is not None
        summaries = []
        before = model._pattern_count(model._trade_choices())
        for bounds in (screen.affine_bounds, screen.tighter_bounds):
            buy, sell = model.allowed_directions(screen, bounds)
            remaining = model._pattern_count(model._trade_choices(buy, sell))
            closest = 1e4 * (np.max(bounds[:, :n], axis=0).min() - screen.incumbent.wealth)
            frozen = np.count_nonzero(~buy[:n] & ~sell[:n])
            pruned = 2 * n - np.count_nonzero(buy[:n]) - np.count_nonzero(sell[:n])
            summaries.append((frozen, pruned, remaining, round(closest, 3)))
        gap = 1e4 * (screen.relaxed_wealth - screen.incumbent.wealth)
        print(f"n={n}, fee={fee}, multiplier={multiplier}: gap={gap:.3f}bp, "
              f"before={before}, (frozen, pruned, remaining, closest bp)={summaries}, "
              f"setup={screen.setup_solves}, {elapsed:.3f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--large", action="store_true", help="run the larger synthetic comparison")
    modes.add_argument("--screening", action="store_true", help="compare screened and full enumeration")
    modes.add_argument("--screening-timing", action="store_true", help="time screening across fee types")
    modes.add_argument("--screening-scale", action="store_true", help="compare bounds without enumeration")
    args = parser.parse_args()
    if args.large:
        print_large()
    elif args.screening:
        print_screening()
    elif args.screening_timing:
        print_screening_timing()
    elif args.screening_scale:
        print_screening_scale()
    else:
        print_small()
