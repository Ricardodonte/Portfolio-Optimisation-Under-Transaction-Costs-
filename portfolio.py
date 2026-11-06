"""Single-period portfolio rebalancing with proportional and fixed trade fees."""

from dataclasses import dataclass
from enum import IntEnum
from itertools import product
from statistics import NormalDist

import cvxpy as cp
import numpy as np


class Trade(IntEnum):
    SELL = -1
    IDLE = 0
    BUY = 1
    EITHER = 2


@dataclass(frozen=True)
class Solution:
    trades: np.ndarray
    holdings: np.ndarray
    wealth: float
    fees: float
    budget_residual: float


@dataclass(frozen=True)
class HeuristicResult:
    relaxation: Solution
    bound: float
    candidate: Solution | None
    iterations: int
    converged: bool
    solves: int


@dataclass(frozen=True)
class CashCorrectionResult:
    candidate: Solution | None
    corrected: Solution | None
    upper_bound: float
    deficit: float
    gap_bound: float | None
    stopped_early: bool
    solves: int


@dataclass(frozen=True)
class ScreeningResult:
    candidate: Solution | None
    frozen: tuple[int, ...]
    patterns_before: int
    patterns_after: int
    setup_solves: int


@dataclass(frozen=True)
class TradeScreen:
    incumbent: Solution | None
    relaxed_wealth: float
    affine_bounds: np.ndarray
    tighter_bounds: np.ndarray
    setup_solves: int


class Portfolio:
    """Amounts are in currency units, and returns are gross holding-period factors.

    lower and upper bound the signed trades and must contain zero. The caller
    must choose bounds valid for the original problem to obtain global bounds.
    Fixed and proportional fees may differ by trade direction.
    Each shortfall is a (confidence, terminal wealth floor) pair under normal
    returns. Each concentration limit is a (number of largest positions,
    fraction of total holdings) pair.
    """

    def __init__(
        self, holdings, returns, covariance, lower, upper, *,
        buy_rate=0, sell_rate=0, buy_fee=0, sell_fee=0,
        min_position=None, max_position=None, max_total_short=None,
        concentration_limits=(), volatility=None, shortfalls=(),
    ):
        self.holdings = np.asarray(holdings, dtype=float)
        if self.holdings.ndim != 1 or not self.holdings.size:
            raise ValueError("holdings must be a nonempty vector")
        self.n = self.holdings.size
        self.returns = self._vector(returns)
        self.lower, self.upper = self._vector(lower), self._vector(upper)
        self.buy_rate, self.sell_rate = self._vector(buy_rate), self._vector(sell_rate)
        self.buy_fee, self.sell_fee = self._vector(buy_fee), self._vector(sell_fee)
        if not np.isfinite(self.holdings).all():
            raise ValueError("holdings must be finite")
        if np.any(self.lower > 0) or np.any(self.upper < 0):
            raise ValueError("trade bounds must contain zero")
        if min(map(np.min, (self.buy_rate, self.sell_rate, self.buy_fee, self.sell_fee))) < 0:
            raise ValueError("transaction fees must be nonnegative")
        self.min_position = None if min_position is None else self._vector(min_position)
        self.max_position = None if max_position is None else self._vector(max_position)
        self.max_total_short = max_total_short
        if max_total_short is not None and (
            not np.isfinite(max_total_short) or max_total_short < 0
        ):
            raise ValueError("max_total_short must be finite and nonnegative")
        self.concentration_limits = tuple(concentration_limits)
        for count, fraction in self.concentration_limits:
            if not isinstance(count, (int, np.integer)) or not 1 <= count < self.n:
                raise ValueError("concentration limits require 1 <= count < n")
            if not 0 <= fraction <= 1:
                raise ValueError("concentration fractions must be between zero and one")
        self.volatility = volatility
        if volatility is not None and (not np.isfinite(volatility) or volatility < 0):
            raise ValueError("volatility must be finite and nonnegative")
        self.shortfalls = tuple(shortfalls)
        for probability, floor in self.shortfalls:
            if not 0.5 <= probability < 1 or not np.isfinite(floor):
                raise ValueError("shortfalls require 0.5 <= probability < 1 and a finite floor")

        covariance = np.asarray(covariance, dtype=float)
        if (covariance.shape != (self.n, self.n) or not np.isfinite(covariance).all()
                or not np.allclose(covariance, covariance.T)):
            raise ValueError("covariance must be a finite symmetric n-by-n matrix")
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        scale = np.max(np.abs(eigenvalues))
        if eigenvalues[0] < -1e-10 * scale:
            raise ValueError("covariance must be positive semidefinite")
        self.risk_factor = np.sqrt(np.maximum(eigenvalues, 0))[:, None] * eigenvectors.T

    def _vector(self, value):
        vector = np.broadcast_to(np.asarray(value, dtype=float), (self.n,)).copy()
        if not np.isfinite(vector).all():
            raise ValueError("model inputs must be finite")
        return vector

    def _result(self, trades):
        holdings = self.holdings + trades
        fees = (
            self.buy_rate @ np.maximum(trades, 0)
            + self.sell_rate @ np.maximum(-trades, 0)
            + self.buy_fee @ (trades > 0)
            + self.sell_fee @ (trades < 0)
        )
        return Solution(trades, holdings, float(self.returns @ holdings),
                        float(fees), float(trades.sum() + fees))

    def _portfolio_constraints(self, trades):
        holdings = self.holdings + trades
        wealth = self.returns @ holdings
        risk = cp.norm(self.risk_factor @ holdings)
        constraints = [trades >= self.lower, trades <= self.upper]
        if self.min_position is not None:
            constraints.append(holdings >= self.min_position)
        if self.max_position is not None:
            constraints.append(holdings <= self.max_position)
        if self.max_total_short is not None:
            constraints.append(cp.sum(cp.pos(-holdings)) <= self.max_total_short)
        for count, fraction in self.concentration_limits:
            constraints.append(cp.sum_largest(holdings, count) <= fraction * cp.sum(holdings))
        risk_constraint = None
        if self.volatility is not None:
            risk_constraint = risk <= self.volatility
            constraints.append(risk_constraint)
        for probability, floor in self.shortfalls:
            constraints.append(NormalDist().inv_cdf(probability) * risk <= wealth - floor)
        return constraints, risk_constraint

    def _solve(self, buy_rate, sell_rate, target, pattern=None, *,
               bound_from_duals=False, screen=False):
        trades = cp.Variable(self.n)
        wealth = self.returns @ (self.holdings + trades)
        fees = buy_rate @ cp.pos(trades) + sell_rate @ cp.pos(-trades)
        constraints, risk_constraint = self._portfolio_constraints(trades)

        if pattern is not None:
            buys, sells = pattern == Trade.BUY, pattern == Trade.SELL
            idle, either = pattern == Trade.IDLE, pattern == Trade.EITHER
            if buys.any():
                constraints.append(trades[buys] >= 0)
            if sells.any():
                constraints.append(trades[sells] <= 0)
            if idle.any():
                constraints.append(trades[idle] == 0)
            fees += self.buy_fee[buys | either].sum() + self.sell_fee[sells].sum()

        if target is None:
            constraints.append(cp.sum(trades) + fees <= 0)
            objective = cp.Maximize(wealth)
        else:
            # The article's cost-minimization formulation replaces the budget
            # constraint with a minimum expected terminal wealth.
            if not np.isfinite(target):
                raise ValueError("target must be finite")
            constraints.append(wealth >= target)
            objective = cp.Minimize(fees)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL)
        if problem.status == cp.INFEASIBLE:
            return None, None
        if problem.status != cp.OPTIMAL:
            raise RuntimeError(f"Solver returned {problem.status}")
        values = np.asarray(trades.value).copy()
        if screen:
            others = [c for c in constraints[2:-1] if c is not risk_constraint]
            multiplier = max(0.0, float(constraints[-1].dual_value))
            affine = self._direction_bounds(trades, constraints[2:-1], multiplier)
            curved = self._direction_bounds(trades, others, multiplier, risk_constraint)
            return self._result(values), (affine, np.minimum(affine, curved))
        if bound_from_duals:
            bound = self._affine_upper_bound(trades, constraints[2:])
            return self._result(values), bound
        if pattern is not None:
            values[idle] = 0
            values[buys] = np.maximum(values[buys], 0)
            values[sells] = np.minimum(values[sells], 0)
        return self._result(values), float(problem.value)

    def _affine_upper_bound(self, trades, constraints):
        # Subtract nonnegative multiples of the convex residuals, using a
        # supporting affine function for each residual at the solver's point.
        # The resulting affine majorant can be maximized exactly over the box.
        value, slope = self._affine_terms(trades, constraints)
        corner = np.where(slope >= 0, self.upper, self.lower)
        return float(value + slope @ corner)

    def _affine_terms(self, trades, constraints):
        point = trades.value
        value = float(self.returns @ (self.holdings + point))
        slope = self.returns.copy()
        for constraint in constraints:
            residual = constraint.expr
            gradient = residual.grad[trades]
            if gradient is None or constraint.dual_value is None:
                continue
            weights = np.maximum(np.asarray(constraint.dual_value).reshape(-1), 0)
            value -= float(weights @ np.asarray(residual.value).reshape(-1))
            if np.isscalar(gradient):
                slope -= gradient * weights
            else:
                slope -= np.asarray(gradient @ weights).reshape(-1)
        return float(value - slope @ point), slope

    def _direction_bounds(self, trades, constraints, multiplier, risk_constraint=None):
        """Bound terminal wealth with a buy or a sell imposed on each asset."""
        intercept, slope = self._affine_terms(trades, constraints)
        slope -= multiplier  # The exact budget also charges the signed trade amount.
        point = trades.value
        curvature = np.zeros(self.n)
        if risk_constraint is not None:
            exposure = self.risk_factor @ (self.holdings + point)
            risk = np.linalg.norm(exposure)
            risk_multiplier = max(0.0, float(risk_constraint.dual_value))
            if risk > 0 and risk_multiplier > 0:
                weight = risk_multiplier / (2 * risk)
                risk_gradient = 2 * self.risk_factor.T @ exposure
                intercept -= weight * (risk**2 - self.volatility**2 - risk_gradient @ point)
                slope -= weight * risk_gradient
                curvature = weight * self._risk_diagonal_floor()
        buy_slope = slope - multiplier * self.buy_rate
        sell_slope = slope + multiplier * self.sell_rate
        buy_value = self._branch_max(buy_slope, curvature, point, 0, self.upper)
        sell_value = self._branch_max(sell_slope, curvature, point, self.lower, 0)
        buy_value -= multiplier * self.buy_fee
        sell_value -= multiplier * self.sell_fee
        buy_value[self.upper == 0] = -np.inf
        sell_value[self.lower == 0] = -np.inf

        # A forced trade changes only one asset's contribution to the bound.
        idle_value = -curvature * point**2
        best_value = np.maximum(idle_value, np.maximum(buy_value, sell_value))
        unrestricted_assets = intercept + best_value.sum() - best_value
        return np.array([unrestricted_assets + buy_value, unrestricted_assets + sell_value])

    def _risk_diagonal_floor(self):
        # Zero-variance assets have zero covariance with every asset.
        risky = np.any(self.risk_factor != 0, axis=0)
        factor = self.risk_factor[:, risky]
        floor = np.zeros(self.n)
        if factor.size:
            eigenvalues = np.linalg.eigvalsh(factor.T @ factor)
            floor[risky] = max(0.0, eigenvalues[0] - 1e-12 * eigenvalues[-1])
        return floor

    @staticmethod
    def _branch_max(slope, curvature, point, lower, upper):
        """Maximize a concave quadratic (or a line) over the trade interval."""
        peak = np.divide(slope, 2 * curvature, out=np.zeros_like(slope), where=curvature > 0)
        peak = np.where(curvature > 0, point + peak, np.where(slope >= 0, upper, lower))
        peak = np.clip(peak, lower, upper)
        return slope * peak - curvature * (peak - point)**2

    def relaxation(self, *, target=None):
        """Return the relaxed trade and its objective bound.

        The bound is upper for wealth and lower for cost when target is given.
        The trade's fees and budget residual are evaluated at the true fees.
        """
        return self._relaxation(target)

    def _relaxation(self, target, *, bound_from_duals=False, screen=False):
        buy_slope = np.divide(self.buy_fee, self.upper, out=np.zeros(self.n), where=self.upper > 0)
        sell_slope = np.divide(self.sell_fee, -self.lower, out=np.zeros(self.n), where=self.lower < 0)
        return self._solve(self.buy_rate + buy_slope, self.sell_rate + sell_slope, target,
                           bound_from_duals=bound_from_duals, screen=screen)

    def heuristic(self, *, target=None, delta=1e-3, tolerance=1e-7, max_iterations=50):
        """Reweight the fixed fees, then reoptimize with the active set fixed.

        Trades of magnitude at most delta are deactivated for fixed-fee assets.
        The candidate is None if the selected pattern is infeasible.
        """
        relaxed, bound = self.relaxation(target=target)
        return self._iterate(relaxed, bound, target, delta, tolerance, max_iterations)

    def _iterate(self, relaxed, bound, target, delta, tolerance, max_iterations):
        if not np.isfinite([delta, tolerance]).all() or min(delta, tolerance) <= 0 or max_iterations < 1:
            raise ValueError("delta, tolerance and max_iterations must be positive")
        if relaxed is None:
            raise ValueError("the convex relaxation is infeasible")
        current = relaxed
        converged = False
        for iteration in range(1, max_iterations + 1):
            previous = current.trades
            divisor = np.abs(previous) + delta
            updated, _ = self._solve(
                self.buy_rate + self.buy_fee / divisor,
                self.sell_rate + self.sell_fee / divisor,
                target,
            )
            if updated is None:
                break
            current = updated
            if np.max(np.abs(current.trades - previous)) <= tolerance:
                converged = True
                break
        pattern = np.sign(current.trades)
        pattern[np.abs(current.trades) <= delta] = Trade.IDLE
        symmetric = self.buy_fee == self.sell_fee
        pattern[symmetric & ((pattern != Trade.IDLE) | (self.buy_fee == 0))] = Trade.EITHER
        candidate, _ = self._solve(self.buy_rate, self.sell_rate, target, pattern)
        return HeuristicResult(relaxed, bound, candidate, iteration, converged, iteration + 2)

    def solve_with_cash_correction(
        self, cash_index, epsilon, *, feasibility_tolerance=1e-8,
        delta=1e-3, tolerance=1e-7, max_iterations=50,
    ):
        """Maximize wealth with optional early stopping after cash repair.

        Feasibility and the affine upper bound are evaluated in floating point.
        Epsilon is an absolute tolerance in terminal wealth units.
        """
        if not isinstance(cash_index, (int, np.integer)) or not 0 <= cash_index < self.n:
            raise ValueError("cash_index must identify an asset")
        if (self.returns[cash_index] <= 0 or np.any(self.risk_factor[:, cash_index] != 0)
                or any(rates[cash_index] != 0 for rates in (
                    self.buy_rate, self.sell_rate, self.buy_fee, self.sell_fee,
                ))):
            raise ValueError("cash must have a positive certain return and zero trading fees")
        if (not np.isfinite([epsilon, feasibility_tolerance]).all()
                or min(epsilon, feasibility_tolerance) < 0):
            raise ValueError("epsilon and feasibility_tolerance must be finite and nonnegative")
        relaxed, upper = self._relaxation(None, bound_from_duals=True)
        if relaxed is None:
            raise ValueError("the convex relaxation is infeasible")

        deficit = max(0.0, relaxed.budget_residual)
        trades = relaxed.trades.copy()
        trades[cash_index] -= deficit
        trial = self._result(trades)
        feasible = self._is_feasible(trial, feasibility_tolerance)
        corrected = trial if feasible else None
        gap = max(0.0, upper - trial.wealth) if feasible else None
        if gap is not None and gap <= epsilon:
            return CashCorrectionResult(corrected, corrected, upper, deficit, gap, True, 1)

        fallback = self._iterate(relaxed, upper, None, delta, tolerance, max_iterations)
        candidate = fallback.candidate
        if corrected is not None and (candidate is None or corrected.wealth > candidate.wealth):
            candidate = corrected
        return CashCorrectionResult(candidate, corrected, upper, deficit, gap, False, fallback.solves)

    def _is_feasible(self, candidate, tolerance):
        constraints, _ = self._portfolio_constraints(cp.Constant(candidate.trades))
        return (candidate.budget_residual <= tolerance
                and all(np.max(c.violation()) <= tolerance for c in constraints))

    def exact(self, *, target=None, max_patterns=10_000):
        """Enumerate active sets and asymmetric fee directions for a small instance."""
        return self._enumerate(self._trade_choices(), target, max_patterns)

    def screened_exact(self, *, max_patterns=10_000, numerical_margin=1e-8):
        """Screen trade directions before exact wealth maximization.

        The margin is applied to both incumbent feasibility and the comparison
        with the directional bounds, in floating-point arithmetic.
        """
        screen = self.screen_trades(numerical_margin=numerical_margin)
        before = self._pattern_count(self._trade_choices())
        if screen is None:
            return ScreeningResult(None, (), before, 0, 1)
        can_buy, can_sell = self.allowed_directions(
            screen, screen.tighter_bounds, numerical_margin=numerical_margin,
        )
        choices = self._trade_choices(can_buy, can_sell)
        frozen = np.flatnonzero(~can_buy & ~can_sell & ((self.upper > 0) | (self.lower < 0)))
        candidate = self._enumerate(choices, None, max_patterns)
        return ScreeningResult(candidate, tuple(map(int, frozen)), before,
                               self._pattern_count(choices), screen.setup_solves)

    def screen_trades(self, *, numerical_margin=1e-8):
        """Return directional bounds and an incumbent without running enumeration."""
        if not np.isfinite(numerical_margin) or numerical_margin < 0:
            raise ValueError("numerical_margin must be finite and nonnegative")
        relaxed, bounds = self._relaxation(None, screen=True)
        if relaxed is None:
            return None
        heuristic = self._iterate(relaxed, relaxed.wealth, None, 1e-3, 1e-7, 50)
        idle = self._result(np.zeros(self.n))
        incumbents = [
            candidate for candidate in (idle, heuristic.candidate)
            if candidate is not None and self._is_feasible(candidate, numerical_margin)
        ]
        incumbent = max(incumbents, key=lambda candidate: candidate.wealth, default=None)
        return TradeScreen(incumbent, relaxed.wealth, *bounds, heuristic.solves)

    def allowed_directions(self, screen, bounds, *, numerical_margin=1e-8):
        cutoff = screen.incumbent.wealth if screen.incumbent is not None else -np.inf
        buy_bound, sell_bound = bounds
        can_buy = (self.upper > 0) & (buy_bound + numerical_margin >= cutoff)
        can_sell = (self.lower < 0) & (sell_bound + numerical_margin >= cutoff)
        return can_buy, can_sell

    def _trade_choices(self, can_buy=None, can_sell=None):
        if can_buy is None:
            can_buy, can_sell = self.upper > 0, self.lower < 0
        choices = []
        for buys, sells, buy, sell in zip(can_buy, can_sell, self.buy_fee, self.sell_fee):
            directions = []
            if buys:
                directions.append(Trade.BUY)
            if sells:
                directions.append(Trade.SELL)
            if buy == sell and len(directions) == 2:
                directions = [Trade.EITHER]
            if buy == sell == 0:
                choices.append(directions or [Trade.IDLE])
            else:
                choices.append([Trade.IDLE, *directions])
        return choices

    @staticmethod
    def _pattern_count(choices):
        return int(np.prod([len(options) for options in choices], dtype=object))

    def _enumerate(self, choices, target, max_patterns):
        if self._pattern_count(choices) > max_patterns:
            raise ValueError("too many patterns for exhaustive enumeration")
        best = None
        for pattern in product(*choices):
            candidate, _ = self._solve(self.buy_rate, self.sell_rate, target, np.array(pattern))
            if candidate is None:
                continue
            score = candidate.wealth if target is None else -candidate.fees
            if best is None or score > best_score:
                best, best_score = candidate, score
        return best


if __name__ == "__main__":
    # Synthetic example: two risky assets and a fee-free cash position.
    model = Portfolio(
        holdings=[0.3, 0.2, 0.5], returns=[1.12, 1.04, 1.0],
        covariance=np.diag([0.04, 0.01, 0]),
        lower=[-0.3, -0.2, -0.5], upper=[0.7, 0.8, 0.5],
        buy_rate=[0.01, 0.01, 0], sell_rate=[0.01, 0.01, 0],
        buy_fee=[0.005, 0.005, 0], sell_fee=[0.005, 0.005, 0],
        min_position=0, volatility=0.12,
    )
    result = model.heuristic()
    exact = model.exact()
    print(f"Relaxed upper bound: {result.relaxation.wealth:.8f}")
    print(f"Heuristic: {result.iterations} iterations, converged={result.converged}")
    if result.candidate is not None:
        print(f"Heuristic wealth: {result.candidate.wealth:.8f}")
        print(f"Exact fees: {result.candidate.fees:.8f}")
        print(f"Budget residual: {result.candidate.budget_residual:.2e}")
        print(f"Trades: {result.candidate.trades}")
    print(f"Exhaustive optimum: {exact.wealth:.8f}")
