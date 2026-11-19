#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates on Exophase, one week after launch.

Model, in plain words: an earn rate is earners divided by players
tracked, and both sides move. New players join faster than they earn
trophies, so a rate usually falls when Exophase refreshes its counts
even while the number of earners grows. This program therefore
forecasts earners and players separately and divides them again:

  * Players tracked: fit log(players) against log(1 + hours since
    release). That is a power-law growth curve; read it at the target
    to get the future player count.
  * Earners per trophy: two views, blended. One is the trophy's own
    growth in log time, shrunk toward the median growth of its tier
    (platinum trophies are earned far slower than bronze ones). The
    other is elasticity: how fast earners grow when the player count
    grows, regressed per trophy and shrunk toward its tier's prior
    (hard trophies barely move with the crowd, easy ones move with it).
  * The median rate is the blended earners forecast divided by the
    players forecast. A second, smaller weight comes from a straight
    trend of the rate's logit in log time, which catches trophies
    whose rate was already drifting in the history.
  * The spread (in logit units) combines the fit's residual noise,
    a term that grows with the horizon, a refresh term (wide when the
    history has not shown a refresh yet, because one may be coming),
    and extra noise for trophies with only a handful of earners.

When the history is completely flat, as on launch night, the data
fits say nothing moves, so the priors do the work: players are
expected to keep growing and rates to dilute, which is exactly what
a refresh does. A trophy already at 100% stays at 100%.

Standard library only, no network, no randomness: two runs on the
same case write the same bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

IN_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

WANTED_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
NORMAL_Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# How fast earners are assumed to follow the player count, by tier,
# when the history cannot show it yet. Hard trophies lag the crowd.
TIER_FOLLOW = {"platinum": 0.05, "gold": 0.20, "silver": 0.35, "bronze": 0.55}
# Fallback earners growth in log time, by tier, for the same case.
TIER_GROWTH = {"platinum": 0.10, "gold": 0.30, "silver": 0.50, "bronze": 0.80}


def _clamp_pct(value):
    return min(99.99, max(0.01, value))


def _logit(pct):
    p = _clamp_pct(pct)
    return math.log(p / (100.0 - p))


def _inv_logit_pct(logit_value):
    if logit_value >= 0:
        odds = math.exp(-logit_value)
        return 100.0 / (1.0 + odds)
    odds = math.exp(logit_value)
    return 100.0 * odds / (1.0 + odds)


def _round4(value):
    """Round to 4 decimals, halves up, whole numbers as plain ints."""
    scaled = value * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    out = whole / 10000.0
    return int(out) if out == int(out) else out


def _parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _hours_between(start, end):
    return (end - start).total_seconds() / 3600.0


def _fit_line(xs, ys):
    """Least squares y = a + b*x. Returns (slope, intercept, resid_std)."""
    n = len(xs)
    if n < 2:
        return 0.0, (ys[0] if ys else 0.0), 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        resid = [y - my for y in ys]
        return 0.0, my, math.sqrt(sum(r * r for r in resid) / n)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    return slope, intercept, math.sqrt(sum(r * r for r in resid) / n)


def _median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _forecast_trophy(trophy, series, target_hours, horizon_days,
                     player_growth, tier_beta, tier_follow, flat_history):
    """series: list of (hours, players, rate) for this trophy, oldest first."""
    last_hours, last_players, last_rate = series[-1]
    if last_rate is not None and last_rate >= 99.99:
        return [99.8, 99.9, 100, 100, 100]

    tier = trophy.get("tier", "bronze")
    follow_prior = TIER_FOLLOW.get(tier, 0.55)
    growth_prior = TIER_GROWTH.get(tier, 0.80)

    # --- players at the target, from the fitted growth curve ---
    players_now = max(1.0, float(last_players))
    players_target = players_now * player_growth
    growth_factor = max(1.0, players_target / players_now)

    # --- earners: time growth, shrunk toward the tier median ---
    earners = []
    log_time = []
    for hours, players, rate in series:
        if rate is None:
            continue
        earners.append(max(0.5, rate / 100.0 * players))
        log_time.append(math.log1p(max(0.0, hours)))
    last_earners = earners[-1]
    if len(earners) >= 3:
        beta, _, _ = _fit_line(log_time, [math.log(e) for e in earners])
        beta = min(3.0, max(-1.0, beta))
        beta = 0.6 * beta + 0.4 * tier_beta
    else:
        beta = tier_beta if tier_beta is not None else growth_prior
    time_factor = ((1.0 + max(0.0, target_hours)) /
                   (1.0 + max(0.0, last_hours))) ** beta
    earners_by_time = last_earners * time_factor

    # --- earners: elasticity to the player count ---
    if len(earners) >= 3 and len(set(p for _, p, _ in series)) >= 3:
        log_p = [math.log(max(1.0, float(p))) for _, p, _ in series
                 if _ is not None]
        # recompute earners aligned with those reads
        aligned = [(math.log(max(1.0, float(p))),
                    math.log(max(0.5, r / 100.0 * p)))
                   for _, p, r in series if r is not None]
        xs = [a for a, _ in aligned]
        ys = [b for _, b in aligned]
        gamma, _, _ = _fit_line(xs, ys)
        gamma = min(1.5, max(-0.5, gamma))
        gamma = 0.5 * gamma + 0.5 * follow_prior
    else:
        gamma = follow_prior
    earners_by_elasticity = last_earners * (growth_factor ** gamma)

    # Blend the two earners views (geometric mean) and divide by players.
    earners_pred = math.sqrt(max(0.5, earners_by_time) *
                             max(0.5, earners_by_elasticity))
    rate_pred = _clamp_pct(100.0 * earners_pred / players_target)

    # --- trend of the rate itself, as a smaller second opinion ---
    rate_points = [(math.log1p(max(0.0, h)), _logit(r))
                   for h, _, r in series if r is not None]
    if len(rate_points) >= 3:
        xs = [a for a, _ in rate_points]
        ys = [b for _, b in rate_points]
        slope, intercept, resid = _fit_line(xs, ys)
        trend_logit = intercept + slope * math.log1p(max(0.0, target_hours))
        median_logit = 0.7 * _logit(rate_pred) + 0.3 * trend_logit
    else:
        resid = 0.4
        median_logit = _logit(rate_pred)

    # --- spread in logit units ---
    spread_sq = resid * resid
    spread_sq += (0.22 * math.sqrt(max(0.0, horizon_days))) ** 2
    if flat_history:
        # No refresh has been seen yet, but one may land before the
        # target: widen the band instead of pretending rates are frozen.
        spread_sq += 0.7 * 0.7
    else:
        spread_sq += 0.3 * 0.3
    spread_sq += min(0.25, 0.5 / math.sqrt(max(1.0, last_earners)))
    spread = min(2.0, max(0.45, math.sqrt(spread_sq)))

    quantiles = []
    previous = 0.01
    for z in NORMAL_Z:
        value = _inv_logit_pct(median_logit + z * spread)
        value = min(100.0, max(0.01, value))
        value = max(previous, value)
        quantiles.append(_round4(value))
        previous = value
    return quantiles


def forecast_case(case):
    if case.get("levels") != WANTED_LEVELS:
        raise ValueError("levels are not the five scored levels")
    history = case.get("history") or []
    if not history:
        raise ValueError("case has no history")
    trophies = case["trophies"]
    release_text = (case.get("game") or {}).get("release_date", "2026-11-19")
    release = _parse_time(release_text + "T00:00:00Z")
    target = _parse_time(case["target"])
    checkpoint = _parse_time(case["at"])
    horizon_days = case["horizon_minutes"] / 1440.0
    target_hours = _hours_between(release, target)

    # Player counts over time, and whether any refresh ever moved them.
    player_points = []
    for read in history:
        players = read.get("players_tracked")
        if players:
            player_points.append(
                (math.log1p(max(0.0, _hours_between(release, _parse_time(read["at"])))),
                 math.log(float(players))))
    flat_history = len({p for _, p in player_points}) < 2
    if flat_history:
        # No refresh has been seen yet, so the fit says players never
        # move -- but Exophase refreshes every few hours and the tracked
        # count jumps when it does (on launch night it went 638 to
        # 3,868 across one refresh). Assume the count keeps growing
        # while the target approaches, at roughly a quarter of a log
        # unit an hour, capped so a long flat history cannot explode.
        horizon_hours = case["horizon_minutes"] / 60.0
        player_growth = min(4.0, math.exp(0.25 * horizon_hours))
    elif len(player_points) >= 2:
        alpha, intercept, _ = _fit_line([a for a, _ in player_points],
                                        [b for _, b in player_points])
        alpha = min(2.0, max(0.0, alpha))
        log_players_target = intercept + alpha * math.log1p(max(0.0, target_hours))
        last_players = history[-1].get("players_tracked") or 1
        player_growth = math.exp(log_players_target) / max(1.0, float(last_players))
        player_growth = min(40.0, max(1.0, player_growth))
    else:
        player_growth = 1.0

    # Tier medians of earners growth, for shrinkage.
    tier_betas = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        betas = []
        for idx, trophy in enumerate(trophies):
            if trophy.get("tier") != tier:
                continue
            xs, ys = [], []
            for read in history:
                rate = read["rates"][idx]
                if rate is None:
                    continue
                xs.append(math.log1p(max(0.0, _hours_between(
                    release, _parse_time(read["at"])))))
                ys.append(math.log(max(0.5, rate / 100.0 *
                                       (read.get("players_tracked") or 1))))
            if len(xs) >= 3:
                slope, _, _ = _fit_line(xs, ys)
                betas.append(min(3.0, max(-1.0, slope)))
        tier_betas[tier] = _median(betas) if betas else TIER_GROWTH.get(tier, 0.8)

    rows = []
    for idx, trophy in enumerate(trophies):
        series = []
        for read in history:
            rate = read["rates"][idx]
            series.append((_hours_between(release, _parse_time(read["at"])),
                           read.get("players_tracked") or 1, rate))
        rows.append({
            "id": trophy["id"],
            "q": _forecast_trophy(
                trophy, series, target_hours, horizon_days,
                player_growth, tier_betas.get(trophy.get("tier")),
                TIER_FOLLOW.get(trophy.get("tier"), 0.55), flat_history),
        })
    return {"trophies": rows}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(IN_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case)
        out_path = OUT_DIR / (path.stem + ".out")
        out_path.write_text(
            json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
