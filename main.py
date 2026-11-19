#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates on Exophase, one week after launch.

An earn rate is earners divided by players tracked, and both sides move.
New players join faster than they earn trophies, so a rate usually falls
when Exophase refreshes its counts even while the number of earners
grows. This program therefore forecasts earners and players separately
and divides them again:

  * Players tracked: a power-law growth curve, fitted as log(players)
    against log(1 + hours since release), and read at the target. When
    the history is flat, so the fit says nothing moves, the launch-night
    prior does the work instead: one refresh can multiply the players
    tracked by about 3868 / 638, and a refresh comes roughly every four
    and a half hours, so the forecast is the log-scale expectation over
    the refreshes the horizon is expected to hold.
  * Earners per trophy: two views, blended. One view is the trophy's
    own growth in log time, its slope shrunk toward the median growth
    of its tier, with a tier fallback slope when the history cannot
    show a slope at all. The other view is elasticity: how fast
    earners grow when the player count grows, fitted per trophy and
    shrunk toward its tier's prior (hard trophies barely move with the
    crowd, easy ones move with it). The blended earners forecast is
    divided by the players forecast to get a rate.
  * A second, smaller weight comes from a straight trend of the rate's
    logit in log time, which catches trophies whose rate was already
    drifting in the history.
  * The spread, in logit units, combines the trend fit's residual
    noise, a term that grows with the horizon, a refresh term (wide
    when the history has not shown a refresh yet, because one may be
    coming), and extra noise for trophies with only a handful of
    earners. The five quantiles come from the median and the spread
    with the standard normal z-scores.

When the history is completely flat, as on launch night, the data fits
say nothing moves, so the priors do the work: players are expected to
keep growing and rates to dilute, which is exactly what a refresh
does. A trophy already at 100% stays at 100%.

Standard library only, no network, no randomness: two runs on the
same case write the same bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

IN_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

WANTED_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
NORMAL_Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# How fast earners are assumed to follow the player count, by tier,
# when the history cannot show it yet. Hard trophies lag the crowd.
TIER_FOLLOW = {"platinum": 0.05, "gold": 0.20, "silver": 0.35, "bronze": 0.55}
# Fallback earners growth in log time, by tier, for the same case.
TIER_GROWTH = {"platinum": 0.10, "gold": 0.30, "silver": 0.50, "bronze": 0.80}
# How much one refresh multiplied the players tracked on launch night.
REFRESH_JUMP = 3868.0 / 638.0
# Roughly how often Exophase refreshes a game's counts: the first
# refresh on launch night came about four and a half hours after the
# first probe, so a flat history is expected to see about one refresh
# per four and a half hours of horizon, and the players forecast is
# the log-scale expectation over that count.
REFRESH_HOURS = 4.5
# Blend of the two earners views, in log space (elasticity view heavier:
# it is the one that carries the refresh prior when log time is flat).
BLEND_LOGTIME = 0.35
BLEND_ELASTICITY = 0.65
# Weight of the straight logit-rate trend in the median, in logit space.
TREND_WEIGHT = 0.30
# Refresh-term spread (logit) when no refresh has been seen yet, and
# once refreshes have been seen in the history.
SPREAD_REFRESH_UNSEEN = 0.50
SPREAD_REFRESH_SEEN = 0.30
SPREAD_FLOOR = 0.35
SPREAD_CAP = 2.00


def _clamp_pct(value):
    return min(99.99, max(0.01, value))


def _logit_pct(value):
    p = _clamp_pct(value) / 100.0
    return math.log(p / (1.0 - p))


def _inv_logit_pct(value):
    if value > 30.0:
        return 100.0
    if value < -30.0:
        return 0.0
    return 100.0 / (1.0 + math.exp(-value))


def _parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    if n % 2:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def _fit_line(xs, ys):
    """Least squares slope and intercept, or None when degenerate."""
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return (slope, mean_y - slope * mean_x)


def _forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history", [])
    release = _parse_time(case["game"]["release_date"] + "T00:00:00Z")
    target = _parse_time(case["target"])
    horizon_days = case.get("horizon_minutes", 0) / 1440.0
    target_x = math.log1p(max((target - release).total_seconds() / 3600.0, -0.9))

    # Time coordinate per read, and the last read's numbers.
    reads = []
    for read in history:
        at = _parse_time(read["at"])
        hours = (at - release).total_seconds() / 3600.0
        reads.append({
            "x": math.log1p(max(hours, -0.9)),
            "players": float(read.get("players_tracked") or 0),
            "rates": read.get("rates") or [],
        })
    if not reads:
        return {"trophies": [
            {"id": t["id"], "q": [0.5, 1.0, 2.0, 4.0, 8.0]}
            for t in trophies
        ]}

    last = reads[-1]
    last_players = last["players"] if last["players"] > 0 else 1.0
    last_x = last["x"]

    # Did the history show a refresh? A refresh moves the player count
    # or the rates between two neighbouring reads.
    refresh_seen = False
    for prev, cur in zip(reads, reads[1:]):
        if prev["players"] > 0 and abs(cur["players"] - prev["players"]) / prev["players"] > 0.005:
            refresh_seen = True
            break
        for a, b in zip(prev["rates"], cur["rates"]):
            if a is not None and b is not None and abs(b - a) > 0.02:
                refresh_seen = True
                break
        if refresh_seen:
            break

    # Players tracked at the target: power-law fit, or the refresh
    # prior when the history is flat. A refresh is not certain to
    # have happened by the target, so the flat-history forecast is
    # the log-scale expectation over the refreshes the horizon is
    # expected to hold, each multiplying the players by one jump.
    player_xs = [r["x"] for r in reads if r["players"] > 0]
    player_ys = [math.log(r["players"]) for r in reads if r["players"] > 0]
    player_fit = _fit_line(player_xs, player_ys)
    flat_players = (
        player_fit is None
        or len(set(player_ys)) < 2
    )
    if flat_players:
        expected_refreshes = (case.get("horizon_minutes", 0) / 60.0) / REFRESH_HOURS
        players_target = last_players * (REFRESH_JUMP ** expected_refreshes)
    else:
        slope, intercept = player_fit
        players_target = math.exp(intercept + slope * target_x)
        players_target = max(players_target, last_players)

    # Per-trophy series: (log-time x, players, rate) triples.
    series = []
    for idx, trophy in enumerate(trophies):
        points = []
        for read in reads:
            rate = read["rates"][idx] if idx < len(read["rates"]) else None
            if rate is None or read["players"] <= 0:
                continue
            points.append((read["x"], read["players"], float(rate)))
        series.append(points)

    # Tier medians of the fitted log-time earners slopes, for shrinkage.
    tier_slopes = {}
    for idx, trophy in enumerate(trophies):
        points = series[idx]
        if len(points) < 3:
            continue
        xs = [p[0] for p in points]
        ys = [math.log(max(0.5, p[2] / 100.0 * p[1])) for p in points]
        if len(set(ys)) < 2:
            continue
        fit = _fit_line(xs, ys)
        if fit is not None:
            tier_slopes.setdefault(trophy.get("tier"), []).append(fit[0])
    tier_median_slope = {
        tier: _median(slopes) for tier, slopes in tier_slopes.items()
    }

    rows = []
    for idx, trophy in enumerate(trophies):
        tier = trophy.get("tier") or "bronze"
        points = series[idx]
        last_rate = points[-1][2] if points else None
        if last_rate is None:
            last_rate = 5.0 if tier == "bronze" else 2.0 if tier == "silver" else 1.0 if tier == "gold" else 0.5
        if last_rate >= 99.99:
            rows.append({"id": trophy["id"], "q": [100.0, 100.0, 100.0, 100.0, 100.0]})
            continue
        earners_last = max(0.5, last_rate / 100.0 * last_players)

        # View one: the trophy's own growth in log time.
        growth = TIER_GROWTH.get(tier, 0.80)
        if len(points) >= 3:
            xs = [p[0] for p in points]
            ys = [math.log(max(0.5, p[2] / 100.0 * p[1])) for p in points]
            fit = _fit_line(xs, ys) if len(set(ys)) >= 2 else None
            if fit is not None:
                own_slope = fit[0]
                anchor = tier_median_slope.get(tier, growth)
                weight = len(points) / (len(points) + 3.0)
                growth = weight * own_slope + (1.0 - weight) * anchor
        earners_logtime = earners_last * math.exp(
            growth * (target_x - last_x)
        )

        # View two: elasticity of earners to the player count.
        elasticity = TIER_FOLLOW.get(tier, 0.55)
        if len(points) >= 3:
            xs = [math.log(p[1]) for p in points if p[1] > 0]
            ys = [math.log(max(0.5, p[2] / 100.0 * p[1])) for p in points if p[1] > 0]
            fit = _fit_line(xs, ys) if len(set(ys)) >= 2 and len(set(xs)) >= 2 else None
            if fit is not None:
                own = fit[0]
                weight = len(points) / (len(points) + 3.0)
                elasticity = weight * own + (1.0 - weight) * elasticity
                elasticity = min(1.5, max(-0.5, elasticity))
        earners_elastic = earners_last * (
            (players_target / last_players) ** elasticity
        )

        # Blend the two earners views, in log space.
        earners_blend = math.exp(
            BLEND_LOGTIME * math.log(max(0.5, earners_logtime))
            + BLEND_ELASTICITY * math.log(max(0.5, earners_elastic))
        )
        rate_ep = earners_blend / players_target * 100.0

        # Straight trend of the rate's logit in log time.
        trend_rate = last_rate
        residual = 0.0
        if len(points) >= 3:
            xs = [p[0] for p in points]
            ys = [_logit_pct(p[2]) for p in points]
            fit = _fit_line(xs, ys) if len(set(ys)) >= 2 else None
            if fit is not None:
                slope, intercept = fit
                trend_rate = _inv_logit_pct(intercept + slope * target_x)
                sq = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
                residual = math.sqrt(sq / len(xs))

        # Median: mostly the earners-over-players rate, with the trend
        # blended in on the logit scale.
        median_logit = (
            (1.0 - TREND_WEIGHT) * _logit_pct(rate_ep)
            + TREND_WEIGHT * _logit_pct(trend_rate)
        )
        median_rate = _inv_logit_pct(median_logit)

        # Spread, in logit units.
        horizon_term = 0.15 * math.sqrt(max(horizon_days, 0.0))
        refresh_term = SPREAD_REFRESH_SEEN if refresh_seen else SPREAD_REFRESH_UNSEEN
        small_term = 0.5 / math.sqrt(max(1.0, earners_last))
        spread = math.sqrt(
            residual ** 2 + horizon_term ** 2 + refresh_term ** 2 + small_term ** 2
        )
        spread = min(SPREAD_CAP, max(SPREAD_FLOOR, spread))

        quantiles = []
        for z in NORMAL_Z:
            q = _inv_logit_pct(median_logit + z * spread)
            quantiles.append(round(min(99.99, max(0.01, q)), 4))
        quantiles.sort()
        rows.append({"id": trophy["id"], "q": quantiles})

    return {"trophies": rows}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(IN_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = _forecast_case(case)
        out_path = OUT_DIR / (path.stem + ".out")
        out_path.write_text(
            json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
