# GTA VI Trophy Rarity Forecast

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, its earn rate on Exophase one week after launch.

[![Bounty](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/sedayu79/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/sedayu79/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Entry for the [GTA VI trophy rarity forecast bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) on make.fun.

## How it works

`main.py` (Python 3.13, standard library only, no network, deterministic) forecasts earners and players separately and divides them again: players tracked follow a power-law growth curve in log time, earners follow a blend of their own log-time growth (shrunk toward their tier's median) and elasticity to the player count (shrunk toward tier priors). For each trophy it writes five quantiles (5%, 25%, 50%, 75%, 95%) of the earn rate at the case's target. See the docstring at the top of `main.py` for the full model.

## Run it

```sh
MAKEFUN_INPUT_DIR=<dir with .in case files> MAKEFUN_OUTPUT_DIR=<out dir> python3 main.py
```

Each case writes one `<case>.out` file with 55 trophies and five quantiles each.
