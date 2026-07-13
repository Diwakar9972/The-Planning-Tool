"""
Levisol Planning Tool — 1000-experiment CPU simulation on the graph model.

Scenario generator (seeded => fully reproducible):
  * Demand per CFA x slab ~ Normal(Jan forecast, historical CV), floored at 0
  * Plant availability ~ Uniform(0.85, 1.00)  (maintenance / breakdown)
  * Conditional links: each hub->CFA link has a 10% chance of disruption
    (weather / traffic): capacity capped at 60% of its nominal load and
    transit time stretched 1.5x
Each scenario is solved EXACTLY as an LP with the composite 40/30/20/10
objective. The recommended plan is the optimum of the expected scenario;
the 1000 experiments quantify how robust the network is around it.
"""
import numpy as np
import pandas as pd

from .graph_model import build_network, solve_scenario, PLANTS, HUBS

SEED = 42


def make_scenarios(data, net, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    base = net['demand']
    cv = net['cv']
    idx = list(base.index)
    mean = base.to_numpy(float)
    sd = np.array([cv.get(k, 0.2) for k in idx]) * mean
    scenarios = []
    links = [(h, c) for h in HUBS for c in net['cfas']]
    nominal_share = mean.sum() / len(net['cfas'])
    for _ in range(n):
        d = np.maximum(rng.normal(mean, sd), 0.0)
        avail = {p: rng.uniform(0.85, 1.0) for p in PLANTS}
        link_cap, link_tm = {}, {}
        for (h, c) in links:
            if rng.random() < 0.10:
                link_cap[(h, c)] = 0.6 * nominal_share
                link_tm[(h, c)] = 1.5
        scenarios.append(dict(demand=dict(zip(idx, d)), plant_avail=avail,
                              link_cap=link_cap, link_time_mult=link_tm))
    return scenarios


def run_simulation(data, n=1000, seed=SEED, weights=None):
    from .graph_model import WEIGHTS
    weights = weights or WEIGHTS
    net = build_network(data)

    # recommended plan = expected scenario (deterministic anchor)
    base = dict(demand=dict(net['demand']), plant_avail={p: 1.0 for p in PLANTS},
                link_cap={}, link_time_mult={})
    plan = solve_scenario(data, net, base, weights=weights)

    rows = []
    for sc in make_scenarios(data, net, n=n, seed=seed):
        r = solve_scenario(data, net, sc, weights=weights)
        rows.append({
            'composite_score': r['score'], 'cost_rs': r['cost'] + r['penalty'],
            'avg_transit_days': r['avg_days'], 'env_proxy': r['env'],
            'fill_rate': r['fill_rate'], 'unmet_kl': r['unmet'],
            'util_BOM': r['util']['BOM'], 'util_AHM': r['util']['AHM'],
            'util_KOL': r['util']['KOL'],
            **{f'flow_{p}_{h}': r['ph_flow'][(p, h)] for p in PLANTS for h in HUBS},
        })
    df = pd.DataFrame(rows)
    return net, plan, df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    q = df.quantile([0.05, 0.5, 0.95]).T
    q.columns = ['P5', 'P50', 'P95']
    q['mean'] = df.mean()
    return q.round(3)
