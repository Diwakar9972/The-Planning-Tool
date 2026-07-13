"""
Levisol Planning Tool — Graph-based distribution network model.

Graph:  Sources (plants BOM/AHM/KOL, per production-line slab)
        -> Hubs (MHW, MHE) with conditional processing capacity
        -> Destinations (10 CFAs) with scenario demand.

Multi-criteria composite objective (weightages):
    40% cost  (production ₹ + freight ₹)
    30% time  (volume-weighted transit days)
    20% capacity utilisation (congestion penalty above a utilisation knee)
    10% environmental impact (CO2 proxy ~ tonne-km ~ freight ₹)
Each criterion is normalised by a baseline value so terms are commensurate,
then combined into one linear objective and solved exactly as an LP (HiGHS).

Conditional elements:
  * Conditional hub: throughput above a threshold incurs an overflow
    processing cost (models processing capacity that degrades with load).
  * Conditional links: a scenario can cap a link's capacity and stretch its
    transit time (weather / traffic disruption).

Aggregation: SKUs are aggregated to the 5 production-line slabs so one
scenario solves in milliseconds — the SKU-level MILP remains the detailed
monthly planner; this layer answers network-design and robustness questions.
"""
import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import linprog

PLANTS = ['BOM', 'AHM', 'KOL']
HUBS = ['MHW', 'MHE']
SLABS = ['<=1.5LT', '3-5LT', '7-20LT', '50LT', '180-210LT']

WEIGHTS = {'cost': 0.40, 'time': 0.30, 'util': 0.20, 'env': 0.10}
UTIL_KNEE = 0.85          # utilisation above this attracts congestion penalty
PH_COST_PER_DAY = 2000.0  # freight-₹ -> days proxy for plant->hub edges


def build_network(data):
    """Static network parameters shared by all scenarios."""
    cfas = sorted({c for (_h, c) in data['hub_cfa_cost'].keys()})
    lt = data['lt']
    hc_time = lt.groupby('CFA')['lt_hub_cfa'].median().to_dict()

    # aggregate Jan demand and penalty by CFA x slab
    jan = data['jan'].copy()
    jan['slab'] = jan['Product Name'].map(data['sku_slab'])
    jan['pen'] = jan['Product Name'].map(data['sku_penalty'])
    dem = jan.groupby(['CFA', 'slab'])['jan_fcst'].sum()
    pen = jan.groupby(['CFA', 'slab']).apply(
        lambda g: np.average(g['pen'], weights=np.maximum(g['jan_fcst'], 1e-6)),
        include_groups=False)

    # demand CV per CFA x slab from 6-month sales (for scenario generation)
    from .data_loader import MONTHS
    s = data['sales'].copy()
    s['slab'] = s['Product Name'].map(data['sku_slab'])
    agg = s.groupby(['CFA', 'slab'])[MONTHS].sum()
    cv = (agg.std(axis=1, ddof=1) / agg.mean(axis=1).clip(lower=1e-9)).clip(0.03, 0.6)

    # hub opening inventory by slab
    hi = data['hub_inv'].copy()
    hi['slab'] = hi['Product Name'].map(data['sku_slab'])
    hub_open = hi.groupby(['Hub', 'slab'])['open_inv'].sum()

    G = nx.DiGraph()
    for p in PLANTS:
        G.add_node(p, kind='source', cost=data['prod_cost'][p],
                   capacity=data['plant_capacity'][p])
    for h in HUBS:
        G.add_node(h, kind='hub')
    for c in cfas:
        G.add_node(c, kind='destination')
    for p in PLANTS:
        for h in HUBS:
            fr = data['plant_hub_cost'][(p, h)]
            G.add_edge(p, h, freight=fr, days=fr / PH_COST_PER_DAY, env=fr)
    for h in HUBS:
        for c in cfas:
            fr = data['hub_cfa_cost'][(h, c)]
            G.add_edge(h, c, freight=fr, days=float(hc_time.get(c, 3)), env=fr)

    return dict(G=G, cfas=cfas, demand=dem, penalty=pen, cv=cv, hub_open=hub_open)


def solve_scenario(data, net, scenario, weights=WEIGHTS,
                   hub_threshold_frac=0.8, hub_overflow_cost=800.0):
    """
    Solve one scenario as an exact LP with the composite objective.
    scenario = dict(demand={(cfa,slab):kL}, plant_avail={plant:0-1},
                    link_cap={(h,c):kL or None}, link_time_mult={(h,c):x})
    Returns plan dict with flows, criteria values, composite score.
    """
    G, cfas = net['G'], net['cfas']
    dem = scenario['demand']
    keys_pc = [(p, s) for p in PLANTS for s in SLABS if data['plant_capacity'][p][s] > 0]
    keys_x = [(p, h, s) for (p, s) in keys_pc for h in HUBS]
    keys_y = [(h, c, s) for h in HUBS for c in cfas for s in SLABS]
    keys_u = [(c, s) for c in cfas for s in SLABS]
    keys_ov = [(p, s) for (p, s) in keys_pc]          # utilisation overflow
    keys_hov = list(HUBS)                             # hub processing overflow

    off = {}
    n = 0
    for name, keys in [('x', keys_x), ('y', keys_y), ('u', keys_u),
                       ('ov', keys_ov), ('hov', keys_hov)]:
        off[name] = {k: n + i for i, k in enumerate(keys)}
        n += len(keys)

    # ---- normalisers from a baseline: everything met at mean cost/time ----
    tot_dem = sum(dem.values())
    mean_prod = np.mean([data['prod_cost'][p] for p in PLANTS])
    mean_fr = np.mean([e['freight'] for _, _, e in G.edges(data=True)])
    mean_days = np.mean([e['days'] for _, _, e in G.edges(data=True)])
    C0 = tot_dem * (mean_prod + 2 * mean_fr)      # ₹ scale
    T0 = tot_dem * mean_days                      # kL·days scale
    E0 = tot_dem * 2 * mean_fr                    # env proxy scale
    U0 = 0.15 * tot_dem                           # kL above knee scale
    SCALE = C0                                    # report score in ₹-equivalent

    w = weights
    c_vec = np.zeros(n)
    for (p, h, s), j in off['x'].items():
        e = G.edges[p, h]
        c_vec[j] = SCALE * (w['cost'] * (data['prod_cost'][p] + e['freight']) / C0
                            + w['time'] * e['days'] / T0
                            + w['env'] * e['env'] / E0)
    for (h, c, s), j in off['y'].items():
        e = G.edges[h, c]
        tmult = scenario.get('link_time_mult', {}).get((h, c), 1.0)
        c_vec[j] = SCALE * (w['cost'] * e['freight'] / C0
                            + w['time'] * e['days'] * tmult / T0
                            + w['env'] * e['env'] / E0)
    for (c, s), j in off['u'].items():
        c_vec[j] = SCALE * w['cost'] * float(net['penalty'].get((c, s), 200000.0)) / C0
    for (p, s), j in off['ov'].items():
        c_vec[j] = SCALE * w['util'] / U0
    for h in HUBS:
        c_vec[off['hov'][h]] = hub_overflow_cost  # ₹/kL conditional processing

    A_eq, b_eq, A_ub, b_ub = [], [], [], []

    def row(pairs):
        r = np.zeros(n)
        for j, v in pairs:
            r[j] = v
        return r

    # demand balance: sum_h y + u = D
    for (c, s) in keys_u:
        pairs = [(off['y'][(h, c, s)], 1.0) for h in HUBS] + [(off['u'][(c, s)], 1.0)]
        A_eq.append(row(pairs)); b_eq.append(float(dem.get((c, s), 0.0)))

    # hub conservation by slab: outbound <= opening + inbound
    for h in HUBS:
        for s in SLABS:
            pairs = ([(off['y'][(h, c, s)], 1.0) for c in cfas] +
                     [(off['x'][(p, h, s)], -1.0) for p in PLANTS if (p, h, s) in off['x']])
            A_ub.append(row(pairs)); b_ub.append(float(net['hub_open'].get((h, s), 0.0)))

    # plant line capacity with availability shock; utilisation overflow above knee
    for (p, s) in keys_pc:
        cap = data['plant_capacity'][p][s] * scenario.get('plant_avail', {}).get(p, 1.0)
        prod_pairs = [(off['x'][(p, h, s)], 1.0) for h in HUBS]
        A_ub.append(row(prod_pairs)); b_ub.append(cap)
        A_ub.append(row(prod_pairs + [(off['ov'][(p, s)], -1.0)]))
        b_ub.append(UTIL_KNEE * cap)

    # conditional hub processing: inflow above threshold -> overflow var
    for h in HUBS:
        pairs = [(off['x'][k], 1.0) for k in off['x'] if k[1] == h]
        thresh = hub_threshold_frac * tot_dem / len(HUBS)
        A_ub.append(row(pairs + [(off['hov'][h], -1.0)])); b_ub.append(thresh)

    # conditional link capacity caps
    for (h, c), cap in scenario.get('link_cap', {}).items():
        if cap is None:
            continue
        pairs = [(off['y'][(h, c, s)], 1.0) for s in SLABS]
        A_ub.append(row(pairs)); b_ub.append(cap)

    res = linprog(c_vec, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                  bounds=(0, None), method='highs')
    if res.x is None:
        raise RuntimeError(res.message)
    x = res.x

    # ---- criteria accounting ----
    cost = time_kld = env = 0.0
    for (p, h, s), j in off['x'].items():
        v = x[j]
        e = G.edges[p, h]
        cost += v * (data['prod_cost'][p] + e['freight'])
        time_kld += v * e['days']; env += v * e['env']
    for (h, c, s), j in off['y'].items():
        v = x[j]
        e = G.edges[h, c]
        tmult = scenario.get('link_time_mult', {}).get((h, c), 1.0)
        cost += v * e['freight']; time_kld += v * e['days'] * tmult; env += v * e['env']
    unmet = sum(x[j] for j in off['u'].values())
    pen_cost = sum(x[j] * float(net['penalty'].get(k, 200000.0))
                   for k, j in off['u'].items())
    over_util = sum(x[j] for j in off['ov'].values())
    served = tot_dem - unmet
    util_by_plant = {}
    for p in PLANTS:
        prod = sum(x[off['x'][k]] for k in off['x'] if k[0] == p)
        cap = sum(data['plant_capacity'][p][s] * scenario.get('plant_avail', {}).get(p, 1.0)
                  for s in SLABS)
        util_by_plant[p] = prod / cap if cap else 0.0
    ph_flow = {(p, h): sum(x[off['x'][k]] for k in off['x'] if k[0] == p and k[1] == h)
               for p in PLANTS for h in HUBS}
    return dict(score=res.fun, cost=cost, penalty=pen_cost,
                avg_days=time_kld / max(served, 1e-9), env=env,
                unmet=unmet, fill_rate=served / max(tot_dem, 1e-9),
                over_util=over_util, util=util_by_plant, ph_flow=ph_flow,
                y=[(k, x[j]) for k, j in off['y'].items() if x[j] > 1e-6],
                x=[(k, x[j]) for k, j in off['x'].items() if x[j] > 1e-6])
