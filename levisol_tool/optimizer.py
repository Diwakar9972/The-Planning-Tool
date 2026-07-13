"""
Levisol Planning Tool — Deterministic Production & Distribution MILP
Solved exactly with HiGHS branch-and-bound (scipy.optimize.milp).

Decision variables
  B[s,p]   integer # of 25-kL batches of SKU s at plant p
  X[s,p,h] kL shipped plant p -> hub h            (continuous)
  Y[s,h,c] kL dispatched hub h -> CFA c           (continuous)
  U[s,c]   kL of CFA net requirement left unmet   (continuous, penalised)
  V[s,h]   kL of hub safety-stock shortfall       (continuous, penalised)

Objective  (minimise, all in ₹)
  production + plant->hub freight + hub->CFA freight
  + unmet demand x penalty (x multiplier if contractual)
  + hub SS shortfall x shortfall cost

Constraints (plain language)
  1. Everything produced is shipped to a hub (no stranded stock at plants).
  2. Production per plant per pack-size line <= that line's monthly capacity,
     and is an integer multiple of 25 kL.
  3. A hub can only dispatch what it opened with plus what arrives.
  4. After dispatch, each hub retains its safety-stock norm (V measures any
     shortfall instead of failing).
  5. CFA net requirement = Jan forecast + CFA safety-stock target - opening
     inventory (floored at 0). Anything not shipped shows up in U — the model
     never crashes on infeasibility; scarcity simply becomes a priced choice.
"""
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

BATCH = 25.0
PLANTS = ['BOM', 'AHM', 'KOL']
HUBS = ['MHW', 'MHE']


def build_and_solve(data, cfa_norms, hub_norms,
                    cfa_stock_target='SS',      # 'SS' | 'ROP' | 'none'
                    contractual_multiplier=10.0,
                    hub_shortfall_cost=50000.0,
                    hub_ss_scale=1.0,           # global x on hub SS norms
                    hub_ss_override=None,       # {(sku,hub): kL} hard overrides
                    time_limit=60):
    jan = data['jan'].set_index(['Product Name', 'CFA'])['jan_fcst']
    cfa_open = data['cfa_inv'].set_index(['Product Name', 'CFA'])['open_inv']
    hub_open = data['hub_inv'].set_index(['Product Name', 'Hub'])['open_inv']

    cn = cfa_norms.set_index(['SKU', 'CFA'])
    hn = hub_norms.set_index(['SKU', 'Hub'])

    pairs = list(jan.index)                       # (sku, cfa) demand pairs
    skus = sorted({s for s, _ in pairs})
    tgt_col = {'SS': 'Safety stock (kL)', 'ROP': 'Reorder point (kL)'}.get(cfa_stock_target)

    # ---- Net requirement per SKU-CFA ----
    net_req = {}
    for (s, c) in pairs:
        tgt = float(cn.loc[(s, c), tgt_col]) if tgt_col and (s, c) in cn.index else 0.0
        opn = float(cfa_open.get((s, c), 0.0))
        net_req[(s, c)] = max(0.0, float(jan.loc[(s, c)]) + tgt - opn)

    # ---- Hub SS requirement per SKU-hub (0 if no norm) ----
    hub_ss = {(s, h): hub_ss_scale * float(hn.loc[(s, h), 'Safety stock (kL)'])
              if (s, h) in hn.index else 0.0
              for s in skus for h in HUBS}
    if hub_ss_override:
        for k, v in hub_ss_override.items():
            if k in hub_ss:
                hub_ss[k] = float(v)

    # ---- Variable indexing ----
    idx, names = {}, []

    def add(name):
        idx[name] = len(names)
        names.append(name)

    for s in skus:
        for p in PLANTS:
            if data['plant_capacity'][p][data['sku_slab'][s]] > 0:
                add(('B', s, p))
    for s in skus:
        for p in PLANTS:
            if ('B', s, p) in idx:
                for h in HUBS:
                    add(('X', s, p, h))
    for (s, c) in pairs:
        for h in HUBS:
            add(('Y', s, h, c))
    for (s, c) in pairs:
        add(('U', s, c))
    for s in skus:
        for h in HUBS:
            add(('V', s, h))
    n = len(names)

    # ---- Objective ----
    cost = np.zeros(n)
    for name, j in idx.items():
        kind = name[0]
        if kind == 'B':
            _, s, p = name
            cost[j] = BATCH * data['prod_cost'][p]
        elif kind == 'X':
            _, s, p, h = name
            cost[j] = data['plant_hub_cost'][(p, h)]
        elif kind == 'Y':
            _, s, h, c = name
            cost[j] = data['hub_cfa_cost'][(h, c)]
        elif kind == 'U':
            _, s, c = name
            mult = contractual_multiplier if data['sku_contract'].get(s, False) else 1.0
            cost[j] = data['sku_penalty'][s] * mult
        elif kind == 'V':
            cost[j] = hub_shortfall_cost

    # ---- Constraints ----
    rows, lo, hi = [], [], []
    A = lil_matrix((0, n))

    def add_row(coefs, lb, ub):
        nonlocal A
        A.resize((A.shape[0] + 1, n))
        for j, v in coefs:
            A[A.shape[0] - 1, j] = v
        lo.append(lb)
        hi.append(ub)

    # 1. production balance: sum_h X = 25*B
    for s in skus:
        for p in PLANTS:
            if ('B', s, p) in idx:
                coefs = [(idx[('X', s, p, h)], 1.0) for h in HUBS] + [(idx[('B', s, p)], -BATCH)]
                add_row(coefs, 0.0, 0.0)

    # 2. line capacity per plant per slab
    for p in PLANTS:
        for slab, cap in data['plant_capacity'][p].items():
            coefs = [(idx[('B', s, p)], BATCH) for s in skus
                     if data['sku_slab'][s] == slab and ('B', s, p) in idx]
            if coefs:
                add_row(coefs, 0.0, cap)

    # 3+4. hub balance and hub SS: open + inflow - outflow >= SS - V  (and >= 0)
    for s in skus:
        for h in HUBS:
            inflow = [(idx[('X', s, p, h)], 1.0) for p in PLANTS if ('X', s, p, h) in idx]
            outflow = [(idx[('Y', s, h, c)], -1.0) for (s2, c) in pairs if s2 == s]
            opn = float(hub_open.get((s, h), 0.0))
            add_row(inflow + outflow, -opn, np.inf)                       # end stock >= 0
            if hub_ss[(s, h)] > 0:
                add_row(inflow + outflow + [(idx[('V', s, h)], 1.0)],
                        hub_ss[(s, h)] - opn, np.inf)                      # end stock + V >= SS

    # 5. CFA demand: sum_h Y + U = net requirement
    for (s, c) in pairs:
        coefs = [(idx[('Y', s, h, c)], 1.0) for h in HUBS] + [(idx[('U', s, c)], 1.0)]
        add_row(coefs, net_req[(s, c)], net_req[(s, c)])

    # ---- Bounds & integrality ----
    ub = np.full(n, np.inf)
    integrality = np.zeros(n)
    for name, j in idx.items():
        if name[0] == 'B':
            integrality[j] = 1
        elif name[0] == 'U':
            ub[j] = net_req[(name[1], name[2])]
        elif name[0] == 'V':
            ub[j] = hub_ss[(name[1], name[2])]

    res = milp(c=cost,
               constraints=LinearConstraint(A.tocsr(), np.array(lo), np.array(hi)),
               integrality=integrality,
               bounds=Bounds(np.zeros(n), ub),
               options={'time_limit': time_limit, 'mip_rel_gap': 2e-3})
    if res.x is None:
        raise RuntimeError(f'Solver failed: {res.message}')

    return _extract(res, idx, data, net_req, hub_ss, hub_open, contractual_multiplier,
                    hub_shortfall_cost)


def _extract(res, idx, data, net_req, hub_ss, hub_open, cmult, hub_sc):
    x = res.x
    prod, flows_ph, flows_hc, unmet, hub_short = [], [], [], [], []
    for name, j in idx.items():
        v = x[j]
        if v < 1e-6:
            continue
        if name[0] == 'B':
            _, s, p = name
            prod.append((s, p, round(v), round(v) * BATCH))
        elif name[0] == 'X':
            _, s, p, h = name
            flows_ph.append((s, p, h, v))
        elif name[0] == 'Y':
            _, s, h, c = name
            flows_hc.append((s, h, c, v))
        elif name[0] == 'U':
            _, s, c = name
            unmet.append((s, c, v))
        elif name[0] == 'V':
            _, s, h = name
            hub_short.append((s, h, v))

    prod_df = pd.DataFrame(prod, columns=['SKU', 'Plant', 'Batches (25kL)', 'Production (kL)'])
    ph_df = pd.DataFrame(flows_ph, columns=['SKU', 'Plant', 'Hub', 'Volume (kL)'])
    hc_df = pd.DataFrame(flows_hc, columns=['SKU', 'Hub', 'CFA', 'Volume (kL)'])
    un_df = pd.DataFrame(unmet, columns=['SKU', 'CFA', 'Unmet (kL)'])
    vs_df = pd.DataFrame(hub_short, columns=['SKU', 'Hub', 'SS shortfall (kL)'])

    # cost breakdown
    c_prod = sum(r[3] * data['prod_cost'][r[1]] for r in prod)
    c_ph = sum(r[3] * data['plant_hub_cost'][(r[1], r[2])] for r in flows_ph)
    c_hc = sum(r[3] * data['hub_cfa_cost'][(r[1], r[2])] for r in flows_hc)
    c_un = sum(r[2] * data['sku_penalty'][r[0]] *
               (cmult if data['sku_contract'].get(r[0], False) else 1.0) for r in unmet)
    c_vs = sum(r[2] * hub_sc for r in hub_short)

    summary = pd.DataFrame({
        'Cost head': ['Production', 'Plant→Hub freight', 'Hub→CFA freight',
                      'Unmet-demand penalty', 'Hub SS shortfall penalty', 'TOTAL'],
        '₹': [c_prod, c_ph, c_hc, c_un, c_vs, c_prod + c_ph + c_hc + c_un + c_vs],
    })

    # hub ending stock after plan
    end_rows = []
    for (s, h), ss in hub_ss.items():
        opn = float(hub_open.get((s, h), 0.0))
        inn = ph_df.query('SKU==@s and Hub==@h')['Volume (kL)'].sum()
        out = hc_df.query('SKU==@s and Hub==@h')['Volume (kL)'].sum()
        end = opn + inn - out
        if opn or inn or out or ss:
            end_rows.append((s, h, opn, inn, out, end, ss, max(0.0, ss - end)))
    hub_end = pd.DataFrame(end_rows, columns=[
        'SKU', 'Hub', 'Opening (kL)', 'Inbound (kL)', 'Outbound (kL)',
        'Ending (kL)', 'SS norm (kL)', 'Shortfall (kL)'])

    net_df = pd.DataFrame([(s, c, v) for (s, c), v in net_req.items()],
                          columns=['SKU', 'CFA', 'Net requirement (kL)'])
    return dict(production=prod_df, plant_hub=ph_df, hub_cfa=hc_df, unmet=un_df,
                hub_shortfall=vs_df, cost_summary=summary, hub_ending=hub_end,
                net_req=net_df, mip_status=res.message, total_cost=summary['₹'].iloc[-1])
