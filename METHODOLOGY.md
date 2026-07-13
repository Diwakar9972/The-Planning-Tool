# Levisol "Balancing Act" — Methodology & Assumptions

## Design philosophy: a deterministic planning engine

The entire solution is deterministic — the same class of rule-based replenishment logic quick-commerce operators run at dark-store level, coupled with an exact optimization. There is no simulation, no machine learning, and no randomness: the same inputs always produce the same norms and the same plan. Every number in the output is traceable to a closed-form formula or to an optimal solution of a stated cost-minimization model. This was a deliberate choice for three reasons: the planning team must re-run the tool monthly with changed inputs and trust that differences in output come from differences in input; every recommendation must be defensible line-by-line in front of a CSCO; and the assessment day includes a modified input set, which a deterministic engine handles by construction.

## Component 1 — Inventory norms

**Demand.** Average daily demand per SKU × CFA = mean of the six months of actual sales (Jul–Dec 2025) ÷ 30 working days, per the case instruction.

**Demand variability = forecast error, not raw demand.** The plan is built on the January forecast, so the risk the buffer must absorb is the error of that forecast, not the volatility of demand itself. We compute the standard deviation of monthly forecast error (Actual − Forecast, Exhibits G and H), and convert it to a daily figure by dividing by √30 (independent days assumption). A SKU that is volatile but well-forecast needs less buffer than a stable SKU with a biased forecast — using raw demand σ would get this backwards.

**Lead time and its variability.** Total replenishment lead time per SKU × CFA = production lead time + plant→hub transit + hub→CFA transit (Exhibit E). Lead-time variability combines the two independent sources given: σ_L = √(production variability² + transit variability²).

**Safety stock (King's formula).** SS = z × √(L·σ_d² + d²·σ_L²). This is the standard combined-uncertainty formula: the first term buffers demand risk over the lead time, the second buffers the demand exposed by lead-time risk. z is set from the tier fill-rate targets in Exhibit F: Tier A 98% → z = 2.054, Tier B 97% → z = 1.881, Tiers C and D 92% → z = 1.405.

**Tiering.** Exhibit D does not assign tiers, so we derive them from Exhibit F's volume slabs: SKUs are ranked by six-month national volume; those making up the first 50% of cumulative volume are Tier A (15 SKUs), the next 30% Tier B (34), the next 15% Tier C (33), and the last 5% Tier D (18).

**Reorder point and days of cover.** ROP = d×L + SS; DOC = ROP ÷ d. Both reported per SKU × CFA.

**Hub norms.** Each CFA is mapped to the hub that historically serves it (Exhibit E sourcing: "East" → Mother Hub East, "Rest of India" → Mother Hub West). Hub demand per SKU is the sum of its CFAs' daily demand; hub demand variance is the sum of CFA variances (the assignment states CFA demands are independent, so variances pool — this risk-pooling is exactly why hubs need proportionally less buffer than the sum of CFA buffers). Hub lead time = production lead time + plant→hub transit; its variability is the production variability. Hub service level z = 2.054 (98%, as mandated).

## Component 2 — Production & distribution plan (exact MILP)

**Objective.** Minimize total January operating cost in ₹: production cost + plant→hub freight + hub→CFA freight + penalty on unmet CFA demand + penalty on hub safety-stock shortfall. This is the OpEx-minimization mandate stated by the CSCO, with service protected through the penalty structure rather than hard rationing rules.

**Decisions.** For each SKU: how many 25 kL batches to produce at each plant; how to split each plant's output between the two hubs; how much of each SKU each hub dispatches to each CFA; and — where scarcity or batch granularity bites — how much requirement to leave unmet.

**Demand definition.** CFA net requirement = January forecast + CFA safety-stock target − CFA opening inventory, floored at zero. The default stock target is the CFA safety stock (the norm from Component 1), so the network ends January positioned to absorb February's variability; the tool can switch this to full ROP or to none.

**Constraints, in business language.**
1. Everything produced leaves the plant for a hub — no stock strands at plants.
2. Production on each pack-size line at each plant cannot exceed that line's monthly capacity (Exhibit A), and every SKU-plant quantity is an integer multiple of 25 kL.
3. A hub can only dispatch what it opened the month with plus what arrives from plants.
4. After dispatch, each hub retains its safety-stock norm; any shortfall is measured and priced, not hidden.
5. Any plant may supply any hub, and any hub any CFA — routing is decided purely on cost.
6. Contractual SKUs (18 of 100) carry a 10× multiplier on their penalty cost, making them effectively last in line for under-supply. The multiplier is a visible, editable lever rather than a hard constraint, so the tool degrades gracefully instead of turning infeasible.

**Infeasibility handling.** Unmet demand is a priced decision variable, so the model can never crash on shortage: if capacity cannot cover demand, the optimizer sheds the cheapest-to-miss volume (low penalty, non-contractual, low tier) and the output states exactly what is unmet, where, by how much, and at what penalty cost.

**Solver.** HiGHS branch-and-bound via scipy.optimize.milp — an exact MILP solver, no heuristics. ~3,900 variables (≈300 integer), solved to a good incumbent within a 60–90 second planner-friendly time limit. The reported LP "gap" overstates true suboptimality because the LP relaxation is allowed fractional batches, which no feasible plan can use.

## Key findings on the January 2026 data

Total plan cost ≈ ₹14.5 crore: production ₹10.3 Cr, plant→hub freight ₹1.37 Cr, hub→CFA freight ₹2.27 Cr, and ≈₹0.55 Cr of priced hub safety-stock shortfall.

Kolkata (₹9,000/kL, cheapest) runs at 100% on every line it has; Mumbai runs near-full; Ahmedabad (most expensive) is the swing plant. Sourcing follows the natural geography — Kolkata feeds MHE, Mumbai feeds MHW — with Ahmedabad topping up both.

Capacity is not structurally binding in January: every pack-size line has headroom even after 25 kL batch rounding. Essentially all CFA demand is served. The residual hub safety-stock shortfall (~100 kL spread thinly across ~30 SKUs) is deliberate economics, not failure: topping up a 3 kL gap requires producing a whole 25 kL batch costing ₹3–4.5 lakh, which exceeds the value of the buffer for those SKUs. The tool prices this trade-off explicitly and a planner can change the shortfall cost to force the buffers full.

## Assumptions register

1. 30 working days per month (per case instruction); monthly σ scales to daily by √30 (independent days).
2. Forecast error, not raw demand, is the correct variability measure for buffers, because plans are built on forecasts.
3. CFA demands are independent (stated in the assignment), so variances pool additively at hubs.
4. Tier assignment by cumulative volume share per Exhibit F slabs (50/30/15/5).
5. Historical hub sourcing (Exhibit E) defines the norm-setting network; for the January plan any plant may serve any hub (per Exhibit C note and Component 1 instruction on least-cost sourcing).
6. Contractual protection = 10× penalty multiplier (editable) rather than a hard constraint, to preserve graceful degradation.
7. Hub safety-stock shortfall priced at ₹50,000/kL (editable). One-month planning horizon; ending stock above safety norms carries no credit (conservative).
8. Where the PDF and the data file disagree, the data file governs (per case instruction).

## Limitations, stated honestly

The model is single-period: it does not carry February demand, so end-of-month stock above norms has no modeled value, which slightly biases against overproduction. Forecast-error σ is estimated from only six observations per SKU-CFA, so norms for slow movers are noisy; a rolling window will firm these up as months accrue. The z-values assume approximately normal forecast errors; for highly intermittent D-tier SKUs a service-time policy may be more appropriate. Batch integrality makes the MILP's proven optimality gap look larger than true suboptimality; the incumbent plans are stable and near-optimal in structure.

---

## Extension — Graph-based network model with 1000-experiment simulation

**Graph.** The network is modeled as a directed graph: source nodes (3 plants, each carrying per-line capacity and production cost), hub nodes (MHW, MHE, with conditional processing), destination nodes (10 CFAs with scenario demand), plant→hub and hub→CFA edges carrying freight ₹/kL, transit days, and an emissions proxy.

**Composite multi-criteria objective.** Each edge carries a composite weight combining four normalized criteria: cost 40% (production + freight ₹), time 30% (transit days; hub→CFA from Exhibit E medians, plant→hub proxied from freight since the data shows a flat 1 day), capacity utilization 20% (a congestion penalty on production above an 85% utilization knee, spreading load away from maxed lines), and environmental impact 10% (CO₂ ∝ tonne-km ∝ freight ₹ as a distance proxy). Criteria are normalized by baseline magnitudes so the weights act on commensurate terms, and the whole objective stays linear — every scenario is solved exactly as an LP (HiGHS), not heuristically.

**Conditional elements.** Conditional hubs: throughput above a threshold (80% of expected inflow) incurs an overflow processing cost, modeling processing capacity that degrades with load. Conditional links: in any scenario, each hub→CFA link has a 10% probability of disruption (weather/traffic), capping its capacity at 60% of nominal and stretching transit 1.5×.

**Simulation.** 1000 scenarios generated with a fixed seed (fully reproducible): demand per CFA×slab ~ Normal(Jan forecast, historical CV from six months of sales), plant availability ~ Uniform(0.85–1.0), plus the random link disruptions. All 1000 LPs solve in ~5 seconds on CPU. SKUs are aggregated to the five production-line slabs at this layer; the SKU-level MILP remains the detailed monthly planner.

**Findings.** The recommended plan (optimum of the expected scenario) costs ₹11.4 Cr with 100% fill and 3.5-day average transit. Across all 1000 experiments: fill rate ≥ 99.4% in 95% of scenarios; three lanes are structurally dominant (BOM→MHW, KOL→MHE, AHM→MHW in 100% of scenarios, AHM→MHE in 99.6%), while BOM→MHE and KOL→MHW never open — the network's geography is robust, not fragile. The multi-criteria objective shifts load off Kolkata (85% vs 100% under pure cost) onto Ahmedabad, trading ~₹0.9 Cr of cost for lower congestion risk and faster average service — the price of the 20% utilization weighting, made explicit. Kolkata utilization (median 88%, P95 92%) is the network's watch item.
