from pyomo.environ import *
import numpy as np
from sklearn.cluster import KMeans
from data.v2_SystemCharacteristics import get_fixed_data

data = get_fixed_data()

# Inlined process models (avoids module-level side effects in the originals)

def _next_occupancy(r1, r2):
    r1_next = r1 + 0.25*(35.0 - r1) + 0.1*(r2 - r1) + np.random.normal(0, 3.0)
    r2_next = r2 + 0.25*(25.0 - r2) + 0.1*(r1 - r2) + np.random.normal(0, 2.5)
    return float(np.clip(r1_next, 20, 50)), float(np.clip(r2_next, 10, 30))

def _next_price(price, prev_price):
    noise = np.random.normal(0, 0.5)
    next_p = price + 0.6*(price - prev_price) + 0.12*(4.0 - price) + noise
    if next_p < 0 and np.random.rand() > 0.2:
        next_p = np.random.uniform(0, 1.2)
    return float(max(min(next_p, 12.0), 0.0))

# Scenario tree parameters
N_BRANCHES    = 3    # number of cluster centroids kept per parent
MAX_LOOKAHEAD = 3
N_SAMPLES     = 30   # candidate children drawn per parent before clustering

def _build_tree(state):
    """
    Build a scenario tree using iterative branch-and-cluster.

    At each stage we expand every frontier node by:
      1. Drawing N_SAMPLES candidate children from the stochastic process models.
      2. Clustering those candidates into N_BRANCHES groups with KMeans.
      3. Replacing each group with its centroid — one child node per cluster.
      4. Setting child probability = parent_prob * (cluster_size / N_SAMPLES).
    The centroids become the new frontier for the next stage.

    Stage 0 is the root (current observed state, probability 1).
    Stages 1..L are the lookahead horizon.
    Node structure is unchanged: {idx, stage, parent, prob, occ1, occ2, price, price_prev}.
    """
    t0 = state["current_time"]
    L = min(data["num_timeslots"] - t0 - 1, MAX_LOOKAHEAD)

    # --- Root node: current observed state, probability 1 ---
    nodes = [{
        "idx": 0, "stage": 0, "parent": None, "prob": 1.0,
        "occ1": state["Occ1"], "occ2": state["Occ2"],
        "price": state["price_t"], "price_prev": state["price_previous"]
    }]

    frontier = [0]

    for _ in range(L):
        nxt = []
        for pid in frontier:
            p = nodes[pid]

            # Step 1 — draw N_SAMPLES candidate children for this parent.
            # Each sample is [occ1, occ2, price]; price_prev is always p["price"].
            raw = np.empty((N_SAMPLES, 3))
            for i in range(N_SAMPLES):
                o1, o2 = _next_occupancy(p["occ1"], p["occ2"])
                pr      = _next_price(p["price"], p["price_prev"])
                raw[i]  = [o1, o2, pr]

            # Step 2 — cluster the N_SAMPLES candidates into N_BRANCHES groups.
            # n_init=3 is sufficient for 30 points / 3 clusters.
            km      = KMeans(n_clusters=N_BRANCHES, n_init=3, random_state=None)
            labels  = km.fit_predict(raw)          # shape (N_SAMPLES,)
            centers = km.cluster_centers_          # shape (N_BRANCHES, 3)

            # Step 3 & 4 — one child node per centroid, weighted by cluster size.
            for k in range(N_BRANCHES):
                cluster_size = int(np.sum(labels == k))
                child_prob   = p["prob"] * (cluster_size / N_SAMPLES)
                o1c, o2c, prc = centers[k]
                nodes.append({
                    "idx":        len(nodes),
                    "stage":      p["stage"] + 1,
                    "parent":     pid,
                    "prob":       child_prob,
                    "occ1":       float(o1c),
                    "occ2":       float(o2c),
                    "price":      float(prc),
                    "price_prev": p["price"],
                })
                nxt.append(nodes[-1]["idx"])

        frontier = nxt

    return nodes

class SPPolicy:
    def select_action(self, state):
        return _select_action(state)


def _select_action(state):
    nodes = _build_tree(state)
    nm = {n["idx"]: n for n in nodes}
    ids = [n["idx"] for n in nodes]
    R = [0, 1]

    t0_abs      = state["current_time"]
    T_out       = data["outdoor_temperature"]
    BIG_M_T     = 100.0                                    # big-M for temperature indicator constraints
    BIG_M_H     = 300.0                                    # big-M for humidity indicator constraint
    T_high      = data["temp_max_comfort_threshold"]       # actual threshold for yh detection
    T_low       = data["temp_min_comfort_threshold"]       # actual threshold for yl detection
    T_ok        = data["temp_OK_threshold"]                # actual threshold for yo detection
    H_threshold = data["humidity_threshold"]               # actual threshold for humidity overrule
    Pbar        = data["heating_max_power"]
    U      = data["vent_min_up_time"]             # = 3 hours
    vc     = state["vent_counter"]
    v_prev_root = 1 if vc > 0 else 0             # was ventilation on before current hour?

    mdl = ConcreteModel()
    mdl.N = Set(initialize=ids)
    mdl.R = Set(initialize=R)

    # Exogenous parameters from scenario tree
    mdl.lam  = Param(mdl.N, initialize={n["idx"]: n["price"] for n in nodes})
    mdl.kap  = Param(mdl.R, mdl.N, initialize={
        (r, n["idx"]): (n["occ1"] if r == 0 else n["occ2"])
        for n in nodes for r in R
    })
    mdl.prob = Param(mdl.N, initialize={n["idx"]: n["prob"] for n in nodes})

    # Decision variables
    mdl.p  = Var(mdl.R, mdl.N, domain=NonNegativeReals)   # heating power
    mdl.Ti = Var(mdl.R, mdl.N, domain=Reals)              # indoor temperature
    mdl.H  = Var(mdl.N, domain=NonNegativeReals)           # humidity
    mdl.v  = Var(mdl.N, domain=Binary)                    # ventilation on/off
    mdl.s  = Var(mdl.N, domain=Binary)                    # ventilation startup
    mdl.u  = Var(mdl.R, mdl.N, domain=Binary)             # low-temp override active
    mdl.yl = Var(mdl.R, mdl.N, domain=Binary)             # temp below T_low
    mdl.yo = Var(mdl.R, mdl.N, domain=Binary)             # temp above T_ok
    mdl.yh = Var(mdl.R, mdl.N, domain=Binary)             # temp above T_high

    c = ConstraintList()
    mdl.c = c

    # ---- Root state initialization (current known state) ----
    c.add(mdl.Ti[0, 0] == state["T1"])
    c.add(mdl.Ti[1, 0] == state["T2"])
    c.add(mdl.H[0]     == state["H"])
    c.add(mdl.u[0, 0]  == state["low_override_r1"])
    c.add(mdl.u[1, 0]  == state["low_override_r2"])

    for n in nodes:
        nid   = n["idx"]
        pid   = n["parent"]
        stage = n["stage"]

        # ---- State dynamics (non-root: child state comes from parent actions) ----
        if pid is not None:
            p_stage = nm[pid]["stage"]
            tout = T_out[min(t0_abs + p_stage, len(T_out) - 1)]
            for r in R:
                c.add(
                    mdl.Ti[r, nid] ==
                    mdl.Ti[r, pid]
                    + data["heat_exchange_coeff"]  * (mdl.Ti[1-r, pid] - mdl.Ti[r, pid])
                    - data["thermal_loss_coeff"]   * (mdl.Ti[r, pid]   - tout)
                    + data["heating_efficiency_coeff"] * mdl.p[r, pid]
                    - data["heat_vent_coeff"]      * mdl.v[pid]
                    + data["heat_occupancy_coeff"] * mdl.kap[r, pid]
                )
            c.add(
                mdl.H[nid] ==
                mdl.H[pid]
                + data["humidity_occupancy_coeff"] * sum(mdl.kap[r, pid] for r in R)
                - data["humidity_vent_coeff"]      * mdl.v[pid]
            )

        # ---- Heating power bounds ----
        for r in R:
            c.add(mdl.p[r, nid] <= Pbar)

        # ---- Temperature binary detection ----
        for r in R:
            # Detect T >= T_high (overheating): force heater off
            c.add(mdl.Ti[r, nid] >= T_high - BIG_M_T*(1 - mdl.yh[r, nid]))
            c.add(mdl.Ti[r, nid] <= T_high + BIG_M_T*mdl.yh[r, nid])
            c.add(mdl.p[r, nid]  <= Pbar*(1 - mdl.yh[r, nid]))

            # Detect T <= T_low (too cold): activate low-temp override
            c.add(mdl.Ti[r, nid] <= T_low + BIG_M_T*(1 - mdl.yl[r, nid]))
            c.add(mdl.Ti[r, nid] >= T_low - BIG_M_T*mdl.yl[r, nid])

            # Detect T >= T_ok: deactivate low-temp override
            c.add(mdl.Ti[r, nid] >= T_ok - BIG_M_T*(1 - mdl.yo[r, nid]))
            c.add(mdl.Ti[r, nid] <= T_ok + BIG_M_T*mdl.yo[r, nid])

        # ---- Low-temperature override controller (non-root) ----
        # At root, u is fixed by initialization above.
        if pid is not None:
            for r in R:
                c.add(mdl.u[r, nid] >= mdl.yl[r, nid])                          # activate when cold
                c.add(mdl.u[r, nid] <= mdl.u[r, pid] + mdl.yl[r, nid])          # only active if was active or just triggered
                c.add(mdl.u[r, nid] >= mdl.u[r, pid] - mdl.yo[r, nid])          # stay active unless T_ok reached
                c.add(mdl.u[r, nid] <= 1 - mdl.yo[r, nid])                      # deactivate once T_ok reached

        # ---- Override forces heater to max ----
        for r in R:
            c.add(mdl.p[r, nid] >= Pbar * mdl.u[r, nid])

        # ---- Ventilation startup indicator ----
        v_before = v_prev_root if pid is None else mdl.v[pid]
        c.add(mdl.s[nid] >= mdl.v[nid] - v_before)
        c.add(mdl.s[nid] <= mdl.v[nid])
        c.add(mdl.s[nid] <= 1 - v_before)

        # ---- Forced ventilation from inherited vent_counter ----
        # vc=2 → forced on at stage 0 only; vc=1 → forced on at stage 0 and 1
        if stage == 0 and vc in [1, 2]:
            c.add(mdl.v[nid] == 1)
        if stage == 1 and vc == 1:
            c.add(mdl.v[nid] == 1)

        # ---- Humidity overrule: force ventilation if H > H_high ----
        c.add(mdl.H[nid] <= H_threshold + BIG_M_H * mdl.v[nid])

        # ---- Minimum ventilation up-time (3 hours) ----
        # For each node, if any ancestor within the last U-1 steps had a startup,
        # ventilation must be on here (backward non-anticipativity formulation).
        curr = n
        for _ in range(U - 1):
            if curr["parent"] is None:
                break
            anc = nm[curr["parent"]]
            c.add(mdl.v[nid] >= mdl.s[anc["idx"]])
            curr = anc

    # ---- Objective: minimise expected electricity cost over the tree ----
    mdl.obj = Objective(
        expr=sum(
            mdl.prob[nid] * mdl.lam[nid] * (
                data["ventilation_power"] * mdl.v[nid]
                + sum(mdl.p[r, nid] for r in R)
            )
            for nid in ids
        ),
        sense=minimize
    )

    solver = SolverFactory("gurobi")
    solver.options["TimeLimit"] = 5
    solver.solve(mdl, tee=False)

    try:
        p1 = float(value(mdl.p[0, 0]))
        p2 = float(value(mdl.p[1, 0]))
        v0 = int(round(float(value(mdl.v[0]))))
    except Exception:
        # Fallback to conservative heuristic if solver fails
        v0 = 1 if (state["H"] > H_threshold or vc in [1, 2]) else 0
        p1 = Pbar if state["low_override_r1"] == 1 else 0.0
        p2 = Pbar if state["low_override_r2"] == 1 else 0.0

    return {"HeatPowerRoom1": p1, "HeatPowerRoom2": p2, "VentilationON": v0}
