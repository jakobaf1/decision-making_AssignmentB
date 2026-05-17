# -*- coding: utf-8 -*-
"""
Task 5 - Hybrid Policy
Group [number]

Rolling-horizon stochastic MILP. At each hour, sample a few price/occupancy
scenarios, solve a multi-scenario MILP over the next H steps, and apply
only the first-stage decision. Idea from Lec05 stochastic lookahead,
adapted to fit within the 15s time limit.
"""

import numpy as np
import pyomo.environ as pyo
from pyomo.environ import (
    ConcreteModel, Set, Var, Binary, NonNegativeReals,
    Constraint, Objective, minimize, value, SolverFactory
)
import data.v2_SystemCharacteristics as SC

class HybridPolicy():
    # -- scenario generation -------------------------------------------------------
    # copied from the provided process files so this script is self-contained

    def __init__(self):
        self.N_SCENARIOS = 10   # kept low to stay within time limit; re-solving every hour
        self.H_LOOKAHEAD = 5    # looking 5 steps ahead

    def _next_price(self, p_cur, p_prev, rng):
        mean_p = 4.0
        nxt = p_cur + 0.6*(p_cur - p_prev) + 0.12*(mean_p - p_cur) + rng.normal(0, 0.5)
        if nxt < 0 and rng.random() > 0.2:
            nxt = rng.uniform(0, mean_p * 0.3)
        return float(np.clip(nxt, 0.0, 12.0))


    def _next_occupancy(self, r1, r2, rng):
        r1n = r1 + 0.25*(35.0 - r1) + 0.1*(r2 - r1) + rng.normal(0, 3.0)
        r2n = r2 + 0.25*(25.0 - r2) + 0.1*(r1 - r2) + rng.normal(0, 2.5)
        return float(np.clip(r1n, 20, 50)), float(np.clip(r2n, 10, 30))


    def _generate_scenarios(self, state, horizon, n):
        # horizon = number of future steps we need (so H-1 in practice)
        # fewer than 100 scenarios but we re-solve every hour anyway
        rng = np.random.default_rng()

        price = {s: {} for s in range(n)}
        occ1  = {s: {} for s in range(n)}
        occ2  = {s: {} for s in range(n)}

        for s in range(n):
            pc, pp = state["price_t"], state["price_previous"]
            r1, r2 = state["Occ1"], state["Occ2"]
            for k in range(horizon):
                pc_new       = self._next_price(pc, pp, rng)
                r1n, r2n     = self._next_occupancy(r1, r2, rng)
                price[s][k]  = pc_new
                occ1[s][k]   = r1n
                occ2[s][k]   = r2n
                pp, pc       = pc, pc_new
                r1, r2       = r1n, r2n

        return price, occ1, occ2


    # -- stochastic MILP -----------------------------------------------------------

    def _solve(self, state, params, price_sc, occ1_sc, occ2_sc, H, N, T_out):
        """
        H-step, N-scenario MILP. Step 0 is here-and-now (shared across scenarios),
        steps 1..H-1 are scenario-specific. Returns (p1, p2, v) for step 0.
        """

        P_r      = params["heating_max_power"]
        P_vent   = params["ventilation_power"]
        z_exch   = params["heat_exchange_coeff"]
        z_loss   = params["thermal_loss_coeff"]
        z_conv   = params["heating_efficiency_coeff"]
        z_cool   = params["heat_vent_coeff"]
        z_occ    = params["heat_occupancy_coeff"]
        eta_occ  = params["humidity_occupancy_coeff"]
        eta_vent = params["humidity_vent_coeff"]
        T_low    = params["temp_min_comfort_threshold"]
        T_ok     = params["temp_OK_threshold"]
        T_high   = params["temp_max_comfort_threshold"]
        H_thr    = params["humidity_threshold"]
        U_vent   = params["vent_min_up_time"]

        T1_0  = state["T1"];  T2_0 = state["T2"];  H0 = state["H"]
        c0    = state["vent_counter"]
        ul_r1 = state["low_override_r1"];  ul_r2 = state["low_override_r2"]
        lam_0 = state["price_t"]
        k1_0  = state["Occ1"];  k2_0 = state["Occ2"]

        BIG_T = 50.0;  BIG_H = 200.0

        ROOMS = [1, 2];  SCENS = list(range(N));  STEPS = list(range(H))

        m = ConcreteModel()
        m.R = Set(initialize=ROOMS)
        m.S = Set(initialize=SCENS)
        m.T = Set(initialize=STEPS)

        m.p   = Var(m.R, m.S, m.T, within=NonNegativeReals, bounds=(0, P_r))
        m.v   = Var(m.S, m.T, within=Binary)
        m.sv  = Var(m.S, m.T, within=Binary)
        m.Tr1 = Var(m.S, m.T, within=pyo.Reals)
        m.Tr2 = Var(m.S, m.T, within=pyo.Reals)
        m.Hum = Var(m.S, m.T, within=pyo.Reals)
        m.y_low  = Var(m.R, m.S, m.T, within=Binary)
        m.y_ok   = Var(m.R, m.S, m.T, within=Binary)
        m.y_high = Var(m.R, m.S, m.T, within=Binary)
        m.u      = Var(m.R, m.S, m.T, within=Binary)

        # non-anticipativity: step-0 decision is the same for every scenario
        def na_p(m, r, s, t):
            if t == 0 and s > 0: return m.p[r, s, 0] == m.p[r, 0, 0]
            return Constraint.Skip
        m.na_p = Constraint(m.R, m.S, m.T, rule=na_p)

        def na_v(m, s, t):
            if t == 0 and s > 0: return m.v[s, 0] == m.v[0, 0]
            return Constraint.Skip
        m.na_v = Constraint(m.S, m.T, rule=na_v)

        # fix observed initial state
        def fix_T1(m, s, t):
            return (m.Tr1[s, 0] == T1_0) if t == 0 else Constraint.Skip
        def fix_T2(m, s, t):
            return (m.Tr2[s, 0] == T2_0) if t == 0 else Constraint.Skip
        def fix_H(m, s, t):
            return (m.Hum[s, 0] == H0)   if t == 0 else Constraint.Skip
        def fix_u(m, r, s, t):
            if t == 0: return m.u[r, s, 0] == (ul_r1 if r == 1 else ul_r2)
            return Constraint.Skip
        m.fix_T1 = Constraint(m.S, m.T, rule=fix_T1)
        m.fix_T2 = Constraint(m.S, m.T, rule=fix_T2)
        m.fix_H  = Constraint(m.S, m.T, rule=fix_H)
        m.fix_u  = Constraint(m.R, m.S, m.T, rule=fix_u)

        # helpers
        def kappa(r, s, t):
            if t == 0: return k1_0 if r == 1 else k2_0
            return occ1_sc[s][t-1] if r == 1 else occ2_sc[s][t-1]
        def lam(s, t):
            return lam_0 if t == 0 else price_sc[s][t-1]
        def Tout(t):
            return T_out[min(t, len(T_out)-1)]
        def Tr(m, r, s, t):
            return m.Tr1[s, t] if r == 1 else m.Tr2[s, t]

        # temperature dynamics
        def T1_dyn(m, s, t):
            if t == 0: return Constraint.Skip
            return m.Tr1[s, t] == (m.Tr1[s,t-1]
                + z_exch*(m.Tr2[s,t-1] - m.Tr1[s,t-1])
                - z_loss*(m.Tr1[s,t-1] - Tout(t-1))
                + z_conv*m.p[1,s,t-1] - z_cool*m.v[s,t-1] + z_occ*kappa(1,s,t-1))
        def T2_dyn(m, s, t):
            if t == 0: return Constraint.Skip
            return m.Tr2[s, t] == (m.Tr2[s,t-1]
                + z_exch*(m.Tr1[s,t-1] - m.Tr2[s,t-1])
                - z_loss*(m.Tr2[s,t-1] - Tout(t-1))
                + z_conv*m.p[2,s,t-1] - z_cool*m.v[s,t-1] + z_occ*kappa(2,s,t-1))
        m.T1_dyn = Constraint(m.S, m.T, rule=T1_dyn)
        m.T2_dyn = Constraint(m.S, m.T, rule=T2_dyn)

        # humidity dynamics
        def H_dyn(m, s, t):
            if t == 0: return Constraint.Skip
            return m.Hum[s, t] == (m.Hum[s,t-1]
                + eta_occ*(kappa(1,s,t-1) + kappa(2,s,t-1))
                - eta_vent*m.v[s,t-1])
        m.H_dyn = Constraint(m.S, m.T, rule=H_dyn)

        # big-M detection (same logic as Part A MILP)
        def c_ylow_a(m, r, s, t):
            return Tr(m,r,s,t) <= T_low + BIG_T*(1 - m.y_low[r,s,t])
        def c_ylow_b(m, r, s, t):
            return Tr(m,r,s,t) >= T_low - BIG_T*m.y_low[r,s,t]
        def c_yok_a(m, r, s, t):
            return Tr(m,r,s,t) >= T_ok - BIG_T*(1 - m.y_ok[r,s,t])
        def c_yok_b(m, r, s, t):
            return Tr(m,r,s,t) <= T_ok + BIG_T*m.y_ok[r,s,t]
        def c_yhigh_a(m, r, s, t):
            return Tr(m,r,s,t) >= T_high - BIG_T*(1 - m.y_high[r,s,t])
        def c_yhigh_b(m, r, s, t):
            return Tr(m,r,s,t) <= T_high + BIG_T*m.y_high[r,s,t]
        m.c_ylow_a  = Constraint(m.R, m.S, m.T, rule=c_ylow_a)
        m.c_ylow_b  = Constraint(m.R, m.S, m.T, rule=c_ylow_b)
        m.c_yok_a   = Constraint(m.R, m.S, m.T, rule=c_yok_a)
        m.c_yok_b   = Constraint(m.R, m.S, m.T, rule=c_yok_b)
        m.c_yhigh_a = Constraint(m.R, m.S, m.T, rule=c_yhigh_a)
        m.c_yhigh_b = Constraint(m.R, m.S, m.T, rule=c_yhigh_b)

        # overrule controllers
        def c_high_off(m, r, s, t):
            return m.p[r,s,t] <= P_r*(1 - m.y_high[r,s,t])
        def c_u_act(m, r, s, t):
            return m.u[r,s,t] >= m.y_low[r,s,t]
        def c_u_pl(m, r, s, t):
            if t == 0: return Constraint.Skip
            return m.u[r,s,t] >= m.u[r,s,t-1] - m.y_ok[r,s,t]
        def c_u_ph(m, r, s, t):
            if t == 0: return Constraint.Skip
            return m.u[r,s,t] <= m.u[r,s,t-1] + m.y_low[r,s,t]
        def c_u_deact(m, r, s, t):
            if t == 0: return Constraint.Skip
            return m.u[r,s,t] <= 1 - m.y_ok[r,s,t]
        def c_low_max(m, r, s, t):
            return m.p[r,s,t] >= P_r*m.u[r,s,t]
        m.c_high_off = Constraint(m.R, m.S, m.T, rule=c_high_off)
        m.c_u_act    = Constraint(m.R, m.S, m.T, rule=c_u_act)
        m.c_u_pl     = Constraint(m.R, m.S, m.T, rule=c_u_pl)
        m.c_u_ph     = Constraint(m.R, m.S, m.T, rule=c_u_ph)
        m.c_u_deact  = Constraint(m.R, m.S, m.T, rule=c_u_deact)
        m.c_low_max  = Constraint(m.R, m.S, m.T, rule=c_low_max)

        # ventilation startup + min up-time
        v_prev = 1 if c0 > 0 else 0
        def c_sv_lo(m, s, t):
            prev = v_prev if t == 0 else m.v[s, t-1]
            return m.sv[s,t] >= m.v[s,t] - prev
        def c_sv_a(m, s, t):
            return m.sv[s,t] <= m.v[s,t]
        def c_sv_b(m, s, t):
            prev = v_prev if t == 0 else m.v[s, t-1]
            return m.sv[s,t] <= 1 - prev
        def c_uptime(m, s, t):
            end = min(t + U_vent - 1, H - 1)
            if end <= t: return Constraint.Skip
            return sum(m.v[s,tau] for tau in range(t, end+1)) >= (end-t+1)*m.sv[s,t]
        m.c_sv_lo  = Constraint(m.S, m.T, rule=c_sv_lo)
        m.c_sv_a   = Constraint(m.S, m.T, rule=c_sv_a)
        m.c_sv_b   = Constraint(m.S, m.T, rule=c_sv_b)
        m.c_uptime = Constraint(m.S, m.T, rule=c_uptime)

        # if vent was already running, keep it on for the remaining locked hours
        hours_left = max(0, U_vent - c0) if c0 > 0 else 0
        def c_inertia(m, s, t):
            if t < hours_left: return m.v[s,t] == 1
            return Constraint.Skip
        m.c_inertia = Constraint(m.S, m.T, rule=c_inertia)

        # humidity overrule
        def c_hum_vent(m, s, t):
            return m.Hum[s,t] <= H_thr + BIG_H*m.v[s,t]
        m.c_hum_vent = Constraint(m.S, m.T, rule=c_hum_vent)

        # objective: expected cost over scenarios
        def obj_rule(m):
            return (1.0/N) * sum(
                lam(s,t) * (m.p[1,s,t] + m.p[2,s,t] + P_vent*m.v[s,t])
                for s in SCENS for t in STEPS
            )
        m.obj = Objective(rule=obj_rule, sense=minimize)

        solver = SolverFactory("gurobi")
        solver.options["TimeLimit"]  = 10
        solver.options["MIPGap"]     = 0.01
        solver.options["OutputFlag"] = 0

        res = solver.solve(m, tee=False)

        ok = [pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
            pyo.TerminationCondition.maxTimeLimit]
        if res.solver.termination_condition not in ok:
            return None

        return float(value(m.p[1,0,0])), float(value(m.p[2,0,0])), int(float(value(m.v[0,0])) > 0.5)


    # -- policy entry point --------------------------------------------------------


    def select_action(self, state):
        params  = SC.get_fixed_data()
        t_now   = int(state.get("current_time", 0))
        T_total = int(params["num_timeslots"])
        T_out   = params["outdoor_temperature"]

        H = max(min(self.H_LOOKAHEAD, T_total - t_now), 1)
        T_out_window = [T_out[min(t_now + k, T_total - 1)] for k in range(H)]

        price_sc, occ1_sc, occ2_sc = self._generate_scenarios(
            state, horizon=max(H-1, 1), n=self.N_SCENARIOS
        )

        result =self. _solve(state, params, price_sc, occ1_sc, occ2_sc,
                        H=H, N=self.N_SCENARIOS, T_out=T_out_window)

        if result is not None:
            p1, p2, v = result
        else:
            # fallback if solver crashes - shouldn't really happen
            P_fb = params["heating_max_power"]
            p1 = P_fb if state["T1"] < 20.0 else 0.0
            p2 = P_fb if state["T2"] < 20.0 else 0.0
            v  = 1 if state["H"] > 65.0 else 0

        return {
            "HeatPowerRoom1": float(p1),
            "HeatPowerRoom2": float(p2),
            "VentilationON":  int(v)
        }
