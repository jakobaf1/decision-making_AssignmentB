import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.v2_SystemCharacteristics import get_fixed_data

# ADP POLICY - Task 4
# Sample-based Backward Induction, Forward-Backward Algorithm

data = get_fixed_data()

# System parameters
zeta_exch = data["heat_exchange_coeff"]
zeta_loss = data["thermal_loss_coeff"]
zeta_conv = data["heating_efficiency_coeff"]
zeta_cool = data["heat_vent_coeff"]
zeta_occ  = data["heat_occupancy_coeff"]
eta_occ   = data["humidity_occupancy_coeff"]
eta_vent  = data["humidity_vent_coeff"]

T_low     = data["temp_min_comfort_threshold"]
T_OK      = data["temp_OK_threshold"]
T_high    = data["temp_max_comfort_threshold"]
H_high    = data["humidity_threshold"]
P_max     = data["heating_max_power"]
P_vent    = data["ventilation_power"]
T_out     = data["outdoor_temperature"]
num_slots = data["num_timeslots"]
tol       = 1e-4

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")

P_LEVELS  = [0.0, P_max/4, P_max/2, 3*P_max/4, P_max]


# Chosen features 
def phi(state):
    return np.array([
        1.0,
        state["T1"],
        state["T2"],
        state["H"],
        state["price_t"],
        float(state["vent_counter"]),
        float(state["low_override_r1"]),
        float(state["low_override_r2"]),
    ])

N_FEATURES = 8


# Overrule controls
def apply_overrules(state, p1, p2, v):
    if state["H"] > H_high:
        v = 1
    elif state["vent_counter"] in [1, 2]:
        v = 1
    if state["T1"] > T_high:
        p1 = 0.0
    elif state["low_override_r1"] == 1:
        p1 = P_max
    if state["T2"] > T_high:
        p2 = 0.0
    elif state["low_override_r2"] == 1:
        p2 = P_max
    return p1, p2, v


# Transition dynamics
def transition(state, p1, p2, v, t, next_occ1, next_occ2,
               next_price, next_price_prev):
    new_T1 = (state["T1"]
              + zeta_exch * (state["T2"] - state["T1"])
              + zeta_loss * (T_out[t]    - state["T1"])
              + zeta_conv * p1
              - zeta_cool * v
              + zeta_occ  * state["Occ1"])

    new_T2 = (state["T2"]
              + zeta_exch * (state["T1"] - state["T2"])
              + zeta_loss * (T_out[t]    - state["T2"])
              + zeta_conv * p2
              - zeta_cool * v
              + zeta_occ  * state["Occ2"])

    new_H = (state["H"]
             + eta_occ * (state["Occ1"] + state["Occ2"])
             - eta_vent * v)

    y1 = 0
    if new_T1 < T_low - tol:
        y1 = 1
    elif state["low_override_r1"] == 1 and new_T1 < T_OK - tol:
        y1 = 1

    y2 = 0
    if new_T2 < T_low - tol:
        y2 = 1
    elif state["low_override_r2"] == 1 and new_T2 < T_OK - tol:
        y2 = 1

    return {
        "T1":              new_T1,
        "T2":              new_T2,
        "H":               new_H,
        "Occ1":            next_occ1,
        "Occ2":            next_occ2,
        "price_t":         next_price,
        "price_previous":  next_price_prev,
        "vent_counter":    state["vent_counter"] + 1 if v == 1 else 0,
        "low_override_r1": y1,
        "low_override_r2": y2,
        "current_time":    t + 1,
    }


def immediate_cost(state, p1, p2, v):
    return state["price_t"] * (p1 + p2 + P_vent * v)


# Greedy action selection
def greedy_action(state, t, theta_next, prices_day, occ1_day, occ2_day):
    best_cost = np.inf
    best = {"HeatPowerRoom1": 0.0, "HeatPowerRoom2": 0.0, "VentilationON": 0}

    if t + 1 < num_slots:
        next_occ1       = occ1_day[t + 1]
        next_occ2       = occ2_day[t + 1]
        next_price      = prices_day[t + 2] if t + 2 < len(prices_day) else prices_day[-1]
        next_price_prev = prices_day[t + 1]
    else:
        next_occ1 = next_occ2 = 0.0
        next_price = next_price_prev = 0.0

    for p1_raw in P_LEVELS:
        for p2_raw in P_LEVELS:
            for v_raw in [0, 1]:
                p1, p2, v = apply_overrules(state, p1_raw, p2_raw, v_raw)
                cost = immediate_cost(state, p1, p2, v)

                if theta_next is not None and t + 1 < num_slots:
                    ns = transition(state, p1, p2, v, t,
                                    next_occ1, next_occ2,
                                    next_price, next_price_prev)
                    cost += theta_next @ phi(ns)

                if cost < best_cost:
                    best_cost = cost
                    best = {
                        "HeatPowerRoom1": p1_raw,
                        "HeatPowerRoom2": p2_raw,
                        "VentilationON":  v_raw,
                    }
    return best


# Training of algorithm
def train_adp(num_iterations=300):
    prices_all = np.genfromtxt(os.path.join(DATA_DIR, "v2_PriceData.csv"),
                               delimiter=",", skip_header=1)
    occ1_all   = np.genfromtxt(os.path.join(DATA_DIR, "OccupancyRoom1.csv"),
                               delimiter=",", skip_header=1)
    occ2_all   = np.genfromtxt(os.path.join(DATA_DIR, "OccupancyRoom2.csv"),
                               delimiter=",", skip_header=1)

    num_days = prices_all.shape[0]
    thetas   = [np.zeros(N_FEATURES) for _ in range(num_slots)]
    Phi_acc  = [[] for _ in range(num_slots)]
    Y_acc    = [[] for _ in range(num_slots)]

    for i in range(num_iterations):
        day        = i % num_days
        prices_day = prices_all[day]
        occ1_day   = occ1_all[day]
        occ2_day   = occ2_all[day]

        # forward pass
        state = {
            "T1":              data["T1"],
            "T2":              data["T2"],
            "H":               data["H"],
            "Occ1":            occ1_day[0],
            "Occ2":            occ2_day[0],
            "price_t":         prices_day[1],
            "price_previous":  prices_day[0],
            "vent_counter":    data["vent_counter"],
            "low_override_r1": data["low_override_r1"],
            "low_override_r2": data["low_override_r2"],
            "current_time":    0,
        }

        visited_states = []
        for t in range(num_slots):
            visited_states.append(dict(state))
            theta_next = thetas[t + 1] if t + 1 < num_slots else None
            action = greedy_action(state, t, theta_next,
                                   prices_day, occ1_day, occ2_day)
            p1, p2, v = apply_overrules(state, action["HeatPowerRoom1"],
                                              action["HeatPowerRoom2"],
                                              action["VentilationON"])
            if t < num_slots - 1:
                next_price      = prices_day[t + 2] if t + 2 < len(prices_day) else prices_day[-1]
                next_price_prev = prices_day[t + 1]
                state = transition(state, p1, p2, v, t,
                                   occ1_day[t + 1], occ2_day[t + 1],
                                   next_price, next_price_prev)

        # backward pass
        for t in reversed(range(num_slots)):
            s_t        = visited_states[t]
            theta_next = thetas[t + 1] if t + 1 < num_slots else None
            best_action = greedy_action(s_t, t, theta_next,
                                        prices_day, occ1_day, occ2_day)
            p1, p2, v = apply_overrules(s_t, best_action["HeatPowerRoom1"],
                                             best_action["HeatPowerRoom2"],
                                             best_action["VentilationON"])
            cost_t = immediate_cost(s_t, p1, p2, v)

            if t + 1 < num_slots:
                next_price      = prices_day[t + 2] if t + 2 < len(prices_day) else prices_day[-1]
                next_price_prev = prices_day[t + 1]
                ns = transition(s_t, p1, p2, v, t,
                                occ1_day[t + 1], occ2_day[t + 1],
                                next_price, next_price_prev)
                target = cost_t + thetas[t + 1] @ phi(ns)
            else:
                target = cost_t

            Phi_acc[t].append(phi(s_t))
            Y_acc[t].append(target)

            Phi_mat = np.array(Phi_acc[t])
            Y_vec   = np.array(Y_acc[t])
            thetas[t], _, _, _ = np.linalg.lstsq(Phi_mat, Y_vec, rcond=None)

    print("ADP training complete.")
    return thetas



print("Training ADP value function approximation...")
THETAS = train_adp(num_iterations=300)
print("Training done. Thetas fitted for all timeslots.")


# policy function to be used in environment
def select_action(state):
    """
    Entry point for the environment. Returns greedy action
    under the trained linear VFA.
    """
    t = state["current_time"]

    proxy_prices    = np.full(num_slots + 2, state["price_t"])
    proxy_prices[0] = state["price_previous"]
    proxy_prices[1] = state["price_t"]

    proxy_occ1 = np.full(num_slots, state["Occ1"])
    proxy_occ2 = np.full(num_slots, state["Occ2"])

    theta_next = THETAS[t + 1] if t + 1 < num_slots else None
    return greedy_action(state, t, theta_next,
                         proxy_prices, proxy_occ1, proxy_occ2)
