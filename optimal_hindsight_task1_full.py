from pyomo.environ import *
import csv
import matplotlib.pyplot as plt
import numpy as np
from data.v2_SystemCharacteristics import get_fixed_data
data = get_fixed_data()

# takes the two occupancy filenames and generates a dictionary with the relevant data
def generate_occupancy_dict(file_room1="data/OccupancyRoom1.csv",
                            file_room2="data/OccupancyRoom2.csv"):
    """
    Reads occupancy CSV files and returns a dictionary
    indexed by (room, day, hour) for Pyomo Param initialization.
    """

    occupancy = {}

    files = {
        0: file_room1,
        1: file_room2
    }

    for room, filename in files.items():
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            
            # Skip header row (0–9)
            next(reader)

            for day_index, row in enumerate(reader):
                for hour_index, value in enumerate(row):
                    occupancy[(room, day_index, hour_index)] = float(value)

    return occupancy


# takes a filename and generates a dictionary with the given data
def generate_price_dict(filename="data/PriceData.csv"):
    """
    Reads a CSV file and returns a dictionary
    indexed by (day, hour).

    Parameters:
        filename (str): Path to CSV file
        
    Returns:
        dict: {(day, hour): value}
    """
    
    prize = {}

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        
        # Read header to determine number of hours
        header = next(reader)
        num_hours = len(header)

        for day_index, row in enumerate(reader):
            for hour_index in range(0,num_hours):
                prize[(day_index, hour_index)] = float(row[hour_index])

    return prize

def build_model_for_day(day, occ_data, price_data, look_horizon):
    #### Pyomo model ####
    model = ConcreteModel()

    # Sets
    time_slots = list(range(0,data['num_timeslots']))
    model.T = Set(initialize=time_slots, ordered=True)

    rooms = list(range(0,2))
    model.R = Set(initialize=rooms, ordered=True)


    #### parameters (in order from Solution_to_assignment_partA_2026-pdf) ###

    model.lambda_tilde = Param(model.T,
                        initialize={t: price_data[(day, t)]
                                    for t in model.T})
    # find data for the relevant day
    occ_day = {
        (r, t): occ_data[(r, day, t)]
        for (r, d, t) in occ_data
        if d == day
    }

    model.kappa = Param(model.R, model.T, initialize=occ_day)
    model.L = Param(initialize=look_horizon)
    model.T_out = Param(model.T, initialize=data["outdoor_temperature"])
    model.P_vent = Param(initialize=data["ventilation_power"])
    model.P_bar = Param(model.R, initialize=data['heating_max_power'])
    model.T_low = Param(initialize=data["temp_min_comfort_threshold"])
    model.T_high = Param(initialize=data["temp_max_comfort_threshold"])
    model.T_ok = Param(initialize=data["temp_OK_threshold"])
    model.H_high = Param(initialize=data["humidity_threshold"])
    model.M_temp = Param(initialize=data["temp_max_comfort_threshold"])
    model.M_hum = Param(initialize=data["humidity_threshold"])
    model.U_vent = Param(initialize=data["vent_min_up_time"])

    model.xi_exch = Param(initialize=data["heat_exchange_coeff"])
    model.xi_loss = Param(initialize=data["thermal_loss_coeff"])
    model.xi_conv = Param(initialize=data["heating_efficiency_coeff"])
    model.xi_cool = Param(initialize=data["heat_vent_coeff"])
    model.xi_occ = Param(initialize=data["heat_occupancy_coeff"])

    model.eta_occ = Param(initialize=data["humidity_occupancy_coeff"])
    model.eta_vent = Param(initialize=data["humidity_vent_coeff"])

    # variables
    model.p = Var(model.R, model.T, domain=NonNegativeReals) # heating power in room r at time t
    model.T_in = Var(model.R,model.T, domain=NonNegativeReals) # Indoor temp in room r at time t
    model.H = Var(model.T, domain = NonNegativeReals) # indoor humidity at time t
    model.v = Var(model.T, domain=Binary) # binary indicating whether ventilation is on at time t
    model.s = Var(model.T, domain=Binary) # binary variable indicating ventilation startup at time t
    model.y_low = Var(model.R, model.T, domain=Binary) # binary var to detect when room temp is below T_low
    model.y_ok = Var(model.R, model.T, domain=Binary) # binary var to detect when room temp is between thresholds
    model.y_high = Var(model.R, model.T, domain=Binary) # binary var to detect when room temp is above thresholds
    model.u = Var(model.R, model.T, domain = Binary) # binary indicating if heater's overruler is active at time t

    # -------------
    # Constraints
    # -------------
    def temperature_dynamics(m,r,t):
        r_mark = 0 if r == 1 else 1
        if t == 0:
            return (m.T_in[r,t] == (data["T1"] if r == 0 else data["T2"])) 
        else:
            return (m.T_in[r,t] == m.T_in[r,t-1] + 
                m.xi_exch*(m.T_in[r_mark,t-1] - m.T_in[r,t-1]) -
                m.xi_loss*(m.T_in[r,t-1] - m.T_out[t-1]) + m.xi_conv*m.p[r,t-1] -
                m.xi_cool*m.v[t-1] + m.xi_occ*m.kappa[r,t-1])
        
    model.room_temp = Constraint(model.R, model.T, rule=temperature_dynamics)

    def humidity_dynamics(m,t):
        if t == 0:
            return m.H[t] == data["H"]
        else:
            return m.H[t] == m.H[t-1] + m.eta_occ*sum(m.kappa[r,t-1] for r in m.R) - m.eta_vent*m.v[t-1]
    model.humidity_level = Constraint(model.T, rule=humidity_dynamics)

    def heating_power_limits_lb(m,r,t):
        return 0 <= m.p[r,t]
    model.heating_power_limits_lb = Constraint(model.R, model.T, rule = heating_power_limits_lb)

    def heating_power_limits_ub(m,r,t):
        return m.p[r,t] <= m.P_bar[r]
    model.heating_power_limits_ub = Constraint(model.R, model.T, rule = heating_power_limits_ub)

    def temp_high_1(m,r,t):
        return m.T_in[r,t] >= m.T_high - m.M_temp*(1-m.y_high[r,t])
    def temp_high_2(m,r,t):
        return m.T_in[r,t] <= m.T_high + m.M_temp*m.y_high[r,t]
    model.temp_high_1 = Constraint(model.R, model.T, rule = temp_high_1)
    model.temp_high_2 = Constraint(model.R, model.T, rule = temp_high_2)

    def overrule_when_hot(m,r,t):
        return m.p[r,t] <= m.P_bar[r]*(1-m.y_high[r,t])
    model.overrule_when_hot = Constraint(model.R,model.T, rule = overrule_when_hot)

    def detect_when_cold_1(m,r,t):
        return m.T_in[r,t] <= m.T_low + m.M_temp*(1-m.y_low[r,t])
    def detect_when_cold_2(m,r,t):
        return m.T_in[r,t] >= m.T_low - m.M_temp*m.y_low[r,t]
    
    model.detect_when_cold_1 = Constraint(model.R,model.T, rule=detect_when_cold_1)
    model.detect_when_cold_2 = Constraint(model.R,model.T, rule=detect_when_cold_2)

    def detect_temp_ok_1(m,r,t):
        return m.T_in[r,t] >= m.T_ok - m.M_temp*(1-m.y_ok[r,t])
    def detect_temp_ok_2(m,r,t):
        return m.T_in[r,t] <= m.T_ok + m.M_temp*m.y_ok[r,t]
    model.detect_temp_ok_1 = Constraint(model.R, model.T, rule=detect_temp_ok_1)
    model.detect_temp_ok_2 = Constraint(model.R, model.T, rule=detect_temp_ok_2)

    def overrule_heater_cold_1(m,r,t):
        return m.u[r,t] >= m.y_low[r,t]
    def overrule_heater_cold_2(m,r,t):
        return m.u[r,t] <= (m.u[r,t-1] if t > 0 else 0) + m.y_low[r,t]
    model.overrule_heater_cold_1 = Constraint(model.R, model.T, rule=overrule_heater_cold_1)
    model.overrule_heater_cold_2 = Constraint(model.R, model.T, rule=overrule_heater_cold_2)

    def overrule_heater_to_max(m,r,t):
        return m.p[r,t] >= m.P_bar[r]*m.u[r,t]
    model.overrule_heater_to_max = Constraint(model.R, model.T, rule=overrule_heater_to_max)

    def deactivate_overrule_temp_1(m,r,t):
        return m.u[r,t] >= (m.u[r,t-1] if t > 0 else (data["low_override_r1"] if r == 0 else data["low_override_r2"])) - m.y_ok[r,t]
    def deactivate_overrule_temp_2(m,r,t):
        if t <= 0:
            return m.u[r,t] == (data["low_override_r1"] if r == 0 else data["low_override_r2"])
        return m.u[r,t] <= 1 - m.y_ok[r,t]
    
    model.deactivate_overrule_temp_1 = Constraint(model.R, model.T, rule=deactivate_overrule_temp_1)
    model.deactivate_overrule_temp_2 = Constraint(model.R, model.T, rule=deactivate_overrule_temp_2)

    def ventilation_start_1(m,t):
        return m.s[t] >= m.v[t] - (m.v[t-1] if t > 0 else 0)
    def ventilation_start_2(m,t):
        return m.s[t] <= m.v[t]
    def ventilation_start_3(m,t):
        return m.s[t] <= 1 - (m.v[t-1] if t > 0 else 0)
    model.ventilation_start_1 = Constraint(model.T, rule=ventilation_start_1)
    model.ventilation_start_2 = Constraint(model.T, rule=ventilation_start_2)
    model.ventilation_start_3 = Constraint(model.T, rule=ventilation_start_3)

    def min_vent_hours(m,t):
        U_vent = value(m.U_vent)
        L = value(m.L)
        min_hours = min(t+U_vent-1,L-1)
        return sum(m.v[tau] for tau in range(t,min_hours+1)) >= min(U_vent,L-t)*m.s[t]
    model.min_vent_hours = Constraint(model.T, rule=min_vent_hours)

    def overrule_ventilation_on(m,t):
        return m.H[t] <= m.H_high + m.M_hum*m.v[t]
    model.overrule_vent_on = Constraint(model.T, rule=overrule_ventilation_on)

    # ------------------
    # Objective Function
    # ------------------
    def obj_rule(m):
        return (
            sum(m.lambda_tilde[t] 
                * (m.P_vent * m.v[t]
                + sum(m.p[r, t]
                    for r in m.R))
                for t in m.T)
            )
    model.obj = Objective(rule=obj_rule, sense=minimize)

    return model


def run_multiple_days(days, occupancy, prices, look_ahead):
    solver = SolverFactory('gurobi')
    if not solver.available():
        raise RuntimeError("Gurobi solver is not available in your environment.")
    
    results_all_days = {}

    for day in range(0, days):

        model = build_model_for_day(day, occupancy, prices, look_ahead)

        solver.solve(model)

        results_all_days[day] = {
        "objective": value(model.obj),
    }
        
    days_list = sorted(results_all_days.keys())
    objective_values = [results_all_days[d]["objective"] for d in days_list]

    print(f"Average daily electricity cost over {days} days:", np.mean(objective_values))

    plt.figure()
    plt.plot(days_list, objective_values)
    plt.xlabel("Day")
    plt.ylabel("Objective Value")
    plt.title("Optimal Objective Value per Day")
    plt.show()

# --------------------------
# Solve with Gurobi
# --------------------------
occupancy = generate_occupancy_dict()
prices = generate_price_dict()

run_multiple_days(100, occupancy, prices, data["num_timeslots"])