from pyomo.environ import *
from data.PlotsRestaurant import plot_HVAC_results
import csv
import matplotlib.pyplot as plt
import numpy as np
from data.SystemCharacteristics import get_fixed_data
data = get_fixed_data()

# The following script/modelling structure is based on the v3StudentProblem script given.

# takes the two occupancy filenames and generates a dictionary with the relevant data
def generate_occupancy_dict(file_room1="assignment_partA/data/OccupancyRoom1.csv",
                            file_room2="assignment_partA/data/OccupancyRoom2.csv"):
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
def generate_price_dict(filename="assignment_partA/data/PriceData.csv"):
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

# -------------------------
# Function which builds the model
# -------------------------
def build_model_for_day(day, occ_data, price_data):
    #### Pyomo model ####
    model = ConcreteModel()

    # Sets
    time_slots = list(range(0,data['num_timeslots']))
    model.T = Set(initialize=time_slots, ordered=True)

    rooms = list(range(0,2))
    model.R = Set(initialize=rooms, ordered=True)


    # parameters
    model.D = Param(initialize=day)
    model.P = Param(model.R, initialize=data['heating_max_power'])
    model.p_vent = Param(initialize=data["ventilation_power"])
    model.T_low = Param(initialize=data["temp_min_comfort_threshold"])
    model.T_ok = Param(initialize=data["temp_OK_threshold"])
    model.T_high = Param(initialize=data["temp_max_comfort_threshold"])
    model.H_high = Param(initialize=data["humidity_threshold"])
    model.v_min = Param(initialize=data["vent_min_up_time"])
    model.T_0 = Param(initialize=data["initial_temperature"])

    # find data for the relevant day
    occ_day = {
        (r, t): occ_data[(r, day, t)]
        for (r, d, t) in occ_data
        if d == day
    }

    model.occ = Param(model.R, model.T, initialize=occ_day)

    model.lambda_tilde = Param(model.T,
                        initialize={t: price_data[(day, t)]
                                    for t in model.T})

    model.xi_exch = Param(initialize=data["heat_exchange_coeff"])
    model.xi_loss = Param(initialize=data["thermal_loss_coeff"])
    model.T_out = Param(model.T, initialize=data["outdoor_temperature"])
    model.xi_cool = Param(initialize=data["heat_vent_coeff"])
    model.xi_occ = Param(initialize=data["heat_occupancy_coeff"])
    model.xi_conv = Param(initialize=data["heating_efficiency_coeff"])

    model.H_0 = Param(initialize=data["initial_humidity"])
    model.eta_occ = Param(initialize=data["humidity_occupancy_coeff"])
    model.eta_vent = Param(initialize=data["humidity_vent_coeff"])

    # variables
    model.p = Var(model.R, model.T, domain=NonNegativeReals)
    model.v = Var(model.T, domain=Binary)
    model.v_start = Var(model.T, domain=Binary)
    model.T_in = Var(model.R,model.T, domain=NonNegativeReals)
    model.H = Var(model.T, domain = NonNegativeReals)
    model.o = Var(model.R, model.T, domain = Binary)
    model.h_off = Var(model.R, model.T, domain=Binary)

    # -------------
    # Constraints
    # -------------
    def room_temperature_rule(m,r,t):
        if t == 0:
            return (m.T_in[r,t] == m.T_0) 
        else:
            return (m.T_in[r,t] == m.T_in[r,t-1] + 
                m.xi_exch*sum(m.T_in[r,t-1] - m.T_in[r2,t-1] for r2 in m.R) +
                m.xi_loss*(m.T_out[t-1] - m.T_in[r,t-1]) + m.xi_conv*m.p[r,t] +
                m.xi_cool*m.v[t-1] + m.xi_occ*m.occ[r,t-1])
        
    model.room_temp = Constraint(model.R, model.T, rule=room_temperature_rule)

    def humidity_level_rule(m,t):
        if t == 0:
            return m.H[t] == m.H_0
        else:
            return m.H[t] == m.H[t-1] + m.eta_occ*sum(m.occ[r,t-1] for r in m.R) - m.eta_vent*m.v[t-1]
    model.humidity_level = Constraint(model.T, rule=humidity_level_rule)

    def overrule_condition_rule_cold(m,r,t):
        return m.T_in[r,t] >= m.T_low*(1-m.o[r,t])
    model.overrule_temp_cold = Constraint(model.R,model.T, rule=overrule_condition_rule_cold)

    def temp_if_overrule_rule(m,r,t):
        return m.p[r,t] >= m.P[r]*m.o[r,t]
    model.temp_if_overrule = Constraint(model.R, model.T, rule = temp_if_overrule_rule)

    def overrule_until_ok(m,r,t):
        return m.T_high*m.o[r,t] + m.T_in[r,t] >= m.T_ok*(m.o[r,t-1] if t > 0 else 0)
    model.overrule_until_ok = Constraint(model.R, model.T, rule=overrule_until_ok)

    def overrule_condition_rule_hot(m,r,t):
        return m.T_in[r,t]*(1-m.h_off[r,t]) <= m.T_high
    model.overrule_temp_hot = Constraint(model.R,model.T, rule = overrule_condition_rule_hot)

    def overrule_heater_off(m,r,t):
        return m.p[r,t] <= m.P[r]*(1-m.h_off[r,t])
    model.overrule_heater_off = Constraint(model.R, model.T, rule=overrule_heater_off)

    def overrule_ventilation_on(m,t):
        return m.H[t]*(1-m.v[t]) <= m.H_high
    model.overrule_vent_on = Constraint(model.T, rule=overrule_ventilation_on)

    def ventilation_start_def(m,t):
        return m.v_start[t] >= m.v[t] - (m.v[t-1] if t > 0 else 0)
    model.ventilation_start = Constraint(model.T, rule=ventilation_start_def)

    def ventilation_started_means_on(m,t):
        return m.v[t] >= m.v_start[t]
    model.ventilation_started_means_on = Constraint(model.T, rule=ventilation_started_means_on)

    def min_vent_hours(m,t):
        if t > max(m.T)-2:
            return m.v_start[t] == 0
        else:
            return 3*m.v_start[t] <= m.v[t] + m.v[t+1] + m.v[t+2]
    model.min_vent_hours = Constraint(model.T, rule=min_vent_hours)

    def max_heater_power(m,r,t):
        return m.p[r,t] <= m.P[r]
    model.max_heater_power = Constraint(model.R,model.T, rule=max_heater_power)

    # ------------------
    # Objective Function
    # ------------------
    def obj_rule(m):
        return (
            sum(m.lambda_tilde[t] * m.p[r, t]
            for r in m.R
            for t in m.T) + 
            sum(m.lambda_tilde[t] * m.p_vent * m.v[t] 
            for t in m.T)
            )
    model.obj = Objective(rule=obj_rule, sense=minimize)

    return model

# Used to run the model for a single day and store the results
def run_single_day(day, occupancy, prices):
    model = build_model_for_day(day, occupancy, prices)

    solver.solve(model)

    HVAC_results = {}
    # Time index
    HVAC_results["T"] = list(model.T)

    # Temperatures
    HVAC_results["Temp_r1"] = [value(model.T_in[0,t]) for t in model.T]
    HVAC_results["Temp_r2"] = [value(model.T_in[1,t]) for t in model.T]

    # Heater power
    HVAC_results["h_r1"] = [value(model.p[0,t]) for t in model.T]
    HVAC_results["h_r2"] = [value(model.p[1,t]) for t in model.T]

    # Ventilation
    HVAC_results["v"] = [value(model.v[t]) for t in model.T]

    # Humidity
    HVAC_results["Hum"] = [value(model.H[t]) for t in model.T]

    # Price (parameter)
    HVAC_results["price"] = [value(model.lambda_tilde[t]) for t in model.T]

    # Occupancy (parameter)
    HVAC_results["Occ_r1"] = [value(model.occ[0,t]) for t in model.T]
    HVAC_results["Occ_r2"] = [value(model.occ[1,t]) for t in model.T]

    print(f"Electricity cost on day {day}:", value(model.obj))

    plot_HVAC_results(HVAC_results)

# runs the model over "days" amount of days
def run_multiple_days(days, occupancy, prices):
    results_all_days = {}

    for day in range(0, days):

        model = build_model_for_day(day, occupancy, prices)

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
if __name__ == "__main__":
    solver = SolverFactory('gurobi')
    if not solver.available():
        raise RuntimeError("Gurobi solver is not available in your environment.")

    occupancy = generate_occupancy_dict()
    prices = generate_price_dict()

    run_multiple_days(100, occupancy, prices)

    # run_single_day(83, occupancy, prices)

    


