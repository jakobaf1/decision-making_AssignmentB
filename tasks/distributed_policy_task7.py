from pyomo.environ import *
import numpy as np
from data.DataTask7 import fetch_data
data = fetch_data()

class distributed_policy:
    def __init__(self):
        occupancy = np.genfromtxt("data/Task7Occupancies.csv", delimiter=",", skip_header=1)


    def build_model_distributed(self, n, lamda_pen):
        #### Pyomo model ####
        model = ConcreteModel()

        # Sets
        time_slots = list(range(0,data['num_timeslots']))
        model.T = Set(initialize=time_slots, ordered=True)

        rooms = list(range(0,2))
        model.R = Set(initialize=rooms, ordered=True)


        #### parameters ###
        occ = {(r, t): self.occupancy[r, t] for r in model.R for t in model.T}

        model.kappa = Param(model.R, model.T, initialize=occ)
        model.w = Param(model.N, initialize= n + 1)


        model.P_mall = Param(initialize=data['P_mall'])
        model.T_ref = Param(initialize=data["Temperature_reference"])
        model.T_0 = Param(initialize=data["initial_temperature"])

        model.T_out = Param(model.T, initialize=data["outdoor_temperature"])
        model.P_bar = Param(model.R, initialize=data['heating_max_power'])

        model.xi_exch = Param(initialize=data["heat_exchange_coeff"])
        model.xi_loss = Param(initialize=data["thermal_loss_coeff"])
        model.xi_conv = Param(initialize=data["heating_efficiency_coeff"])
        model.xi_cool = Param(initialize=data["heat_vent_coeff"])
        model.xi_occ = Param(initialize=data["heat_occupancy_coeff"])

        # variables
        model.p = Var(model.R, model.T, domain=NonNegativeReals) # heating power in room r at time t for store n
        model.p_store = Var(model.T, domain=NonNegativeReals) # heating power in stor n at time t
        model.T_in = Var(model.R,model.T, domain=NonNegativeReals) # Indoor temp in room r at time t
        model.T_n = Var(model.T, domain=NonNegativeReals)

        # -------------
        # Constraints
        # -------------

        def temperature_dynamics(m,r,t):
            r_mark = 0 if r == 1 else 1
            if t == 0:
                return (m.T_in[r,t] == model.T_0) 
            else:
                return (m.T_in[r,t] == m.T_in[r,t-1] + 
                    m.xi_exch*(m.T_in[r_mark,t-1] - m.T_in[r,t-1]) -
                    m.xi_loss*(m.T_in[r,t-1] - m.T_out[t-1]) + m.xi_conv*m.p[r,t-1] -
                    m.xi_cool + m.xi_occ*m.kappa[r,t-1])
            
        model.room_temp = Constraint(model.R, model.T, rule=temperature_dynamics)

        def heating_power_limits_lb(m,r,t):
            return 0 <= m.p[r,t]
        model.heating_power_limits_lb = Constraint(model.R, model.T, rule = heating_power_limits_lb)

        def heating_power_limits_ub(m,r,t):
            return m.p[r,t] <= m.P_bar[r]
        model.heating_power_limits_ub = Constraint(model.R, model.T, rule = heating_power_limits_ub)

        def power_store(m,t):
            return m.p_store[t] == sum(m.p[r,t] for r in [0,1])
        model.power_store = Constraint(model.T, rule=power_store)

        # The store temperature is defined as the average of the rooms
        def temperature_store(m,t):
            return m.T_n[t] == sum(m.T_in[r,t] for r in [0,1])/2
        model.power_store = Constraint(model.T, rule=power_store)

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
    
    def select_action(self, state):
        return 