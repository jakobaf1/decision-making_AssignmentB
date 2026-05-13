from pyomo.environ import *
import numpy as np
from data.DataTask7 import fetch_data
data = fetch_data()
import matplotlib.pyplot as plt


class distributed_policy:
    def __init__(self):
        self.occupancy = np.genfromtxt("data/Task7Occupancies.csv", delimiter=",", skip_header=1)
        self.N = 15

    def build_full_model(self):
        #### Pyomo model ####
        model = ConcreteModel()

        # Sets
        time_slots = list(range(0,data['num_timeslots']))
        model.T = Set(initialize=time_slots, ordered=True)

        rooms = list(range(0,2))
        model.R = Set(initialize=rooms, ordered=True)

        stores = list(range(1,16))
        model.N = Set(initialize=stores, ordered=True)

        #### parameters ###
        occ = {(r, t): self.occupancy[r, t] for r in model.R for t in model.T}

        model.kappa = Param(model.R, model.T, initialize=occ)

        model.w = Param(model.N, initialize={n: n+1 for n in model.N})

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
        model.p = Var(model.N, model.R, model.T, domain=NonNegativeReals) # heating power in room r at time t for store n
        model.T_in = Var(model.N, model.R,model.T, domain=NonNegativeReals) # Indoor temp in room r at time t

        # -------------
        # Constraints
        # -------------

        def temperature_dynamics(m,n,r,t):
            r_mark = 0 if r == 1 else 1
            if t == 0:
                return (m.T_in[n,r,t] == model.T_0) 
            else:
                return (m.T_in[n,r,t] == m.T_in[n,r,t-1] + 
                    m.xi_exch*(m.T_in[n,r_mark,t-1] - m.T_in[n,r,t-1]) -
                    m.xi_loss*(m.T_in[n,r,t-1] - m.T_out[t-1]) + m.xi_conv*m.p[n,r,t-1] -
                    m.xi_cool + m.xi_occ*m.kappa[r,t-1])
            
        model.room_temp = Constraint(model.N, model.R, model.T, rule=temperature_dynamics)

        def heating_power_limits_ub(m,n,r,t):
            return m.p[n,r,t] <= m.P_bar[r]
        model.heating_power_limits_ub = Constraint(model.N, model.R, model.T, rule = heating_power_limits_ub)

        def power_cap(m, t):
            return sum(m.p[n,r,t] for n in m.N for r in m.R) <= m.P_mall
        model.power_cap = Constraint(model.T, rule=power_cap)

        # ------------------
        # Objective Function
        # ------------------
        def obj_rule(m):
            return (sum(m.w[n]*((m.T_in[n,r,t] - m.T_ref)*(m.T_in[n,r,t] - m.T_ref)) for t in m.T for r in [0,1] for n in m.N))
        model.obj = Objective(rule=obj_rule, sense=minimize)

        return model

    def build_model_distributed(self, n, lambda_pen):
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
        model.w = Param(initialize= n + 1)


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
        # model.p_store = Var(model.T, domain=NonNegativeReals) # heating power in stor n at time t
        model.T_in = Var(model.R,model.T, domain=NonNegativeReals) # Indoor temp in room r at time t

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

        # def heating_power_limits_lb(m,r,t):
        #     return 0 <= m.p[r,t]
        # model.heating_power_limits_lb = Constraint(model.R, model.T, rule = heating_power_limits_lb)

        def heating_power_limits_ub(m,r,t):
            return m.p[r,t] <= m.P_bar[r]
        model.heating_power_limits_ub = Constraint(model.R, model.T, rule = heating_power_limits_ub)

        # def power_store(m,t):
        #     return m.p_store[t] == sum(m.p[r,t] for r in [0,1])
        # model.power_store = Constraint(model.T, rule=power_store)

        # ------------------
        # Objective Function
        # ------------------
        def obj_rule(m):
            return (sum(m.w*((m.T_in[r,t] - m.T_ref)*(m.T_in[r,t] - m.T_ref)) for r in [0,1] for t in m.T) + sum(lambda_pen[t]*m.p[r,t] for r in range(2) for t in m.T))
        model.obj = Objective(rule=obj_rule, sense=minimize)

        return model
    
    def run_distributed_model(self, iter, alpha_val=0.1, adaptive=False):
        T = data['num_timeslots']
        objective_values = []
        lambdas = []
        p_values = []
        
        a_0 = 5
        alpha = alpha_val
        
        lambda_pen = [0.0 for _ in range(T)]
        p = [[] for _ in range(self.N)]
        w = [n+1 for n in range(1,self.N+1)]
        T_in = np.zeros((self.N, 2, T))
        T_in[:, :, 0] = data["initial_temperature"]

        for k in range(iter):

            if adaptive:
                alpha = a_0/(1+k)

            for n in range(self.N):

                model = self.build_model_distributed(n, lambda_pen)
                solver = SolverFactory('gurobi')
                solver.solve(model)

                p[n] = [sum(value(model.p[r,t]) for r in range(2)) for t in model.T]

                for r in range(2):
                    for t in range(T):
                        T_in[n,r,t] = value(model.T_in[r,t])
                    
            p_values.append({n: p[n].copy() for n in range(self.N)})
            
            for t in range(T):
                lambda_pen[t] = max(0, lambda_pen[t] + alpha*(sum(p[n][t] for n in range(self.N)) - data['P_mall']))

            lambdas.append(lambda_pen.copy())

            org_obj_val = sum(w[n]*(T_in[n,r,t] - data['Temperature_reference'])**2 for n in range(self.N) for t in range(T) for r in range(2))
            obj_pen = sum(lambda_pen[t]*(sum(p[n][t] for n in range(self.N)) - data['P_mall']) for t in range(T))
            # obj_lagrangian = org_obj_val + obj_pen
            objective_values.append(org_obj_val)
    

        return objective_values, lambdas, p_values
    
    def plot_diff_alphas(self, iter):
        alpha = [0.001, 0.01, 0.1, 1, 10]

        objective_values = []
        lambdas = []
        for a in alpha:
            obj, lambda_pens, _ = self.run_distributed_model(iter, alpha_val=a)
            objective_values.append(obj)
            lambdas.append(lambda_pens)
        
        # Adaptive alpha
        adaptive_objectives, lambda_pens, _ = self.run_distributed_model(iter, alpha_val=5, adaptive=True)
        lambdas.append(lambda_pens)
        # actual optimal solution
        true_optimum, _ = self.solve_full_model()

        self.plot_objectives(iter,alpha,objective_values, adaptive_objectives, true_optimum)
        self.plot_lambdas(iter, objective_values, lambdas)

        
    def plot_objectives(self, iter, alpha, objective_values, adaptive_objectives, true_optimum):
        colors = ["#4C9BE8", "#E84C4C", "#4CE87A", "#E8C44C", "#9B4CE8"]
        fig, ax = plt.subplots(figsize=(10, 6))

        for i, (a, obj) in enumerate(zip(alpha, objective_values)):
            ax.plot(range(iter), obj, color=colors[i], linewidth=1.8, label=f"α = {a}, objective value: {obj[-1]:.2f}")
        
        ax.plot(range(iter), adaptive_objectives, color="#4CDEE8", linewidth=1.8, label=f"Adaptive α (α₀=5), objective value: {adaptive_objectives[-1]:.2f}")

        ax.axhline(true_optimum, color="black", linewidth=1.5, 
                linestyle="--", label=f"Optimal objective value: {true_optimum:.2f}")

        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Objective Value", fontsize=12)
        ax.set_title("Distributed Algorithm: Objective Value vs Iterations", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        plt.show()

    def plot_lambdas(self, iter, objectives_list, lambdas):
        alpha = [0.001, 0.01, 0.1, 1, 10]
        colors = ["#4C9BE8", "#E84C4C", "#4CE87A", "#E8C44C", "#9B4CE8"]
        T = data['num_timeslots']

        labels = [f"α = {a}" for a in alpha] + ["Adaptive α (α₀=5)"]
        plot_colors = colors + ["black"]

        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        axes = axes.flatten()

        for idx, (label, color, lambda_history) in enumerate(zip(labels, plot_colors, lambdas)):
            ax = axes[idx]
            lambda_array = np.array(lambda_history)  # shape (iter, T)

            for t in range(T):
                ax.plot(range(iter), lambda_array[:, t], linewidth=1.5, label=f"t={t}")

            ax.set_title(label, fontsize=12, fontweight="bold")
            ax.set_xlabel("Iteration", fontsize=10)
            ax.set_ylabel("λ_t", fontsize=10)
            ax.legend(fontsize=7, ncol=2)
            ax.spines[["top", "right"]].set_visible(False)

        plt.suptitle("Evolution of Lagrange Multipliers λ_t across Iterations",
                    fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def plot_store_consumption(self, alpha_val=0.1, iter=100):
        # objective_values, _, p_values= self.run_distributed_model(iter, alpha_val=alpha_val, adaptive=True)
        # print(f"Final objective value using alpha={alpha_val}: {objective_values[-1]}")
        # use the last iteration's p values as the converged solution
        # p_final = p_values[-1] 
        # total_consumption = [sum(p_final[n][t] for t in range(data['num_timeslots'])) 
        #                     for n in range(self.N)]

        # using optimal solution
        objective_values, total_consumption = self.solve_full_model()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(1, self.N + 1), total_consumption, color="#4C9BE8", edgecolor="white")
        ax.set_xlabel("Store", fontsize=12)
        ax.set_ylabel("Total Energy Consumption", fontsize=12)
        ax.set_title("Total Energy Consumption per Store", fontsize=14, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.show()

    def plot_constraint_violation(self, iter):
        alpha = [0.001, 0.01, 0.1, 1, 10]
        colors = ["#4C9BE8", "#E84C4C", "#4CE87A", "#E8C44C", "#9B4CE8"]
        T = data['num_timeslots']

        cases = [(a, False, f"α = {a}", colors[i]) for i, a in enumerate(alpha)]
        cases.append((5, True, "Adaptive α (α₀=5)", "black"))

        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        axes = axes.flatten()

        for idx, (a, adaptive, label, color) in enumerate(cases):
            ax = axes[idx]
            _, _, p_values = self.run_distributed_model(iter, alpha_val=a, adaptive=adaptive)

            for t in range(T):
                violation = [sum(p_values[k][n][t] for n in range(self.N)) - data['P_mall']
                            for k in range(iter)]
                ax.plot(range(iter), violation, linewidth=1.5, label=f"t={t}")

            ax.axhline(0, color="black", linewidth=1.2, linestyle="--", label="Feasibility boundary")
            ax.set_title(label, fontsize=12, fontweight="bold")
            ax.set_xlabel("Iteration", fontsize=10)
            ax.set_ylabel(r"$\sum_n p_{n,t} - P^{\mathrm{mall}}$", fontsize=10)
            ax.legend(fontsize=7, ncol=2)
            ax.spines[["top", "right"]].set_visible(False)

        plt.suptitle("Constraint Violation per Timeslot across Iterations",
                    fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def solve_full_model(self):
        model = self.build_full_model()
        solver = SolverFactory('gurobi')
        solver.solve(model)

        # print(f"Objective value for full model: {value(model.obj)}")
        return value(model.obj), [sum((value(model.p[n,r,t])) for r in model.R for t in model.T) for n in range(1,16)]

    
    def select_action(self, state):
        return 
    