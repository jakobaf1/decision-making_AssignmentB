import numpy as np
from data.v2_SystemCharacteristics import get_fixed_data
from Checks import *
data = get_fixed_data()


def environment(policy):
    # Prices: shape (num_days, num_timeslots)
    prices = np.genfromtxt("data/PriceData.csv", delimiter=",", skip_header=1)

    # Occupancies: shape (num_rooms, num_days, num_timeslots)
    occ_room1 = np.genfromtxt("data/OccupancyRoom1.csv", delimiter=",", skip_header=1)
    occ_room2 = np.genfromtxt("data/OccupancyRoom2.csv", delimiter=",", skip_header=1)
    occupancy = np.stack([occ_room1, occ_room2], axis=0)

    heating_max = data["heating_max_power"]
    PowerMax = {1: heating_max, 2: heating_max}

    # tolerance introduced to eliminate errors from rounding
    tol = 1e-4 

    costs = []
    E = 100
    for day in range(E):
        policy.day = day
        state = {
                    "T1": data["T1"], #Temperature of room 1
                    "T2": data["T2"], #Temperature of room 2
                    "H": data["H"], #Humidity
                    "Occ1": occupancy[0,day,0], #Occupancy of room 1
                    "Occ2": occupancy[1,day,0], #Occupancy of room 2
                    "price_t": prices[day,0], #Price
                    "price_previous": data["price_previous"], #Previous Price
                    "vent_counter": data["vent_counter"], #For how many consecutive hours has the ventilation been on 
                    "low_override_r1": data["low_override_r1"], #Is the low-temperature overrule controller of room 1 active 
                    "low_override_r2": data["low_override_r2"], #Is the low-temperature overrule controller of room 2 active 
                    "current_time": 0, #What is the hour of the day
                }
        cost_day = 0
        for t in range(data["num_timeslots"]):
            action = check_and_sanitize_action(policy, state, PowerMax)
            
            ## apply actions (or overrule if necessary)

            # Ventilation
            v = action["v"]
            if state["H"] > data["humidity_threshold"]:
                v = 1
            elif state["vent_counter"] in [1, 2]:
                v = 1

            # heating power
            p = [0, 0]
            if state["T1"] > data["temp_max_comfort_threshold"]:
                p[0] = 0
            elif state["low_override_r1"] == 1:
                p[0] = PowerMax[1]
            else:
                p[0] = action["p1"]
            


            if state["T2"] > data["temp_max_comfort_threshold"]:
                p[1] = 0
            elif state["low_override_r2"] == 1:
                p[1] = PowerMax[2]
            else:
                p[1] = action["p2"]

            ### exogenous variables are pre-loaded data, and needs no update

            # temperature dynamics
            new_T1 = (state["T1"]
                    + data["heat_exchange_coeff"]*(state["T2"] - state["T1"])
                    + data["thermal_loss_coeff"]*(data["outdoor_temperature"][t] - state["T1"])
                    + data["heating_efficiency_coeff"]*p[0]
                    - data["heat_vent_coeff"]*v
                    + data["heat_occupancy_coeff"]*state["Occ1"]
                    )
            new_T2 = (state["T2"]
                    + data["heat_exchange_coeff"]*(state["T1"] - state["T2"])
                    + data["thermal_loss_coeff"]*(data["outdoor_temperature"][t] - state["T2"])
                    + data["heating_efficiency_coeff"]*p[1]
                    - data["heat_vent_coeff"]*v
                    + data["heat_occupancy_coeff"]*state["Occ2"]
                    )
            # humidity dynamics
            new_H = (state["H"]
                    + data["humidity_occupancy_coeff"]*(state["Occ1"] + state["Occ2"])
                    - data["humidity_vent_coeff"]*v
                    )

            # low-temperature hysterisis update
            y1 = 0
            if new_T1 < data["temp_min_comfort_threshold"] - tol:
                y1 = 1
            elif state["low_override_r1"] == 1 and new_T1 < data["temp_OK_threshold"] - tol:
                y1 = 1

            y2 = 0
            if new_T2 < data["temp_min_comfort_threshold"] - tol:
                y2 = 1
            elif state["low_override_r2"] == 1 and new_T2 < data["temp_OK_threshold"]- tol:
                y2 = 1

            # ventilation consecutive hours counter
            new_c = state["vent_counter"] + 1 if v == 1 else 0

            # cost
            cost_day += prices[day,t]*(sum(p[r] for r in [0,1]) + data["ventilation_power"]*v)
            
            if t < data["num_timeslots"] - 1:
                # update state
                state["T1"] = new_T1
                state["T2"] = new_T2
                state["H"] = new_H
                state["low_override_r1"] = y1
                state["low_override_r2"] = y2
                state["vent_counter"] = new_c
                
                state["current_time"] = t + 1
                state["price_previous"] = state["price_t"]
                
                state["price_t"] = prices[day,t+1]
                state["Occ1"] = occupancy[0,day,t+1]
                state["Occ2"] = occupancy[1,day,t+1]
            
        costs.append(cost_day)
    
    avg_cost = sum(costs)/E
    return avg_cost



### Environment as a class ###


# # Define environment class
# class Environment:
#     def __init__(self, occupancy, prices, data):
#         self.occupancy = occupancy
#         self.prices = prices
#         self.data = data

#     def get_state(self):
#         return {
#                 "T1": self.T1, #Temperature of room 1
#                 "T2": self.T2, #Temperature of room 2
#                 "H": self.H, #Humidity
#                 "Occ1": self.occupancy[0, self.day, self.t], #Occupancy of room 1
#                 "Occ2": self.occupancy[1, self.day, self.t], #Occupancy of room 2
#                 "price_t": self.prices[self.day, self.t], #Price
#                 "price_previous": self.prices[self.day, self.t-1] if self.t > 0 else 0, #Previous Price
#                 "vent_counter": self.vent_counter, #For how many consecutive hours has the ventilation been on 
#                 "low_override_r1": self.low_override_r1, #Is the low-temperature overrule controller of room 1 active 
#                 "low_override_r2": self.low_override_r2, #Is the low-temperature overrule controller of room 2 active 
#                 "current_time": self.t #What is the hour of the day
#         }
    
#     def reset_day(self, day):
#         self.day = day
#         self.t = 0
#         self.T1 = self.data["T1"]
#         self.T2 = self.data["T2"]
#         self.H = self.data["H"]
#         self.vent_counter = self.data["vent_counter"]
#         self.low_override_r1 = self.data["low_override_r1"]
#         self.low_override_r2 = self.data["low_override_r2"]
#         self.cost_day = 0
    
#     def transtition(self, action):
#         state = self.get_state()
#         data = self.data

#         # apply actions (or overrule if necessary)

#         # Ventilation
#         v = action["VentilationOn"]
#         if state["H"] > data["humidity_threshold"]:
#             v = 1
#         elif state["vent_counter"] in [1,2]:
#             v = 1

#         # heating power
#         p = [action["HeatPowerRoom1"], action["HeatPowerRoom2"]]
#         if state["T1"] > data["temp_max_comfort_threshold"]:
#             p[0] = 0
#         elif state["low_override_r1"] == 1:
#             p[0] = data["heating_max_power"]
        
#         if state["T2"] > data["temp_max_comfort_threshold"]:
#             p[1] = 0
#         elif state["low_override_r2"] == 1:
#             p[1] = data["heating_max_power"]

#         ### exogenous variables are pre-loaded data, and needs no update

#         # temperature dynamics
#         new_T1 = (self.T1
#                   + data["heat_exchange_coeff"]*(self.T2 - self.T1)
#                   + data["thermal_loss_coeff"]*(data["outdoor_temperature"][self.t] - self.T1)
#                   + data["heating_efficiency_coef"]*p[0]
#                   + data["heat_vent_coeff"]*v
#                   + data["heat_occupancy_coef"]*self.occupancy[0, self.day, self.t]
#                   )
#         new_T2 = (self.T2
#                   + data["heat_exchange_coeff"]*(self.T1 - self.T2)
#                   + data["thermal_loss_coeff"]*(data["outdoor_temperature"][self.t] - self.T2)
#                   + data["heating_efficiency_coef"]*p[1]
#                   + data["heat_vent_coeff"]*v
#                   + data["heat_occupancy_coef"]*self.occupancy[1, self.day, self.t]
#                   )
#         # humidity dynamics
#         new_H = (self.H
#                  + data["humidity_occupancy_coef"]*sum(self.occupancy[r,self.day,self.t] for r in [0,1])
#                  - data["humidity_vent_coef"]*v
#                  )

#         # low-temperature hysterisis update
#         y1 = 0
#         if new_T1 <= data["temp_min_comfort_threshold"]:
#             y1 = 1
#         elif state["low_override_r1"] == 1 and new_T1 < data["temp_OK_threshold"]:
#             y1 = 1
        
#         y2 = 0
#         if new_T2 <= data["temp_min_comfort_threshold"]:
#             y1 = 1
#         elif state["low_override_r1"] == 1 and new_T2 < data["temp_OK_threshold"]:
#             y1 = 1

#         # ventilation consecutive hours counter
#         new_c = self.vent_counter + 1 if v == 1 else 0

#         # cost
#         self.cost_day += self.prices[self.day,self.t]*(sum(p[r] for r in [0,1]) + data["ventilation_power"]*v)
        
#         # update state
#         self.T1, self.T2, self.H = new_T1, new_T2, new_H
#         self.low_override_r1, self.low_override_r2 = y1, y2
#         self.vent_counter = new_c
        
#         self.t += 1

#     def run_policy_full_day(self, policy):
#         for i in range(self.data["num_timeslots"]):
#             action = policy(self.get_state())
#             self.transtition(action)
#         return self.cost_day
    
#     def evaluate_policy(self, policy, days=100):
#         costs=[]
#         for day in range(days):
#             self.reset_day(day)
#             cost = self.run_policy_full_day(policy)
#             costs.append(cost)
#         return costs