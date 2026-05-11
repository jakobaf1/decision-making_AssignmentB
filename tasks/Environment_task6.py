import numpy as np
from data.v2_SystemCharacteristics import get_fixed_data
from v2_Checks import *
data = get_fixed_data()

import matplotlib.pyplot as plt


def environment(policy):
    # Prices: shape (num_days, num_timeslots)
    prices = np.genfromtxt("data/v2_PriceData.csv", delimiter=",", skip_header=1)

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
        state = {
                    "T1": data["T1"], #Temperature of room 1
                    "T2": data["T2"], #Temperature of room 2
                    "H": data["H"], #Humidity
                    "Occ1": occupancy[0,day,0], #Occupancy of room 1
                    "Occ2": occupancy[1,day,0], #Occupancy of room 2
                    "price_t": prices[day,1], #Price
                    "price_previous": prices[day,0], #Previous Price
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
            v = action["VentilationON"]
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
                p[0] = action["HeatPowerRoom1"]
            


            if state["T2"] > data["temp_max_comfort_threshold"]:
                p[1] = 0
            elif state["low_override_r2"] == 1:
                p[1] = PowerMax[2]
            else:
                p[1] = action["HeatPowerRoom2"]

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
                
                state["price_t"] = prices[day,t+2]
                state["Occ1"] = occupancy[0,day,t+1]
                state["Occ2"] = occupancy[1,day,t+1]
            
        costs.append(cost_day)
    
    avg_cost = sum(costs)/E
    print(f"Average daily cost: {avg_cost}")
    return costs