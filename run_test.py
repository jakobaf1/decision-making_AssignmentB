import csv
from tasks.Environment_task6 import environment, dummy_policy
from tasks.hindsight_policy_task1 import build_model_for_day
from tasks.SP_policy_task3 import select_action as sp_select_action
from data.v2_SystemCharacteristics import get_fixed_data
from pyomo.environ import SolverFactory, value

data = get_fixed_data()


def load_occupancy(file_room1="data/OccupancyRoom1.csv", file_room2="data/OccupancyRoom2.csv"):
    occ = {}
    for room, filename in {0: file_room1, 1: file_room2}.items():
        with open(filename) as f:
            reader = csv.reader(f)
            next(reader)
            for day, row in enumerate(reader):
                for hour, val in enumerate(row):
                    occ[(room, day, hour)] = float(val)
    return occ


def load_prices(filename="data/PriceData.csv"):
    prices = {}
    with open(filename) as f:
        reader = csv.reader(f)
        num_hours = len(next(reader))
        for day, row in enumerate(reader):
            for hour in range(num_hours):
                prices[(day, hour)] = float(row[hour])
    return prices


def make_hindsight_env_policy():
    """
    Wraps the hindsight MILP so it works as a stateless policy(state) callable.
    The environment never updates state["current_time"], so we track progress
    with a call counter: every num_timeslots calls = one day.
    """
    occupancy = load_occupancy()
    prices    = load_prices()
    solver    = SolverFactory("gurobi")
    N         = data["num_timeslots"]

    call_counter   = [0]
    cached_actions = {}

    def select_action(_state):
        n   = call_counter[0]
        t   = n % N      # hour within the day (0-9)
        day = n // N     # which day (0-99)
        call_counter[0] += 1

        if t == 0:
            model = build_model_for_day(day, occupancy, prices, N)
            solver.solve(model, tee=False)
            cached_actions.clear()
            for hour in range(N):
                cached_actions[hour] = {
                    "HeatPowerRoom1": float(value(model.p[0, hour])),
                    "HeatPowerRoom2": float(value(model.p[1, hour])),
                    "VentilationON":  int(round(float(value(model.v[hour]))))
                }
        return cached_actions[t]

    return select_action


hindsight_policy = make_hindsight_env_policy()

print("Dummy policy:    ", environment(dummy_policy))
print("SP policy:       ", environment(sp_select_action))
print("Hindsight policy:", environment(hindsight_policy))
