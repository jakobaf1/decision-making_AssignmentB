import numpy as np
from tasks.Environment_task6 import environment
from tasks.SP_policy_task3 import SPPolicy
from tasks.Dummy_policy import DummyPolicy
from tasks.hindsight_policy_task1 import HindsightPolicy

dummy_costs     = environment(DummyPolicy())
sp_costs        = environment(SPPolicy())
hindsight_costs = environment(HindsightPolicy())

print("Dummy policy:    ", np.mean(dummy_costs))
print("SP policy:       ", np.mean(sp_costs))
print("Hindsight policy:", np.mean(hindsight_costs))
