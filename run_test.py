from tasks.Environment_task6 import environment
from tasks.SP_policy_task3 import SPPolicy
from tasks.Dummy_policy import DummyPolicy
from tasks.hindsight_policy_task1 import HindsightPolicy

print("Dummy policy:    ", environment(DummyPolicy()))
print("SP policy:       ", environment(SPPolicy()))
print("Hindsight policy:", environment(HindsightPolicy()))
