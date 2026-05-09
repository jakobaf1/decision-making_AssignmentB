from tasks.Environment_task6 import *


from tasks.Dummy_policy import *
policy = DummyPolicy()



# from tasks.hindsight_policy_task1 import *
# policy = HindsightPolicy()


print(f"Average daily cost: {environment(policy)}")
