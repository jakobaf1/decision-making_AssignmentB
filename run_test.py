from tasks.Environment_task6 import environment
from tasks.SP_policy_task3 import SPPolicy
from tasks.Dummy_policy import DummyPolicy
from tasks.hindsight_policy_task1 import HindsightPolicy

print("Dummy policy:    ", environment(DummyPolicy()))
print("SP policy:       ", environment(SPPolicy()))
print("Hindsight policy:", environment(HindsightPolicy()))


import matplotlib.pyplot as plt
import numpy as np
def plot_cost_histogram(costs, policy_name="Policy"):
    mean_cost = np.mean(costs)

    fig, ax = plt.subplots(figsize=(10, 5))

    n, bins, patches = ax.hist(costs, bins=20, color="#4C9BE8", edgecolor="white", linewidth=0.6, alpha=0.85)

    ax.axvline(mean_cost, color="#E84C4C", linewidth=2, linestyle="--", label=f"Mean: {mean_cost:.2f}")

    ax.set_xlabel("Daily Electricity Cost", fontsize=12)
    ax.set_ylabel("Number of Days", fontsize=12)
    ax.set_title(f"Distribution of Daily Costs — {policy_name}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.show()

costs = environment(policy)
plot_cost_histogram(costs, policy_name="Optimal Hindsight")