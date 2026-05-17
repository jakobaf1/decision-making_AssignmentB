import argparse
import matplotlib.pyplot as plt
import numpy as np


from tasks.Environment_task6 import *
from tasks.Dummy_policy import *
from tasks.hindsight_policy_task1 import *
from tasks.Hybrid_policy_7 import *
from tasks.distributed_policy_task7 import *


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


def main():
    parser = argparse.ArgumentParser(description="Run policies")
    subparsers = parser.add_subparsers(dest="command")

    # ---- Policy subcommand ----
    policy_parser = subparsers.add_parser("policy", help="Run a policy and plot cost histogram")
    policy_parser.add_argument("name", choices=["dummy", "hindsight", "sp", "adp", "hybrid"],
                                help="Which policy to run")

    # ---- Task 7 subcommand ----
    task7_parser = subparsers.add_parser("task7", help="Run Task 7 distributed model")
    task7_parser.add_argument("plot", choices=["objective", "consumption", "full_model", "violation"],
                               help="Which plot to generate")
    task7_parser.add_argument("--iter", type=int, default=100,
                               help="Number of iterations (default: 100)")

    args = parser.parse_args()

    # ---- Handle policy command ----
    if args.command == "policy":
        if args.name == "dummy":
            policy = DummyPolicy()
        elif args.name == "hindsight":
            policy = HindsightPolicy()
        elif args.name == "hybrid":
            policy = HybridPolicy()
        else:
            raise NotImplementedError(f"Policy '{args.name}' not yet implemented.")

        costs = environment(policy)
        plot_cost_histogram(costs, policy_name=args.name)

    # ---- Handle task7 command ----
    elif args.command == "task7":
        task7 = distributed_policy()
        if args.plot == "objective":
            task7.plot_diff_alphas(args.iter)
        elif args.plot == "consumption":
            task7.plot_store_consumption(iter=args.iter)
        elif args.plot == "full_model":
            obj, _ = task7.solve_full_model()
            print(f"Optimal objective value: {obj}")
        elif args.plot == "violation":
            task7.plot_constraint_violation(args.iter)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()