import pandas as pd
import matplotlib.pyplot as plt
import itertools

def analyze_vanilla():
    # Read the CSV file
    df = pd.read_csv("grid_search_results.csv")

    # Identify the best performing configuration based on avg_return
    best_config = df.loc[df['avg_return'].idxmax()]
    print("Best performing configuration:")
    print(best_config)

    # -------------------------------------------------
    # Part 1: Compute and plot variance of each hyperparameter individually
    # -------------------------------------------------
    hyperparams = ['gamma', 'learning_rate', 'hidden_dim', 'num_hidden_layers', 'policy', 'epsilon', 'temp']
    individual_variances = {}

    # For each hyperparameter, group the results by its distinct values,
    # compute the mean avg_return for each group, and then compute the variance across these means.
    for hp in hyperparams:
        group_means = df.groupby(hp)['avg_return'].mean()
        # Variance across different levels of the hyperparameter
        variance = group_means.var()
        individual_variances[hp] = variance

    # Plot the individual variances in a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(individual_variances.keys(), individual_variances.values())
    plt.xlabel("Hyperparameter")
    plt.ylabel("Variance of Mean avg_return")
    plt.title("Variance of avg_return for each Hyperparameter")
    plt.show()

    # -------------------------------------------------
    # Part 2: Compute and plot joint variances for pairs of hyperparameters (Top 10)
    # -------------------------------------------------
    joint_variances = {}

    # Iterate over every pair of hyperparameters
    for hp1, hp2 in itertools.combinations(hyperparams, 2):
        # Group by the combination of the two hyperparameters and compute the mean avg_return for each combination.
        group_means = df.groupby([hp1, hp2])['avg_return'].mean()
        variance = group_means.var()
        joint_variances[(hp1, hp2)] = variance

    # Sort the joint variances in descending order and pick the top 10 pairs
    top_10_joint = sorted(joint_variances.items(), key=lambda x: x[1], reverse=True)[:10]

    # Prepare labels and variance values for plotting
    labels = [f"{hp1} & {hp2}" for (hp1, hp2), var in top_10_joint]
    variances = [var for (hp1, hp2), var in top_10_joint]

    plt.figure(figsize=(12, 6))
    plt.bar(labels, variances)
    plt.xlabel("Hyperparameter Pair")
    plt.ylabel("Variance of Mean avg_return")
    plt.title("Top 10 Hyperparameter Pairs by Variance in avg_return")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


import numpy as np



def Analyze_Replay_Buffer_Tuning(csv_filename="report/Tunes_1/replay_buffer_tuning_results.csv"):
    # Load results from CSV
    df = pd.read_csv(csv_filename)

    if df.empty:
        print("Error: No data found in CSV file!")
        return

    # Compute variance for capacity
    capacity_variance = df.groupby("capacity")["avg_return"].var().mean()

    # Compute variance for batch size
    batch_size_variance = df.groupby("batch_size")["avg_return"].var().mean()

    # Compute joint variance across all settings
    joint_variance = np.var(df["avg_return"])

    # Find best configuration (highest avg return)
    best_config = df.loc[df["avg_return"].idxmax()]

    # Print variance summary
    print("\n🔥 Variance Summary 🔥")
    print(f"Variance for Capacity: {capacity_variance:.4f}")
    print(f"Variance for Batch Size: {batch_size_variance:.4f}")
    print(f"Joint Variance: {joint_variance:.4f}")

    # Print best configuration
    print("\n🔥 Optimal Replay Buffer Configuration 🔥")
    print(best_config.to_string(index=True))

    # Plot bar chart
    labels = ["Capacity Variance", "Batch Size Variance"]
    variances = [capacity_variance, batch_size_variance]

    plt.figure(figsize=(8, 6))
    plt.bar(labels, variances, color=['blue', 'red'], alpha=0.7)

    # Labels & formatting
    plt.ylabel("Variance of Performance")
    plt.title("Replay Buffer Variance Analysis")
    plt.grid(axis='y', linestyle="--", alpha=0.6)

    # Show plot
    plt.show()

# ✅ Run analysis only if executed directly
if __name__ == "__main__":
    Analyze_Replay_Buffer_Tuning()
