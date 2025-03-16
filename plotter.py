import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------- CONFIGURABLE PARAMETERS --------------
RESULTS_DIR = "results"     # Where your run CSVs are located
PLOTS_DIR = "plots"         # Where plots and smoothed CSVs will be saved
CONFIGS = ["naive", "experience_replay", "target_network", "both"]
N_RUNS = 5
SMOOTHING_WINDOW = 100        # Tune this for averaging width
# -----------------------------------------------------

def compute_normalized_aoc(timesteps, mean_returns):
    """
    Compute the Area Under the Curve (AOC) using the trapezoidal rule,
    then normalize by (500 * timesteps[-1]).
    """
    if len(timesteps) == 0:
        return 0.0
    aoc = np.trapz(mean_returns, x=timesteps)
    max_possible_area = 500 * timesteps[-1]
    if max_possible_area == 0:
        return 0.0
    return aoc / max_possible_area

def gather_run_data(config):
    """
    Gather all runs for a given config, returning:
        timesteps, mean_returns, std_returns, number_of_runs_used
    or None if no run data was found.

    We assume each run file is named <config>_runN.csv, with columns:
        timesteps, return
    """
    run_returns = []
    run_timesteps = None

    for i in range(1, N_RUNS + 1):
        run_filename = f"{config}_run{i}.csv"
        csv_path = os.path.join(RESULTS_DIR, run_filename)
        if not os.path.exists(csv_path):
            # Skip missing run files
            continue

        df = pd.read_csv(csv_path)
        if run_timesteps is None:
            run_timesteps = df["timesteps"].to_numpy()
        # We assume timesteps are consistent across runs
        run_returns.append(df["return"].to_numpy())

    if len(run_returns) == 0:
        return None

    # Convert to np array: shape => (#runs, #timesteps)
    run_returns_array = np.array(run_returns)
    mean_returns = np.mean(run_returns_array, axis=0)
    std_returns = np.std(run_returns_array, axis=0)

    return run_timesteps, mean_returns, std_returns, len(run_returns)

def smooth_data(timesteps, mean_vals, std_vals, window_size=3):
    """
    Smooths the data by taking blocks of 'window_size'.
    For each block [i, i+1, ..., i+window_size-1]:
      - The new x (timestep) is the "middle" point's timestep
      - The new y (mean, std) is the average of those points in the block
    If the data length isn't divisible by window_size, the last incomplete block is dropped.
    """
    new_ts = []
    new_mean = []
    new_std = []
    for i in range(0, len(timesteps), window_size):
        block_ts = timesteps[i : i + window_size]
        block_mean = mean_vals[i : i + window_size]
        block_std = std_vals[i : i + window_size]
        if len(block_ts) < window_size:
            break
        middle_t = block_ts[window_size // 2]  # e.g. index=1 for window_size=3
        avg_mean = np.mean(block_mean)
        avg_std = np.mean(block_std)

        new_ts.append(middle_t)
        new_mean.append(avg_mean)
        new_std.append(avg_std)

    return np.array(new_ts), np.array(new_mean), np.array(new_std)

def plot_individual_config(config, smoothed_ts, smoothed_mean, smoothed_std, n_runs, original_aoc):
    """
    Plot the smoothed data for a single configuration:
      - Blue line for the smoothed mean
      - Gray band for ±1 smoothed standard deviation
      - Title includes AOC from the original (unsmoothed) data
    """
    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_ts, smoothed_mean, label="Mean Return (Smoothed)", color="blue")
    plt.fill_between(
        smoothed_ts,
        smoothed_mean - smoothed_std,
        smoothed_mean + smoothed_std,
        color="gray",
        alpha=0.3,
        label="±1 STD (Smoothed)"
    )
    plt.xlabel("Timesteps")
    plt.ylabel("Evaluation Return")
    plt.title(
        f"{config.title().replace('_',' ')} - "
        f"Mean Return over {n_runs} runs\n(AOC on Original: {original_aoc:.2f})"
    )
    plt.legend()
    plt.tight_layout()

    # Save to the PLOTS_DIR
    output_file = os.path.join(PLOTS_DIR, f"{config}_plot.png")
    plt.savefig(output_file)
    plt.close()
    print(f"Saved individual plot for {config} to {output_file}")

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    comparison_data = {}
    comparison_ts = None

    for config in CONFIGS:
        # 1. Gather run data for this config (original)
        gathered = gather_run_data(config)
        if gathered is None:
            print(f"No run data found for config '{config}'. Skipping.")
            continue

        timesteps, mean_returns, std_returns, n_runs = gathered

        # 2. Compute the AOC on the original data (no smoothing)
        original_aoc = compute_normalized_aoc(timesteps, mean_returns)

        # 3. Smooth the data for plotting
        smoothed_ts, smoothed_mean, smoothed_std = smooth_data(
            timesteps, mean_returns, std_returns, window_size=SMOOTHING_WINDOW
        )

        # 4. Save the smoothed data to CSV
        smoothed_csv = os.path.join(PLOTS_DIR, f"{config}_smoothed.csv")
        df_smoothed = pd.DataFrame({
            "timesteps": smoothed_ts,
            "mean_return": smoothed_mean,
            "std_return": smoothed_std
        })
        df_smoothed.to_csv(smoothed_csv, index=False)
        print(f"Saved smoothed CSV for {config} to {smoothed_csv}")

        # 5. Plot the individual config (smoothed data) but with AOC from original data
        plot_individual_config(config, smoothed_ts, smoothed_mean, smoothed_std, n_runs, original_aoc)

        # For the final comparison plot, store the smoothed mean returns
        comparison_data[config] = smoothed_mean
        if comparison_ts is None:
            comparison_ts = smoothed_ts

    # 6. Create a final comparison plot if we have data
    if comparison_data and comparison_ts is not None:
        plt.figure(figsize=(10, 6))
        for config, mean_vals in comparison_data.items():
            plt.plot(comparison_ts, mean_vals, label=config.title().replace('_',' '))

        plt.xlabel("Timesteps")
        plt.ylabel("Evaluation Return")
        plt.title("Comparison of Smoothed Mean Evaluation Returns")
        plt.legend()
        plt.tight_layout()
        output_file = os.path.join(PLOTS_DIR, "comparison_plot.png")
        plt.savefig(output_file)
        plt.close()
        print(f"Saved final comparison plot to {output_file}")
    else:
        print("No valid data to create a comparison plot.")

if __name__ == "__main__":
    main()
