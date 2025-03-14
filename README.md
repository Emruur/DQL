# README

## Overview
This repository contains scripts for training, tuning, and analyzing Deep Q-Learning (DQL) models with various configurations. The scripts allow users to select different configurations and tuning functions via command-line arguments.

## Setup
Ensure you have Python installed along with the necessary dependencies:

```bash
pip install -r requirements.md
```

## Usage

### Running DQL Training
You can specify which configuration to run by passing an argument to the script:

```bash
python cartpole.py --config "Naive"
```

Available configurations:
- `Naive`
- `Experience Replay`
- `Target Network`
- `Both`

If no argument is provided, all configurations will run by default:

```bash
python cartpole.py
```

If all configurations are run, a comparison plot will be generated and saved in the `results/` directory.

### Running Tuning Functions
You can select a specific tuning function using the `--tune` argument:

```bash
python HPO.py --tune Target_Network
```

Available tuning functions:
- `Target_Network`
- `Replay_Buffer`
- `HPO_vanilla`

If no argument is provided, the script will exit without running any tuning.

Tuning will generate an output file that can be used for analysis.

### Running Analysis Functions
To analyze the results, specify the analysis type and provide a filename:

```bash
python analyze_hyperparams.py --analysis Replay_Buffer_Tuning --file data.json
```

Available analysis functions:
- `Replay_Buffer_Tuning`
- `vanilla`

The filename should be provided to read the necessary data for analysis.

## Directory Structure
```
project/
│-- cartpole.py              # Training script with configurable DQL options
│-- HPO.py                   # Script for tuning hyperparameters
│-- parallel_HPO.py           # Parallelized hyperparameter optimization script
│-- analyze_hyperparams.py    # Script for analyzing results
│-- Agent.py                  # Reinforcement learning agent implementation
│-- Policy.py                 # Policy handling script
│-- results/                  # Directory where result plots and logs are stored
│-- requirements.txt          # List of required dependencies
│-- README.md                 # Documentation
```

## Notes
- Ensure the required dependencies are installed before running the scripts.
- Output results and plots are saved in the `results/` directory.
- Modify configurations as needed in the respective scripts for custom experimentation.

