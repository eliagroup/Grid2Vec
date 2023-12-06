# N-1 Analysis Example

## Intro
In this folder, there are three files which can be used to create a N-1 analysis on a Grid2Vec grid.

1. `collect_nminus1_calculations.py` - this is used to iterate through the timesteps and write an N-1 calculation at each timestep to a `.npy` file
2. `process_nminus1_calculations.py` - this iterates through the files created in the previous process and then collects all of the relevant information into a `.csv` file
3. `analyze_nminus1_calulations.py` - this looks at the data in the `.csv` file created in step 2, and explores which elements are causing the most frequent issues.

## Usage
1. `collect_nminus1_calculations.py --data_path ../../data/grid_planning_scenario/ --num_procs=16 --dc`
2. `process_nminus1_calculations.py --data_path ../../data/grid_planning_scenario/ --num_procs=16 --dc`
