"""
QRC Ladder Configuration Script
===============================
Author: Marcin Plodzien

This module serves as the **Control Center** for the QRC Simulation.
It defines the experimental parameters, physics settings, and the grid search hypercube.

Mechanism:
----------
1.  **Definitions**: Constants define the search space (e.g., `J_rungs`, `T_EVOL_VALUES`).
2.  **Generator**: The `generate_configs()` function computes the Cartesian product of all parameters.
3.  **Consumption**: The Runner (`00_runner_parallel_CPU.py`) imports this module and iterates 
    over the list returned by `generate_configs()` to dispatch jobs.

Critical Parameters:
--------------------
- `CONFIG_NAME`: Defines the root directory for results (`results/{CONFIG_NAME}`).
- `INTEGRATION_METHOD`: Selects the solver ('trotter', 'exact_eig', 'rk4_dense', 'rk4_sparse').
- `OBSERVABLES`: List of strings defining what to measure (passed to `utils.engine`).

"""

import itertools
import numpy as np
import utils.hamiltonians as uh

# ==============================================================================
# 1. EXPERIMENT IDENTITY & ORCHESTRATION
# ==============================================================================
# This name determines the output folder: results/{CONFIG_NAME}/
CONFIG_NAME = "QRC_Ladder_Validation_Run_N_06_TFIM_ZZ_X"

# Number of Parallel Workers for the Runner (ProcessPoolExecutor)
# Set to 1 for Serial Execution (Debugging), or >1 for Parallel.
MAX_WORKERS = 10

# ==============================================================================
# 2. PHYSICS CORE SETTINGS
# ==============================================================================

# Time Integration Strategy
# Options:
# - 'trotter':    Suzuki-Trotter decomposition (1st Order). Fast, memory efficient. 
#                 Best for large N. Error ~ O(dt).
# - 'exact_eig':  Exact Diagonalization. Computes full eigendecomp. 
#                 Best for small N (<= 12). Precision benchmark.
# - 'rk4_dense':  Runge-Kutta 4 (Dense Matrix). Good for N <= 12 with time-dependent H.
# - 'rk4_sparse': Runge-Kutta 4 (Sparse BCOO). Good for N > 12 intermediate scale.
INTEGRATION_METHOD = 'exact_eig'

# Disorder Scaling
# J_LADDER_MID/SCALE are multipliers for J values (if needed). Currently 1.0.
J_LADDER_MID = 1.0   
J_LADDER_SCALE = 1.0 

# Observables to Measure
# Defines the list of operators calculated at each time step.
# - *_local_mean: Vector of values per site.
# - *_total_mean: Scalar average.
# - *_total_std: Scalar standard deviation (fluctuation).
# - 'Norm': Trace(rho) to check conservation of probability.
OBSERVABLES = [
    'Z_local_mean', 'X_local_mean', 'Y_local_mean',
    'Z_total_mean', 'X_total_mean', 'Y_total_mean',
    'Z_total_std',  'X_total_std',  'Y_total_std',
    'Z_local_std',  'X_local_std',  'Y_local_std',
    'ZZ_local_mean', 'XX_local_mean', 'YY_local_mean',
    'ZZ_total_mean', 'XX_total_mean', 'YY_total_mean',
    'ZZ_local_std',  'XX_local_std',  'YY_local_std'
]

# ==============================================================================
# 3. HYPERPARAMETER GRID SEARCH
# ==============================================================================

# A. COUPLINGS (J)
# Format: List of dictionaries defining coupling sets to sweep.
# 'couplings': (Jx, Jy, Jz) tuple.

# 1. Rung Couplings (Inter-rail)
J_rungs = [
    #{"name": "XX",          "couplings": (1.0, 1.0, 0.0)}, 
     {"name": "ZZ",          "couplings": (0.0, 0.0, 1.0)}, 
    #{"name": "Heisenberg",  "couplings": (1.0, 1.0, 1.0)} 
]

# 2. Left Rail Couplings (Intra-rail)
J_rail_left = [
   # {"name": "XX",   "couplings": (1.0, 1.0, 0.0)},
   # {"name": "ZZ",   "couplings": (0.0, 0.0, 1.0)},
   {"name": "ZZ", "couplings": (0.0, 0.0, 1.0)}, 
#{"name": "Heisenberg", "couplings": (1.0, 1.0, 1.0)}, 
   {"name": "None", "couplings": (0.0, 0.0, 0.0)} 
]

# 3. Right Rail Couplings (Intra-rail)
J_rail_right = [
   {"name": "ZZ", "couplings": (0.0, 0.0, 1.0)}, 
#{"name": "Heisenberg", "couplings": (1.0, 1.0, 1.0)}, 
 
]

# B. EXTERNAL FIELDS (Disorder Directions)
# Defines the VECTOR direction of the local random fields h_i.
# Magnitude is controlled by 'h_mag' (fixed to 1.0 below).
# Currently X-field implies h_i * sigma_x.

h_field_rail_left = [
    {"name": "X", "couplings": (1.0, 0.0, 0.0)}, # Field along X
]

h_field_rail_right = [
    {"name": "X", "couplings": (1.0, 0.0, 0.0)},
]

# C. SYSTEM SIZES
# Total Spins N. Ladder Length L = N / 2.
N_TOTAL_SPINS = [6]

# D. TIME EVOLUTION
# Evolution time per input injection step.
T_EVOL_VALUES = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

# Number of integration sub-steps (dt = t_evol / steps).
N_STEPS_PER_T_EVOL = 100

 # Cartesian Product of All Parameters
FIELD_DISORDER = [True, False] # Toggle Random vs Uniform Fields
    

# E. REALIZATIONS
# Number of random seeds to run per configuration.
N_REALIZATIONS_UNIFORM = 1     # Uniform fields don't need averaging
N_REALIZATIONS_DISORDER = 20   # Random fields need meaningful statistics
REALIZATION_SHIFT = 0 # Offset for seed (useful for resuming/extending batches)
SEED_BASE = 42

# F. TASK SETTINGS
INPUT_STATES = ['product'] # Input encoding: Product state vs GHZ-like
DATA_INPUT_TYPES = ['single_data_input', 'batch_data_input'] # Placeholder for future data modes
PREPARE_MEASUREMENT = True

# Target Dataset path (Santa Fe, etc.)
TARGET_DATASET = './datasets/santafe.txt'

# Output Directory
RESULTS_DIR = f"results/{CONFIG_NAME}"

# ==============================================================================
# 4. CONFIGURATION GENERATOR
# ==============================================================================
def generate_configs():
    """
    Generates the list of all configuration dictionaries to be run.
    
    Logic:
    1.  Grid Search: itertools.product over all parameter lists.
    2.  Filtering: Skip invalid configurations (e.g. Odd N for Ladder).
    3.  Construction: Build the 'ham_config' dictionary required by uqc.construct_hamiltonian_*.
    4.  Naming: Create a unique 'name' string for file saving.
    
    Returns:
        list[dict]: List of configuration dictionaries.
    """
    configs = []
    
    # Load Templates
    library = {h['config_name']: h for h in uh.get_hamiltonian_library()}
    ladder_template = library.get('Top_Ladder', None)
    
    if not ladder_template:
        raise ValueError("Could not find Top_Ladder template in utils.hamiltonians.")
    
   
    ladder_grid = itertools.product(
        T_EVOL_VALUES, N_TOTAL_SPINS, INPUT_STATES, DATA_INPUT_TYPES,
        J_rungs, J_rail_left, J_rail_right,
        h_field_rail_left, h_field_rail_right,
        FIELD_DISORDER
    )
    
    for (t_evol, N, state, dmode, j_rung_conf, j_rail_left_conf, j_rail_right_conf, fL_conf, fR_conf, is_disordered) in ladder_grid:
        if N % 2 != 0: continue # Skip if N is odd (Ladder requires pairs)
        
        # Extract Coupling Values and Names
        j_rung_val = j_rung_conf["couplings"]; j_rung_name = j_rung_conf["name"]
        j_rail_left_val = j_rail_left_conf["couplings"]; j_rail_left_name = j_rail_left_conf["name"]
        j_rail_right_val = j_rail_right_conf["couplings"]; j_rail_right_name = j_rail_right_conf["name"]
        
        fL_vec = fL_conf["couplings"]; fL_name = fL_conf["name"]
        fR_vec = fR_conf["couplings"]; fR_name = fR_conf["name"]
        
        field_suffix = f"FL{fL_name}_FR{fR_name}"
        
        # Build Hamiltonian Configuration Dictionary
        # This dict matches the arguments of uqc.construct_hamiltonian_ladder
        ham_config = ladder_template.copy()
        ham_config.update({
            'config_name': f"Ladder_JrL{j_rail_left_name}_JrR{j_rail_right_name}_Rung{j_rung_name}_{field_suffix}",
            'J_rail_left': j_rail_left_val,
            'J_rail_right': j_rail_right_val,
            'J_rung': j_rung_val,
            'field_L': fL_vec,
            'field_R': fR_vec
        })
        # Remove 'field' key from template to force runner to use field_L/R
        ham_config.pop('field', None)
        
        # Unique Run Name
        dis_str = "Dis" if is_disordered else "Uni" # Disorder vs Uniform
        config_name = f"Ladder_N{N}_T{t_evol}_JrL{j_rail_left_name}_JrR{j_rail_right_name}_{field_suffix}_{state}_{dmode}_{dis_str}"
        
        # Determine Realizations
        n_realizations = N_REALIZATIONS_DISORDER if is_disordered else N_REALIZATIONS_UNIFORM

        # Append Full Configuration Object
        configs.append({
            'name': config_name,
            'config_name_batch': CONFIG_NAME,
            'ham_config': ham_config,
            't_evol': float(t_evol), 
            'dt': float(t_evol) / N_STEPS_PER_T_EVOL, # Derived dt
            'h_mag': 1.0, # Fixed disorder magnitude
            'field_disorder': is_disordered, # Passed to engine
            'N': N, 'L': N // 2,
            'J_ladder_mid': J_LADDER_MID, 'J_ladder_scale': J_LADDER_SCALE,
            'input_state_type': state, 'data_input_type': dmode,
            'realization_start': REALIZATION_SHIFT, 'n_realizations': n_realizations, 'seed_base': SEED_BASE,
            'target_dataset': TARGET_DATASET,
            'integration_method': INTEGRATION_METHOD,
            'observables': OBSERVABLES,
            'prepare_measurement': PREPARE_MEASUREMENT,
            'output_dir': f"{RESULTS_DIR}/{ham_config['config_name']}_N{N}/data",
            # Names for Filename Generation
            'param_names': {
                'J_rungs': j_rung_name,
                'J_rail_left': j_rail_left_name,
                'J_rail_right': j_rail_right_name,
                'field_L': fL_name,
                'field_R': fR_name,
                'field_disorder': str(is_disordered)
            }
        })
        
    return configs
