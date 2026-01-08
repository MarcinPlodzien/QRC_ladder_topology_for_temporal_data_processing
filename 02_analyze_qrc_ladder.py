"""
QRC Analysis Script: Capacity Calculation
=========================================
Author: Marcin Plodzien

This script processes the simulation results (collected pickle files) and computes the 
Memory/Prediction Capacity of the quantum reservoir.

Methodology:
------------
1.  **Data Loading**: Reads `collected_simulation_results.pkl`.
2.  **Grouping**: Groups data by hyperparameters (T_evol, h_mag, Topology, etc.).
3.  **Capacity Metric**:
    For each time lag `tau`:
    -   Splits reservoir state history X(t) and target signal y(t) into Train/Test sets.
    -   Fits a linear readout weights W via Ridge Regression: y_train = W * X_train.
    -   Predicts on test set: y_pred = W * X_test.
    -   Computes Squared Correlation: C(tau) = corr(y_test, y_pred)^2.
    
    Total Capacity C_total = Sum_{tau} C(tau).

4.  **Task Types**:
    -   **Memory**: Reconstruct past input u(t - tau).
    -   **Prediction**: Predict future input u(t + tau).
    -   (Note: For chaotic series like Santa Fe, 'prediction' is standard).

"""

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.linear_model import Ridge
from joblib import Parallel, delayed
import time

# ==============================================================================
# 1. CONFIGURATION
import sys
import importlib.util

# ==============================================================================
# SPYDER / IDE CONFIGURATION
# ==============================================================================
# Set this to a path string (or config file) to run directly without CLI args
MANUAL_PATH = None 
# Example: 
MANUAL_PATH = "results/QRC_Ladder_Validation_Run_N_06_TFIM_ZZ_X"

# ==============================================================================
# LOAD PATH Logic
# ==============================================================================
# 1. CLI Argument -> 2. Manual Path -> 3. Default Config
arg = sys.argv[1] if len(sys.argv) > 1 else (MANUAL_PATH or "01_config_qrc_ladder.py")

if arg.endswith('.py'):
    print(f"Loading Configuration from: {arg}")
    spec = importlib.util.spec_from_file_location("config", arg)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    RESULTS_ROOT = config.RESULTS_DIR
else:
    # Assume it is a direct directory path
    RESULTS_ROOT = arg

# Auto-detect 'data' subfolder
if os.path.isdir(os.path.join(RESULTS_ROOT, "data")):
    DATA_PATH = os.path.join(RESULTS_ROOT, "data")
    print(f"Detected 'data' subfolder. Loading results from: {DATA_PATH}")
else:
    DATA_PATH = RESULTS_ROOT
    print(f"Analysis targeting root: {DATA_PATH}")

# NEW: Output to subfolder in results
OUTPUT_DIR = os.path.join(RESULTS_ROOT, "analysis_results")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

OUTPUT_FILE = f"{OUTPUT_DIR}/analysis_summary.pkl"

# Analysis Hyperparameters
TAU_MAX = 50          # Maximum lag to compute capacity for
TASK_TYPE = 'prediction' # 'memory' or 'prediction'
TRAIN_RATIO = 0.2     # First 20% used for training readout
REG_ALPHA = 1e-3      # Ridge Regression Regularization Strength

# Features Filtering
# We exclude non-observable columns from the Feature Matrix X
EXCLUDE_COLS = [
    'k', 's_k', 'Norm', 't_evol', 'h_mag', 'N', 'L', 
    'J_rung_name', 'config_name', 'input_state_type', 
    'data_input_type', 'realization', 'topology', 'dt',
    'j_rung_name', 'j_rail_left_name', 'j_rail_right_name',
    'field_disorder'
]

# ==============================================================================
# 2. CORE LOGIC
# ==============================================================================

def calculate_capacity(group_df, tau_max=20, task_type='memory', return_traces_tau=None):
    """
    Computes the capacity profile C(tau).
    Optionally returns (y_test, y_pred) for a specific tau if return_traces_tau is set.
    
    Args:
        group_df (pd.DataFrame): Time series data.
        tau_max (int): Max lag.
        task_type (str): 'memory' or 'prediction'.
        return_traces_tau (int): If set, returns traces for this specific lag alongside capacities.
        
    Returns:
        np.array: Capacities C(tau).
        (Optional) tuple: (y_test, y_pred) if return_traces_tau is set.
    """
    # 1. Prepare Features & Targets
    # Sort by time step 'k' ensures temporal order
    group_df = group_df.sort_values('k')
    
    # Extract Feature Matrix X (Time x Observables)
    feature_cols = [c for c in group_df.columns if c not in EXCLUDE_COLS]
    X = group_df[feature_cols].values
    
    # Extract Target Signal S (Time x 1)
    S = group_df['s_k'].values
    
    # Standardization (Z-score normalization)
    # Improves Ridge Regression stability
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-9
    X = (X - X_mean) / X_std
    
    n_samples = len(X)
    n_train = int(n_samples * TRAIN_RATIO)
    
    capacities = []
    collected_traces = None
    
    for tau in range(tau_max + 1):
        # 2. Construct Lagged Targets
        if task_type == 'memory':
            # Goal: Output(t) matches Input(t - tau)
            # X comes from reservoir at time t.
            # y matches input at time t - tau.
            
            if tau == 0:
                y = S
                X_curr = X
            else:
                # We align X[t] with S[t-tau]
                # Valid t range: [tau, N-1]
                X_curr = X[tau:]     # Reservoir states from t=tau to End
                y = S[:-tau]         # Inputs from t=0 to End-tau
                
        elif task_type == 'prediction':
            # Goal: Output(t) matches Input(t + tau)
            # Valid t range: [0, N-1-tau]
            
            if tau == 0:
                y = S
                X_curr = X
            else:
                # We align X[t] with S[t+tau]
                X_curr = X[:-tau]    # Reservoir states from t=0 to End-tau
                y = S[tau:]          # Inputs from t=tau to End
        
        # 3. Split Train/Test
        # Sequential split to respect time-series dependencies
        n_curr = len(X_curr)
        n_train_curr = int(n_curr * TRAIN_RATIO)
        
        X_train = X_curr[:n_train_curr]
        y_train = y[:n_train_curr]
        X_test = X_curr[n_train_curr:]
        y_test = y[n_train_curr:]
        
        # 4. Train Readout (Ridge Regression)
        # Linear map: y = W * x + b
        clf = Ridge(alpha=REG_ALPHA)
        clf.fit(X_train, y_train)
        
        # 5. Evaluate Performance
        # Metric: Squared Correlation Coefficient (R^2 in Capacity context)
        # Note: 'clf.score' returns Coefficient of Determination R^2 which can be negative.
        # Standard MC literature often defines Capacity = corr(y, y_pred)^2.
        
        y_pred = clf.predict(X_test)
        
        # Capture traces if requested
        if return_traces_tau is not None and tau == return_traces_tau:
            collected_traces = (y_test, y_pred)
        
        # Handle constant output edge case
        if np.std(y_test) < 1e-9 or np.std(y_pred) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(y_test, y_pred)[0, 1]
            
        cap_tau = corr**2
        capacities.append(cap_tau)
        
    if return_traces_tau is not None:
        return np.array(capacities), collected_traces
    return np.array(capacities)


def process_group(name, group):
    """
    Wrapper to process a group of realizations sharing the same hyperparameters.
    Averages capacity across realizations.
    """
    realizations = group['realization'].unique()
    caps_list = []
    
    # Store traces from ALL realizations to compute statistics
    all_y_preds = []
    all_y_tests = []
    TRACE_TAU = 1 # tau=1 for 1-step prediction visualization
    
    for r in realizations:
        sub_df = group[group['realization'] == r]
        # Skip incomplete realizations
        if len(sub_df) < 100: continue 
        
        # Always request traces
        caps, traces = calculate_capacity(sub_df, tau_max=TAU_MAX, task_type=TASK_TYPE, return_traces_tau=TRACE_TAU)
        
        if traces is not None:
            y_test_r, y_pred_r = traces
            all_y_tests.append(y_test_r)
            all_y_preds.append(y_pred_r)
            
        caps_list.append(caps)
    
    if not caps_list:
        return None
        
    avg_caps = np.mean(caps_list, axis=0)
    
    # Compute Total Capacity statistics across realizations
    totals_per_realization = [np.sum(c) for c in caps_list]
    total_cap_mean = np.mean(totals_per_realization)
    total_cap_std = np.std(totals_per_realization)
    
    # Compute Trace Statistics
    y_pred_mean, y_pred_std = None, None
    y_test_mean = None
    
    if all_y_preds:
        # Ensure all are same length before stacking
        min_len = min(len(p) for p in all_y_preds)
        all_y_preds_trunc = [p[:min_len] for p in all_y_preds]
        all_y_tests_trunc = [t[:min_len] for t in all_y_tests]
        
        preds_stack = np.vstack(all_y_preds_trunc)
        tests_stack = np.vstack(all_y_tests_trunc)
        
        y_pred_mean = np.mean(preds_stack, axis=0)
        y_pred_std = np.std(preds_stack, axis=0)
        y_test_mean = np.mean(tests_stack, axis=0)

    # Extract metadata from first row
    row = group.iloc[0]
    
    res = {
        't_evol': row['t_evol'],
        'h_mag': row['h_mag'],
        'input_state_type': row['input_state_type'],
        'data_input_type': row['data_input_type'],
        'topology': row['topology'],
        'caps_per_tau': avg_caps,
        'total_capacity': total_cap_mean,
        'total_capacity_std': total_cap_std,
        'n_realizations': len(realizations),
        'j_rail_left_name': row.get('j_rail_left_name', 'Unknown'),
        'j_rail_right_name': row.get('j_rail_right_name', 'Unknown'),
        'j_rung_name': row.get('j_rung_name', 'Unknown'),
        'field_disorder': row['field_disorder'], 
        'example_trace_tau': TRACE_TAU,
        'example_y_test': y_test_mean,
        'example_y_pred': y_pred_mean,
        'example_y_pred_std': y_pred_std
    }
    return res

# ==============================================================================
# 3. MAIN
# ==============================================================================

def load_data(path):
    """Loads data from single file or recursively from directory."""
    if os.path.isfile(path):
        print(f"Loading Results from File: {path}...")
        return pd.read_pickle(path)
    
    elif os.path.isdir(path):
        # 1. Check for consolidated file first
        consolidated_path = os.path.join(path, "collected_simulation_results.pkl")
        if os.path.exists(consolidated_path):
             print(f"Found consolidated results file: {consolidated_path}")
             return pd.read_pickle(consolidated_path)
             
        print(f"Loading Results recursively from Directory: {path}...")
        all_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".pkl") and "collected" not in file and "summary" not in file:
                     all_files.append(os.path.join(root, file))
        
        if not all_files:
            print("No suitable .pkl files found in directory.")
            return pd.DataFrame()
            
        print(f"Found {len(all_files)} partial result files.")
        dfs = []
        for f in all_files:
            try:
                # print(f"  Loading {f}...") # Verbose
                dfs.append(pd.read_pickle(f))
            except Exception as e:
                print(f"  Error loading {f}: {e}")
        
        if not dfs:
            return pd.DataFrame()
            
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Combined {len(combined_df)} rows from partial files.")
        return combined_df
    else:
        print(f"Error: Path {path} not found.")
        return pd.DataFrame()

def main():
    df = load_data(DATA_PATH)
    
    if df.empty:
        print("No data loaded. Exiting.")
        return

    print(f"Total Rows: {df.shape[0]}")
    
    # Detect if data is already analyzed (Capacity exists)
    if 'Capacity' in df.columns:
        print("Detected pre-computed Capacity in data. Skipping re-calculation.")
        # Filter relevant columns and save summary
        keep_cols = ['HamType', 'T_evol', 'h_mag', 'N', 'State', 'Data', 'Task', 'Tau', 'Capacity']
        keep_cols = [c for c in keep_cols if c in df.columns]
        summary_df = df[keep_cols]
        
        # Determine output path
        output_file = f"{OUTPUT_DIR}/analysis_summary_precomputed.pkl"
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            
        summary_df.to_pickle(output_file)
        print(f"Pre-computed summary saved to {output_file}")
        
        # Print sample
        if not summary_df.empty:
            print(summary_df.head())
            
        return

    # Define Grouping Keys (Hyperparameters)
    group_cols = [
        't_evol', 'h_mag', 'input_state_type', 'data_input_type', 'topology',
        'j_rung_name', 'j_rail_left_name', 'j_rail_right_name', 'field_disorder'
    ]
    group_cols = [c for c in group_cols if c in df.columns]
    
    grouped = df.groupby(group_cols)
    print(f"Identified {len(grouped)} hyperparameter configurations.")
    
    results = []
    
    # Iterate Groups
    for name, group in grouped:
        print(f"Processing Config: {name}")
        res = process_group(name, group)
        if res:
            results.append(res)
        
    # Formatting Output
    summary_df = pd.DataFrame(results)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    OUTPUT_FILE_PATH = f"{OUTPUT_DIR}/analysis_summary.pkl"
    summary_df.to_pickle(OUTPUT_FILE_PATH)
    
    print(f"Analysis complete. Summary saved to {OUTPUT_FILE_PATH}")
    if not summary_df.empty:
        print(summary_df[['t_evol', 'h_mag', 'total_capacity']].head())

if __name__ == "__main__":
    main()
