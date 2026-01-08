"""
QRC Visualization Script
========================
Author: Marcin Plodzien

This script generates standard analysis plots from the summarized results (`analysis_summary.pkl`).
It is the final step in the pipeline: Simulation -> Analysis -> Visualization.

Plots Generated:
----------------
1.  **Total Capacity vs Time Evolution** (`Capacity_vs_Time.png`):
    -   X-axis: $t_{evol}$ (Log scale).
    -   Y-axis: $\sum C(\\tau)$ (Total Memory Capacity).
    -   Hue: Data Input Type / h_mag.
    -   Style: Topology.
    -   Purpose: Identifies the optimal timescale for the reservoir.

2.  **Capacity Profiles** (`Profile_*.png`):
    -   X-axis: Lag $\\tau$.
    -   Y-axis: $C(\\tau)$.
    -   Curves: Different $t_{evol}$.
    -   Purpose: Inspects the memory fading profile (short-term vs long-term memory).

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
 
import sys
import importlib.util

import sys
import importlib.util

# ==============================================================================
# SPYDER / IDE CONFIGURATION
# ==============================================================================
# Set this to a path string (or config file) to run directly without CLI args
# MANUAL_PATH = None 
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
    RESULTS_DIR = config.RESULTS_DIR
else:
    # Assume it is a direct directory path
    RESULTS_DIR = arg

print(f"Plotting targeting root: {RESULTS_DIR}")

ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis_results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)

PLOTS_DIR = FIGURES_DIR # Output for plots

# Priority order for loading
SUMMARY_FILES = [
    f"{ANALYSIS_DIR}/analysis_summary_precomputed.pkl",
    f"{ANALYSIS_DIR}/analysis_summary.pkl"
]

def load_summary():
    """Loads the analysis summary DataFrame (Precomputed or Standard)."""
    for fpath in SUMMARY_FILES:
        if os.path.exists(fpath):
            print(f"Loading summary from {fpath}...")
            return pd.read_pickle(fpath)
            
    print("No summary file found in plots_analysis/.")
    return None

def standardize_columns(df):
    """Maps columns from Standalone format to Analysis format."""
    # Standalone -> Analysis
    mapping = {
        'HamType': 'topology',
        'T_evol': 't_evol',
        'h_mag': 'h_mag',
        'Data': 'data_input_type',
        'State': 'input_state_type', # Added mapping
        # 'Tau': 'Tau',
        # 'Capacity': 'Capacity'
    }
    df = df.rename(columns=mapping)
    return df

def plot_total_capacity_vs_t_evol(df):
    """
    Plots Total Memory Capacity vs Evolution Time.
    Handles both Precomputed (Long) and Standard (Wide) formats.
    """
    print("Plotting Total Capacity vs t_evol...")
    
    # 1. Compute Total Capacity if not present
    if 'total_capacity' not in df.columns:
        if 'Capacity' in df.columns:
            # DEBUG: Check field_disorder distribution
            if 'field_disorder' in df.columns:
                print("DEBUG: field_disorder value counts:")
                print(df['field_disorder'].value_counts())
            else:
                print("DEBUG: field_disorder column MISSING in df")

            # Aggregate Long Format: Sum(Capacity) per Group
            grp_cols = ['topology', 't_evol', 'h_mag', 'data_input_type', 'input_state_type', 'field_disorder']
            grp_cols = [c for c in grp_cols if c in df.columns]
            df_agg = df.groupby(grp_cols)['Capacity'].sum().reset_index()
            df_agg = df_agg.rename(columns={'Capacity': 'total_capacity'})
        else:
            print("Error: DataFrame missing 'total_capacity' or 'Capacity' column.")
            return
    else:
        df_agg = df

    # Data Prep
    df_agg['t_evol'] = pd.to_numeric(df_agg['t_evol'])
    df_agg = df_agg.sort_values('t_evol')
    
    # Create Composite Label
    if 'input_state_type' not in df_agg.columns: df_agg['input_state_type'] = 'Unknown'
    if 'h_mag' not in df_agg.columns: df_agg['h_mag'] = 0.0
    if 'j_rail_left_name' not in df_agg.columns: df_agg['j_rail_left_name'] = '?'
    
    if 'field_disorder' not in df_agg.columns: df_agg['field_disorder'] = True
    df_agg['DisorderType'] = df_agg['field_disorder'].apply(lambda x: 'Random' if x else 'Uniform')

    # Condition: Input State + Topology Variant (Left Rail)
    df_agg['Condition'] = df_agg.apply(lambda r: f"{r['input_state_type']} | L:{r['j_rail_left_name']}", axis=1)
    
    # Identify Data Input Types for Panels
    if 'data_input_type' not in df_agg.columns: df_agg['data_input_type'] = 'Unknown'
    dtypes = sorted(df_agg['data_input_type'].unique())
    n_panels = len(dtypes)
    
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6), sharey=True)
    if n_panels == 1: axes = [axes]
    
    for i, dtype in enumerate(dtypes):
        ax = axes[i]
        sub = df_agg[df_agg['data_input_type'] == dtype]
        
        print(f"DEBUG: Panel {dtype} DisorderTypes: {sub['DisorderType'].unique()}")

        # Manual Plotting with Error Bars
        conditions = sorted(sub['Condition'].unique())
        disorders = ['Uniform', 'Random']
        
        # Color Palette
        palette = sns.color_palette("tab10", len(conditions))
        color_map = {cond: palette[i] for i, cond in enumerate(conditions)}
        
        # Style Map
        style_map = {'Uniform': {'ls': '-', 'marker': 'o'}, 'Random': {'ls': '--', 'marker': 'x'}}
        
        for cond in conditions:
            for dis in disorders:
                mask = (sub['Condition'] == cond) & (sub['DisorderType'] == dis)
                data = sub[mask]
                
                if data.empty: continue
                
                # Check for error bars
                yerr = None
                if 'total_capacity_std' in data.columns and not data['total_capacity_std'].isna().all():
                    yerr = data['total_capacity_std']
                
                # Plot Mean Line
                ax.plot(
                    data['t_evol'], 
                    data['total_capacity'], 
                    label=f"{cond} ({dis})",
                    color=color_map[cond],
                    linestyle=style_map[dis]['ls'],
                    marker=style_map[dis]['marker'],
                    alpha=0.9
                )
                
                # Add Shaded Error Band
                if yerr is not None:
                     ax.fill_between(
                        data['t_evol'],
                        data['total_capacity'] - yerr,
                        data['total_capacity'] + yerr,
                        color=color_map[cond],
                        alpha=0.2
                     )

        ax.set_xscale('log')
        ax.grid(True, which="both", ls="-", alpha=0.5)
        ax.set_xlabel("Evolution Time ($t_{evol}$)")
        
        if i == 0:
            ax.set_ylabel("Total Prediction Capacity ($\sum C_\\tau$)")
        else:
            ax.set_ylabel("") # Hide y label for secondary panels
            
        ax.set_title(f"Data Input: {dtype}")
        
        # Legend handling: only show on last panel or outer
        if i == n_panels - 1:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            if ax.get_legend(): ax.get_legend().remove()
            
    fig.suptitle("Prediction Capacity vs Time Evolution (QRC)", fontsize=14)
    plt.tight_layout()
    
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
        
    out_path = f"{PLOTS_DIR}/Capacity_vs_Time.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.close()

def plot_capacity_profiles(df):
    """
    Plots Capacity Profile C(tau).
    Handles Long Format ('Capacity', 'Tau') or Wide Format ('caps_per_tau').
    """
    print("Plotting Capacity Profiles...")
    
    # Check Format
    is_long_format = 'Capacity' in df.columns and 'Tau' in df.columns
    
    # Identify unique experimental setups (including topology nuances)
    grouping_cols = ['data_input_type', 'input_state_type', 'h_mag', 'topology', 
                     'j_rail_left_name', 'j_rail_right_name', 'j_rung_name', 'field_disorder']
    
    # Ensure cols exist
    for c in grouping_cols:
        if c not in df.columns: df[c] = '?'
        
    unique_conditions = df[grouping_cols].drop_duplicates()
    
    for _, row in unique_conditions.iterrows():
        d_type = row['data_input_type']
        s_type = row['input_state_type']
        h_mag = row['h_mag']
        topo = row['topology']
        jrl = row['j_rail_left_name']
        jrr = row['j_rail_right_name']
        jrung = row['j_rung_name']
        is_disordered = row['field_disorder']
        dis_str = "Random" if is_disordered else "Uniform"
        
        # Filter
        sub = df[(df['data_input_type'] == d_type) & 
                 (df['input_state_type'] == s_type) &
                 (df['h_mag'] == h_mag) & 
                 (df['topology'] == topo) &
                 (df['j_rail_left_name'] == jrl) &
                 (df['j_rail_right_name'] == jrr) &
                 (df['j_rung_name'] == jrung) &
                 (df['field_disorder'] == is_disordered)]
        
        if sub.empty: continue
        
        plt.figure(figsize=(12, 8))
        
        t_vals = sorted(sub['t_evol'].unique())
        cmap = plt.cm.viridis
        color_map = {t: cmap(i/len(t_vals)) if len(t_vals) > 1 else cmap(0.5) for i, t in enumerate(t_vals)}
        
        if is_long_format:
            for t in t_vals:
                line_data = sub[sub['t_evol'] == t].sort_values('Tau')
                if not line_data.empty:
                    plt.plot(line_data['Tau'], line_data['Capacity'], marker='o', label=f"t_evol={t}", color=color_map[t])
        else:
            # Wide Format
            for idx, r in sub.iterrows():
                t = r['t_evol']
                caps = r['caps_per_tau']; taus = np.arange(len(caps))
                plt.plot(taus, caps, marker='o', label=f"t_evol={t}", color=color_map[t])
        
        plt.xlabel("Lag ($\\tau$)")
        plt.ylabel("Prediction Capacity $C_\\tau$")
        plt.title(f"Profile: {s_type} | {d_type} | h={h_mag} ({dis_str}) | {topo}\nL:{jrl} | R:{jrr} | Rung:{jrung}")
        plt.legend(title="t_evol")
        plt.grid(True)
        
        # Filename includes topology params
        fname = f"Profile_{s_type}_{d_type}_h{h_mag}_{dis_str}_{topo}_L{jrl}.png"
        plt.savefig(f"{PLOTS_DIR}/{fname}", dpi=300)
        print(f"Saved {PLOTS_DIR}/{fname}")
        plt.close()

def plot_prediction_grid(df):
    """
    Plots Grid of Prediction Traces (True vs Predicted).
    Rows: Selected t_evol (up to 4).
    Cols: Single vs Batch Data Input.
    """
    print("Plotting Prediction Grids...")
    
    # Check if we have prediction traces
    if 'example_y_test' not in df.columns:
        print("No prediction traces found (example_y_test). Skipping grid plot.")
        return

    # Grouping (excluding data_input_type which determines columns)
    grouping_cols = ['input_state_type', 'h_mag', 'topology', 
                     'j_rail_left_name', 'j_rail_right_name', 'j_rung_name', 'field_disorder']
    
    # Ensure cols exist
    for c in grouping_cols:
        if c not in df.columns: df[c] = '?'
        
    unique_groups = df[grouping_cols].drop_duplicates()
    
    # Define Data Input Types (Columns)
    dtypes = ['single_data_input', 'batch_data_input']
    
    for _, row in unique_groups.iterrows():
        s_type = row['input_state_type']
        h_mag = row['h_mag']
        topo = row['topology']
        jrl = row['j_rail_left_name']
        jrr = row['j_rail_right_name']
        jrung = row['j_rung_name']
        is_disordered = row['field_disorder']
        dis_str = "Random" if is_disordered else "Uniform"
        
        # Filter for this group
        sub_group = df[(df['input_state_type'] == s_type) &
                       (df['h_mag'] == h_mag) & 
                       (df['topology'] == topo) &
                       (df['j_rail_left_name'] == jrl) &
                       (df['j_rail_right_name'] == jrr) &
                       (df['j_rung_name'] == jrung) &
                       (df['field_disorder'] == is_disordered)]
                       
        if sub_group.empty: continue
        
        # Select 4 representative t_evols
        all_ts = sorted(sub_group['t_evol'].unique())
        if not all_ts: continue
        
        # Selection logic: First, Last, and 2 evenly spaced in between
        if len(all_ts) <= 4:
            selected_ts = all_ts
        else:
            indices = np.linspace(0, len(all_ts)-1, 4, dtype=int)
            selected_ts = [all_ts[i] for i in indices]
            
        n_rows = len(selected_ts)
        n_cols = len(dtypes)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows), sharex=True, sharey=True, squeeze=False)
        
        for r_idx, t_val in enumerate(selected_ts):
            for c_idx, dtype in enumerate(dtypes):
                ax = axes[r_idx, c_idx]
                
                # Get Row Data
                record = sub_group[(sub_group['t_evol'] == t_val) & (sub_group['data_input_type'] == dtype)]
                
                if record.empty:
                    ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                    continue
                
                # Extract Traces (Assume scalar/single row per combination)
                rec = record.iloc[0]
                y_true = rec.get('example_y_test')
                y_pred = rec.get('example_y_pred')
                tau = rec.get('example_trace_tau', '?')
                
                if y_true is None or y_pred is None:
                     ax.text(0.5, 0.5, "No Traces", ha='center', va='center')
                     continue
                     
                # Plot Snippet
                limit = 50 # Reduced limit to make markers visible
                y_std = rec.get('example_y_pred_std')
                
                ax.plot(y_true[:limit], 'k-', marker='o', markersize=4, alpha=0.6, label='True')
                ax.plot(y_pred[:limit], 'r--', marker='x', markersize=4, alpha=0.8, label='Pred')
                
                if y_std is not None:
                     ax.fill_between(
                        range(len(y_pred[:limit])),
                        (y_pred - y_std)[:limit],
                        (y_pred + y_std)[:limit],
                        color='r',
                        alpha=0.2,
                        label='Std'
                     )
                
                if r_idx == 0:
                    ax.set_title(f"{dtype}")
                if c_idx == 0:
                    ax.set_ylabel(f"t_evol={t_val}")
                
                if r_idx == n_rows - 1 and c_idx == n_cols - 1:
                    ax.legend(loc='upper right', fontsize='small')
                    
        fig.suptitle(f"Prediction (tau={tau}) | {s_type} | h={h_mag} ({dis_str}) | {topo}\nL:{jrl}", fontsize=14)
        plt.tight_layout()
        
        fname = f"Prediction_Grid_{s_type}_h{h_mag}_{dis_str}_{topo}_L{jrl}.png"
        plt.savefig(f"{PLOTS_DIR}/{fname}", dpi=300)
        print(f"Saved {PLOTS_DIR}/{fname}")
        plt.close()

def main():
    df = load_summary()
    if df is not None:
        df = standardize_columns(df)
        plot_total_capacity_vs_t_evol(df)
        plot_capacity_profiles(df)
        plot_prediction_grid(df)

if __name__ == "__main__":
    main()
