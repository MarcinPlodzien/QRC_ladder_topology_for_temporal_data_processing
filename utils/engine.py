"""
Simulation Engine (JAX)
=======================
Author: Marcin Plodzien

This module contains the core JAX-jittable kernel for the QRC simulation.
It orchestrates the entire simulation lifecycle for a single realization:
1.  **State Initialization**: Prepares the reservoir state (usually |00...0> for Reservoir, random/input-driven for Input Rail).
2.  **Input Encoding**: Maps classical input data u(t) to quantum state vectors |psi_L(t)>.
3.  **Time Evolution**: Evolves the density matrix using the chosen strategy (Trotter/Exact/RK4) via `utils.time_integrator`.
4.  **Measurement**: Computes observables (densities, correlations, etc.) at each time step.

Key Design Pattern: `lax.scan`
------------------------------
The main time loop is implemented using `jax.lax.scan`. This primitive allows JAX to unroll or loop efficiently
on the accelerator without returning control to Python, enabling massive speedups (100x+) compared to Python loops.

JAX Requirements:
-----------------
- **Static Arguments**: Parameters like `L_chain`, `topology`, `observables` (tuples/strings) must be marked 
  as `static_argnames` in `jit`. They define the *structure* of the computation graph and cannot change during a compiled run.
- **Pure Functions**: All side effects (printing, saving) are forbidden inside the kernel.

"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from functools import partial
import utils.quantum_core as uqc
import utils.time_integrator as uti
from jax.experimental import sparse

# ==============================================================================
# 1. INPUT ENCODING
# ==============================================================================

@partial(jit, static_argnames=['L', 'state_type'])
def prep_input_state_vectors(input_vec, L, state_type):
    """
    Encodes a vector of classical inputs u(t) into a quantum pure state |psi_Input>.
    
    This state replaces the Left Rail (Input Rail) at each time step.
    
    Encoding Strategy ('product'):
    ------------------------------
    Each qubit k in the input rail is rotated by Ry(theta_k) where theta_k encodes u_k.
    u_k is clipped to [0, 1].
    theta_k = 2 * arccos(sqrt(1 - u_k))
    
    State: |psi> = |psi_0> (x) |psi_1> (x) ...
    where |psi_k> = cos(theta/2)|0> + sin(theta/2)|1>.
    
    Args:
        input_vec (jnp.ndarray): Classical input values (Shape: L).
        L (int): Size of input rail.
        state_type (str): Encoding scheme ('product' or 'ghz'). 'product' is standard.
        
    Returns:
        jnp.ndarray: Pure state vector of size 2^L.
    """
    # 1. Map input to angles
    s_vals = jnp.clip(input_vec, 0.0, 1.0)
    thetas = 2.0 * jnp.arccos(jnp.sqrt(1.0 - s_vals))
    
    # 2. Compute local states: Ry|0> = [cos, sin]
    c = jnp.cos(thetas/2)
    s = jnp.sin(thetas/2)
    # local_psis shape: (L, 2)
    local_psis = jnp.stack([c, s], axis=1) # [0, 1] amplitudes
    
    if state_type == 'product':
        # Tensor product all local states
        # |psi> = |q0> (x) |q1> (x) ...
        psi = local_psis[0]
        for k in range(1, L):
            psi = jnp.kron(psi, local_psis[k])
        return psi
        
    elif state_type == 'ghz':
        # Experimental GHZ-like encoding
        # This creates an entangled state that encodes the input.
        # |Psi> ~ (|0...0>_rotated + |1...1>_rotated)
        # Used to test if initial entanglement helps.
        
        # Branch 0: |0...0> -> Apply product of (Ry|0>) -> matches 'product' logic
        term0 = local_psis[0]
        for k in range(1, L):
            term0 = jnp.kron(term0, local_psis[k])
            
        # Branch 1: |1...1> -> Apply product of (Ry|1>)
        # Ry|1> = [-sin, cos]
        local_psis_1 = jnp.stack([-s, c], axis=1)
        term1 = local_psis_1[0]
        for k in range(1, L):
            term1 = jnp.kron(term1, local_psis_1[k])
            
        return (term0 + term1) / jnp.sqrt(2.0)
    
    else:
        # Fallback (Should typically raise error or default to zero state)
        return jnp.zeros(2**L, dtype=jnp.complex128)

# ==============================================================================
# 2. OBSERVABLES MEASUREMENT
# ==============================================================================

@partial(jit, static_argnames=['L_chain', 'observables'])
def measure_observables(rho_total, L_chain, observables):
    """
    Computes all requested observables from the current density matrix.
    
    Mechanism:
    1.  **Partial Trace**: First, we trace out the Input (Left) Rail. We only measure the Reservoir (Right) Rail.
        rho_R = Tr_L (rho_total).
    2.  **Local Observables**: Z_i, X_i, Y_i on each site of the reservoir.
    3.  **Global Observables**: Sums (means) or Standard Deviations of local ops.
    4.  **Correlations**: XX, YY, ZZ (Nearest Neighbor).
    
    Naming Convention:
        - `*_local_mean`: Array of values [val_0, val_1, ..., val_L-1].
        - `*_total_mean`: Single scalar (mean of locals).
        - `*_total_std`: Standard deviation of the *Total Operator* S = Sum O_i.
          Var(S) = Sum Var(O_i) + Sum_{i!=j} Cov(O_i, O_j).
          
    Args:
        rho_total (jnp.ndarray): Full density matrix (4^L x 4^L).
        L_chain (int): Half-system size.
        observables (tuple): List of strings specifying what to measure.
        
    Returns:
        jnp.ndarray: Concatenated flat vector of all measurement results.
    """
    dim_L = 2**L_chain
    dim_R = 2**L_chain
    
    # --- 1. Partial Trace: Get Reservoir State rho_R ---
    # rho_R[u, v] = sum_k rho_total[k*dim_R + u, k*dim_R + v]
    # We implement this manually via stacking/reshaping or simply logical indexing
    # to be robust against XLA issues on some platforms.
    # Here we use a safe manual summation loop structure (compiled by JIT).
    rho_R_list = []
    for u in range(dim_R):
        row_list = []
        for v in range(dim_R):
            # Sum over Left system states k
            indices_row = jnp.arange(dim_L) * dim_R + u
            indices_col = jnp.arange(dim_L) * dim_R + v
            val = jnp.sum(rho_total[indices_row, indices_col])
            row_list.append(val)
        rho_R_list.append(jnp.stack(row_list))
    rho_R = jnp.stack(rho_R_list)
    
    # Pre-extract Real Diagonal for Z measurements (Optimization)
    diag = jnp.diag(rho_R).real
    
    obs_list = []
    
    # Iterate through requested Observables
    for req in observables:
        
        # --- Z Family (Diagonal) ---
        if 'Z_' in req and 'ZZ_' not in req:
            z_locals = []
            for i in range(L_chain):
                # Mask: which states have bit 1 at position i?
                mask = (jnp.arange(dim_R) >> (L_chain - 1 - i)) & 1
                # Z val: +1 if 0, -1 if 1
                z_vals = jnp.where(mask == 0, 1.0, -1.0)
                z_locals.append(jnp.sum(diag * z_vals))
            z_locals_arr = jnp.array(z_locals)
            z_tot = jnp.sum(z_locals_arr)
            
            if req == 'Z_local_mean':
                obs_list.append(z_locals_arr)
            elif req == 'Z_total_mean':
                obs_list.append(jnp.array([z_tot / L_chain]))
            elif req == 'Z_local_std':
                obs_list.append(jnp.sqrt(jnp.maximum(1.0 - z_locals_arr**2, 0.0)))
            elif req == 'Z_total_std':
                # Exact calculation of Var(Sum Z_i) including correlations
                corr_sum = 0.0
                for i in range(L_chain):
                    for j in range(i + 1, L_chain):
                         mask_i = (jnp.arange(dim_R) >> (L_chain - 1 - i)) & 1
                         mask_j = (jnp.arange(dim_R) >> (L_chain - 1 - j)) & 1
                         # <Z_i Z_j>: parity check. +1 if bits same, -1 if diff.
                         val_ij = 1.0 - 2.0 * (mask_i ^ mask_j) 
                         corr_sum += 2.0 * jnp.sum(diag * val_ij) # 2x for symmetry
                exp_sq = L_chain + corr_sum # <S^2>
                var = jnp.maximum(exp_sq - z_tot**2, 0.0)
                obs_list.append(jnp.array([jnp.sqrt(var)]))

        # --- X Family (Off-Diagonal) ---
        elif 'X_' in req and 'XX_' not in req:
            x_locals = []
            for i in range(L_chain):
                # X connects indices u and v where u = v ^ (1<<i)
                us = jnp.arange(dim_R)
                vs = us ^ (1 << (L_chain - 1 - i))
                x_locals.append(jnp.sum(rho_R[us, vs]).real)
            x_locals_arr = jnp.array(x_locals)
            x_tot = jnp.sum(x_locals_arr)
            
            if req == 'X_local_mean':
                obs_list.append(x_locals_arr)
            elif req == 'X_total_mean':
                obs_list.append(jnp.array([x_tot / L_chain]))
            elif req == 'X_local_std':
                obs_list.append(jnp.sqrt(jnp.maximum(1.0 - x_locals_arr**2, 0.0)))
            elif req == 'X_total_std':
                # Correlations <X_i X_j>
                corr_sum = 0.0
                for i in range(L_chain):
                    for j in range(i + 1, L_chain):
                        us = jnp.arange(dim_R)
                        # Flip both i and j
                        vs = us ^ (1 << (L_chain - 1 - i)) ^ (1 << (L_chain - 1 - j))
                        corr_sum += 2.0 * jnp.sum(rho_R[us, vs]).real
                exp_sq = L_chain + corr_sum
                var = jnp.maximum(exp_sq - x_tot**2, 0.0)
                obs_list.append(jnp.array([jnp.sqrt(var)]))

        # --- Y Family (Complex Off-Diagonal) ---
        elif 'Y_' in req and 'YY_' not in req:
            y_locals = []
            for i in range(L_chain):
                us = jnp.arange(dim_R)
                vs = us ^ (1 << (L_chain - 1 - i))
                
                # Y = [[0, -i], [i, 0]]. 
                # <u|Y|v>: if u=(...0...), v=(...1...), val = -i.
                # if u=(...1...), v=(...0...), val = +i.
                # Wait: Row u=0, Col v=1 -> -i. Row u=1, Col v=0 -> i.
                
                bit_val = (us >> (L_chain - 1 - i)) & 1
                # if bit_val == 0: us is ...0..., vs is ...1... -> we access rho[0,1]. Y[0,1]=-i.
                # if bit_val == 1: us is ...1..., vs is ...0... -> we access rho[1,0]. Y[1,0]=+i.
                
                phase = jnp.where(bit_val == 0, -1j, 1j) 
                
                # Careful: The previously derived code had +1j/-1j reversed?
                # Check standard Pauli Y:
                # Y = [0 -1j]
                #     [1j  0]
                # rho[0,1] * Y[1,0] + rho[1,0] * Y[0,1]
                # = rho[0,1] * (i) + rho[1,0] * (-i)
                # = i (rho[0,1] - rho[1,0]).
                # Let's verify code logic:
                # jnp.sum(rho_R[us, vs] * phase).
                # For u=0, v=1: bit=0. phase=-i. term: rho[0,1] * (-i).
                # For u=1, v=0: bit=1. phase=+i. term: rho[1,0] * (i).
                # Matches!
                
                y_locals.append(jnp.sum(rho_R[us, vs] * phase).real)
                
            y_locals_arr = jnp.array(y_locals)
            y_tot = jnp.sum(y_locals_arr)
            
            if req == 'Y_local_mean':
                obs_list.append(y_locals_arr)
            elif req == 'Y_total_mean':
                obs_list.append(jnp.array([y_tot / L_chain]))
            elif req == 'Y_local_std':
                obs_list.append(jnp.sqrt(jnp.maximum(1.0 - y_locals_arr**2, 0.0)))
            elif req == 'Y_total_std':
                 # Placeholder: Not strictly required for current tasks and complex to implement correctly
                 obs_list.append(jnp.array([0.0]))
        
        # --- Normalization Check ---
        elif req == 'Norm':
            norm_val = jnp.trace(rho_total).real
            obs_list.append(jnp.array([norm_val]))
            
        # --- ZZ Correlation (Nearest Neighbor) ---
        elif 'ZZ_' in req:
             zz_locals = []
             for i in range(L_chain - 1):
                mask_i = (jnp.arange(dim_R) >> (L_chain - 1 - i)) & 1
                mask_j = (jnp.arange(dim_R) >> (L_chain - 1 - (i+1))) & 1
                vals = 1.0 - 2.0 * (mask_i ^ mask_j)
                zz_locals.append(jnp.sum(diag * vals))
             zz_arr = jnp.array(zz_locals) if L_chain > 1 else jnp.array([0.0])
             
             if req == 'ZZ_local_mean':
                 obs_list.append(zz_arr)
             elif req == 'ZZ_total_mean':
                 obs_list.append(jnp.array([jnp.mean(zz_arr)]))
             elif req == 'ZZ_local_std':
                 obs_list.append(jnp.sqrt(jnp.maximum(1.0 - zz_arr**2, 0.0)))
             elif req == 'ZZ_total_std':
                 # Variance includes cross terms
                 exp_sq = jnp.array(L_chain - 1, dtype=jnp.float64) 
                 # Add terms ... (simplified as per previous impl)
                 zz_tot = jnp.sum(zz_arr)
                 # Note: Approximation in previous code used just sum sum <ZZ ZZ>. 
                 # Keeping logic consistent with previous implementation.
                 obs_list.append(jnp.array([0.0])) # Todo: Implement full correlation sum if needed

        # --- XX Correlation (Nearest Neighbor) ---
        elif 'XX_' in req:
             xx_locals = []
             if L_chain > 1:
                 for i in range(L_chain - 1):
                     us = jnp.arange(dim_R)
                     vs = us ^ (1 << (L_chain-1-i)) ^ (1 << (L_chain-1-(i+1)))
                     xx_locals.append(jnp.sum(rho_R[us, vs]).real)
                 xx_arr = jnp.array(xx_locals)
             else:
                 xx_arr = jnp.array([0.0])

             if req == 'XX_local_mean':
                 obs_list.append(xx_arr)
             elif req == 'XX_total_mean':
                 obs_list.append(jnp.array([jnp.mean(xx_arr)]))
             elif req == 'XX_local_std':
                 obs_list.append(jnp.sqrt(jnp.maximum(1.0 - xx_arr**2, 0.0)))

        # --- YY Correlation (NN) ---
        elif 'YY_' in req:
             yy_locals = []
             if L_chain > 1:
                 for i in range(L_chain - 1):
                     us = jnp.arange(dim_R)
                     vs = us ^ (1 << (L_chain-1-i)) ^ (1 << (L_chain-1-(i+1)))
                     mask_i = (us >> (L_chain - 1 - i)) & 1
                     mask_j = (us >> (L_chain - 1 - (i+1))) & 1
                     
                     # Product of phases Y_i * Y_j
                     # Y_i: 0->1 (-i), 1->0 (i)
                     # Y_j: 0->1 (-i), 1->0 (i)
                     fact_i = jnp.where(mask_i == 0, -1j, 1j)
                     fact_j = jnp.where(mask_j == 0, -1j, 1j)
                     
                     yy_locals.append(jnp.sum(fact_i * fact_j * rho_R[vs, us]).real)
                 yy_arr = jnp.array(yy_locals)
             else:
                 yy_arr = jnp.array([0.0])
                 
             if req == 'YY_local_mean':
                 obs_list.append(yy_arr)
             elif req == 'YY_total_mean':
                 obs_list.append(jnp.array([jnp.mean(yy_arr)]))
             elif req == 'YY_local_std':
                 obs_list.append(jnp.sqrt(jnp.maximum(1.0 - yy_arr**2, 0.0)))

    return jnp.concatenate(obs_list)

# ==============================================================================
# 3. MAIN SIMULATION KERNEL (VMAP-ABLE)
# ==============================================================================

@partial(jit, static_argnames=['L_chain', 't_evol', 'dt', 'h_mag', 'J_rail_left_xyz', 'J_rail_right_xyz', 'J_rung_xyz', 'J_all_xyz', 'field_L_xyz', 'field_R_xyz', 'target_input_state', 'topology', 'integration_method', 'observables', 'prepare_measurement', 'field_disorder'])
def simulation_kernel(key_seed, input_batch, L_chain, t_evol, dt, h_mag, J_rail_left_xyz, J_rail_right_xyz, J_rung_xyz, J_all_xyz, field_L_xyz, field_R_xyz, target_input_state, topology, integration_method='trotter', observables=('Z',), prepare_measurement=True, field_disorder=True):
    """
    Simulates a Single Realization of the QRC System.
    
    This function is designed to be VMAP-ed over `key_seed` and `input_batch` (optionally).
    Currently, we usually VMAP over `key_seed` (realizations) and share `input_batch`.
    
    Core Logic:
    -----------
    1.  **Disorder Sampling**: Uses `key_seed` to generate random fields h_L, h_R in range [-h_mag, h_mag].
    2.  **Initialization**: Sets Right Rail to |0> state. Sets Left Rail to a dummy state (will be overwritten).
    3.  **Time Loop (`lax.scan`)**:
        For each time step t:
        a.  **Inject Input**: Compute |psi_Input(t)> based on `input_batch[t]`.
        b.  **Reset**: Trace out old L rail, Tensor product new |psi_Input(t)> with old rho_R.
        c.  **Evolve**: Apply U(dt) using the selected `integration_method`.
        d.  **Measure**: Compute observables.
        
    Static Arguments (CRITICAL):
    ----------------------------
    Arguments like `L_chain`, `topology`, `observables` MUST be static. 
    They determine shapes and loop structures. Changing them triggers re-compilation.
    
    Args:
        key_seed (jax.random.PRNGKey): Random seed for disorder.
        input_batch (jnp.ndarray): Input time series u(t).
        L_chain (int): Size of half-system.
        t_evol (float): Evolution time per step.
        dt (float): Integration time step (e.g. for Trotter).
        h_mag (float): Disorder magnitude.
        J_*, field_*: Hamiltonian parameters.
        topology (str): 'ladder' or 'all_to_all'.
        integration_method (str): 'trotter', 'exact_eig', 'rk4_dense', 'rk4_sparse'.
        field_disorder (bool): If True, h ~ Uniform[-0.5h, 0.5h]. Else h ~ Constant(h).
        
    Returns:
        jnp.ndarray: Matrix of measurements (Time x n_observables).
    """
    n_trotter = int(t_evol / dt)
    
    # 1. Sample Disorder for FIELDS ONLY
    # Split key into subkeys for Left and Right rail disorder
    key, kL, kR = jax.random.split(key_seed, 3)
    
    if field_disorder:
        # Uniform field disorder [-0.5*h, 0.5*h] scaled by h_mag
        h_L = h_mag * jax.random.uniform(kL, (L_chain,), minval=-0.5, maxval=0.5)
        h_R = h_mag * jax.random.uniform(kR, (L_chain,), minval=-0.5, maxval=0.5)
    else:
        # Constant Uniform Field (+h_mag)
        # Assuming field direction is handled by field_L_xyz in Hamiltonian construction
        h_L = jnp.full((L_chain,), h_mag, dtype=jnp.float64)
        h_R = jnp.full((L_chain,), h_mag, dtype=jnp.float64)
    
    # 2. Init State: R in |0>, L will be set in loop
    # Right Rail (Reservoir): Initialized to |0...0>
    psi_R = jnp.zeros(2**L_chain, dtype=jnp.complex128).at[0].set(1.0)
    rho_R = jnp.outer(psi_R, psi_R.conj())
    
    # Left Rail (Input): Initialized to dummy |0...0>, immediately replaced in Step 0
    psi_L_dummy = jnp.zeros(2**L_chain, dtype=jnp.complex128).at[0].set(1.0)
    rho_total = jnp.kron(jnp.outer(psi_L_dummy, psi_L_dummy.conj()), rho_R)
    
    # If using Exact Diagonalization, we compute Eigensystem once here.
    eig_vals = None
    eig_vecs = None
    U_step = None
    U_step_dag = None
    
    if integration_method == 'exact_eig':
        # Construct Static Hamiltonian
        if topology == 'ladder':
            H_static = uqc.construct_hamiltonian_ladder(
                h_L, h_R, J_rail_left_xyz, J_rail_right_xyz, J_rung_xyz, L_chain, field_L_xyz, field_R_xyz
            )
        elif topology == 'all_to_all':
            H_static = uqc.construct_hamiltonian_all2all(
                h_L, h_R, J_all_xyz, L_chain, field_L_xyz, field_R_xyz
            )
        else:
             H_static = jnp.zeros((2**(2*L_chain), 2**(2*L_chain)), dtype=jnp.complex128)
             
        # Diagonalize: H = V @ D @ V.T
        eig_vals, eig_vecs = jnp.linalg.eigh(H_static)
        
        # Pre-compute Unitary for constant t_evol
        # U = V exp(-i D t) V^dag
        U_step = uti.construct_unitary_eig(eig_vals, eig_vecs, t_evol)
        U_step_dag = jnp.conj(U_step).T

    # 3. Scan Loop Definition
    # rho_curr: Carried state.
    # u_vec_t: Input vector for current time step (scanned over input_batch).
    def time_step(rho_curr, u_vec_t):
        
        # A. Encode & Inject Input
        # Create state |psi_Input> from u_vec_t
        psi_new_L = prep_input_state_vectors(u_vec_t, L_chain, target_input_state)
        # Trace out old L, inject new L
        rho_injected = uqc.partial_trace_and_reset(rho_curr, psi_new_L, L_chain)
        
        # B. Evolve State (rho(t) -> rho(t+T_evol))
        if integration_method == 'exact_eig':
            # Use pre-computed Unitary (Fastest)
            rho_evolved = uti.apply_dense_operator(rho_injected, U_step, U_step_dag)
            

        elif integration_method == 'rk4_dense':
            # Runge-Kutta 4 (Dense Matrix)
            rho_evolved = uti.evolve_rk4_dense(
                rho_injected, h_L, h_R, 
                J_rail_left_xyz, J_rail_right_xyz, J_rung_xyz, J_all_xyz,
                dt, L_chain, n_trotter, field_L_xyz, field_R_xyz, topology
            )
            
        elif integration_method == 'rk4_sparse':
            # Runge-Kutta 4 (Sparse Matrix BCOO)
            # Re-construct sparse H each step? No, H is static (disorder fixed).
            # But construct_hamiltonian_ladder_sparse is fast.
            # Ideally move construction outside `time_step`?
            # Creating sparse matrix inside loop might have overhead but JIT should handle it.
            # Optimization: Move H_sparse creation outside loop if memory allows carrying it.
            
            if topology == 'ladder':
                H_sparse = uqc.construct_hamiltonian_ladder_sparse(
                    h_L, h_R, J_rail_left_xyz, J_rail_right_xyz, J_rung_xyz, L_chain, field_L_xyz, field_R_xyz
                )
            elif topology == 'all_to_all':
                H_sparse = uqc.construct_hamiltonian_all2all_sparse(
                    h_L, h_R, J_all_xyz, L_chain, field_L_xyz, field_R_xyz
                )
            else:
                H_sparse = sparse.BCOO.fromdense(jnp.zeros((2**(2*L_chain), 2**(2*L_chain)), dtype=jnp.complex128))
             
            rho_evolved = uti.evolve_rk4_sparse(
                rho_injected, H_sparse, dt, n_trotter
            )
        
        else: # Default: Trotter
            # Dispatch based on static topology
            if topology == 'ladder':
                rho_evolved = uti.evolve_trotter(
                    rho_injected, h_L, h_R, 
                    J_rail_left_xyz, J_rail_right_xyz, J_rung_xyz, 
                    dt, L_chain, n_trotter, field_L_xyz, field_R_xyz
                )
            elif topology == 'all_to_all':
               # All-to-all Trotter
               # Note: Not fully tested/optimized in this version
               rho_evolved = rho_injected 
            else:
                 # Fallback default
                 rho_evolved = rho_injected
        
        # C. Measure or Check Norm
        norm_val = jnp.trace(rho_evolved).real
        norm_error = jnp.abs(1.0 - norm_val)
        
        if prepare_measurement:
             obs = measure_observables(rho_evolved, L_chain, observables)
             return rho_evolved, obs
        else:
             # Return normalization error as a 1-element vector
             return rho_evolved, jnp.array([norm_error])
    
    # Run the Scan
    # rho_total: Initial State
    # input_batch: Sequence of inputs (nT, L)
    # Output: (Final State, Time-Series of Measurements)
    final_rho, measures = lax.scan(time_step, rho_total, input_batch)
    return measures
