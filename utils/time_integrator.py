"""
Time Integrator Module (JAX)
============================
Author: Marcin Plodzien


This module implements various numerical methods for simulating the time evolution of the Quantum Reservoir.
The core equation is the Liouville-von Neumann equation for closed systems:
    d(rho)/dt = -i [H, rho]

We implement three primary strategies:

1.  **Trotter-Suzuki Decomposition (`evolve_trotter`)**:
    Approximates the exponential exp(-iHt) by splitting H into non-commuting local terms (H = Sum H_k)
    that can be exponentiated exactly.
    - **Pros**: Memory efficient (no large matrix required), naturally parallelizable steps.
    - **Cons**: Approximation error scaling with dt^2 (or dt for 1st order).
    - **Implementation**: We use a First Order Trotter split (Layered: Fields -> Rungs -> Rails (Odd/Even)).

2.  **Exact Diagonalization (`evolve_eig`)**:
    Computes exact evolution using the full eigendecomposition of H.
    - **Pros**: Exact for any time t (no accumulation error). Fast for many time steps once diagonalized.
    - **Cons**: Exponential memory/compute cost (O(2^3N)). feasible only for N <= 14.

3.  **Runge-Kutta 4th Order (`evolve_rk4`)**:
    Classical ODE solver applied to the vector-space representation of rho.
    - **Pros**: Good accuracy control via dt. Works with Sparse Hamiltonians.
    - **Cons**: Requires sparse matrix-vector multiplication (MV) at every sub-step.

Dependencies:
    - jax
    - jax.numpy
    - utils.quantum_core (for gates and Hamiltonian construction)

"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from functools import partial
import utils.quantum_core as uqc

# ==============================================================================
# 1. TROTTER-SUZUKI DECOMPOSITION
# ==============================================================================

@partial(jit, static_argnames=['L_chain', 'n_trotter'])
def evolve_trotter(rho, h_L, h_R, J_rail_left_xyz, J_rail_right_xyz, J_rung_xyz, dt, L_chain, n_trotter, field_L_xyz, field_R_xyz):
    """
    Evolves the density matrix rho using a First-Order Trotter Decomposition.

    Theoretical Foundation: Density Matrix Evolution
    -------------------------------------------------
    The time evolution of a mixed state $\\rho$ is governed by the Von Neumann equation:
    $$ \\frac{d\\rho}{dt} = -i [H, \\rho] $$
    
    The formal solution for a time-independent Hamiltonian over time $dt$ is:
    $$ \\rho(t + dt) = e^{-i H dt} \\rho(t) e^{+i H dt} = U(dt) \\rho(t) U^\\dagger(dt) $$
    
    Trotter Approximation:
    ----------------------
    Since $H = \\sum H_k$ consists of non-commuting terms (Fields, Rungs, Rails), we approximate $U(dt)$
    using the Trotter-Suzuki decomposition:
    $$ U(dt) \\approx \\prod_k e^{-i H_k dt} = U_F U_R U_{Odd} U_{Even} $$
    
    Applying this to $\\rho$:
    $$ \\rho(t+dt) \\approx U_F U_R \\dots \\rho(t) \\dots U_R^\\dagger U_F^\\dagger $$
    
    Iterative Sandwich Update:
    --------------------------
    We apply the gates sequentially. For each gate $U_k$ (e.g., a specific Rung interaction):
    $$ \\rho_{new} = U_k \\rho_{old} U_k^\\dagger $$
    
    In code (via `utils.quantum_core.apply_2q_gate`), this is implemented as a tensor contraction:
    1.  Apply $U_k$ to the **Ket indices** (Left multiplication).
    2.  Apply $U_k^\\dagger$ (Conjugate Transpose) to the **Bra indices** (Right multiplication).
    
    Decomposition Sequence:
    1.  **Local Fields** ($U_{Field}$): Single-qubit rotations. Commute locally.
    2.  **Rungs** ($U_{Rung}$): Two-qubit $(i, i+L)$ interactions. Commute pairwise.
    3.  **Rails** ($U_{Rail}$): Intra-rail interactions. Non-commuting.
        - Split into **Odd** layer $((0,1), (2,3)...)$
        - and **Even** layer $((1,2), (3,4)...)$ to ensure parallelizability.

    Args:
        rho (jnp.ndarray): Density matrix (D, D).
        h_L, h_R (array): Disorder/Field amplitudes for L/R rails.
        J_rail_* (tuple): (Jx, Jy, Jz) couplings.
        dt (float): Trotter time step (NOT total time).
        L_chain (int): Half-system size.
        n_trotter (int): Number of steps to apply.

    Returns:
        jnp.ndarray: Evolved density matrix.
    """
    
    # --- A. Pre-Compute Gates (JIT-Constant if params constant) ---
    
    # 1. Prepare Field Gates (Single Qubit)
    # The field parameters might vary per site (disorder h_L[i]), so we map generic gate creation.
    fxL, fyL, fzL = field_L_xyz
    fxR, fyR, fzR = field_R_xyz
    
    # Create vectors (L, 3) representing the local field vector h_i * (fx, fy, fz)
    f_vec_L = jnp.stack([h_L * fxL, h_L * fyL, h_L * fzL], axis=1) # Shape (L, 3)
    f_vec_R = jnp.stack([h_R * fxR, h_R * fyR, h_R * fzR], axis=1) # Shape (L, 3)
    
    # Generate batch of L gates for Left Rail and L gates for Right Rail
    # get_Field_gate is mapped over the field vectors (axis 0)
    gates_XL = vmap(uqc.get_Field_gate, in_axes=(0, None))(f_vec_L, dt)
    gates_XR = vmap(uqc.get_Field_gate, in_axes=(0, None))(f_vec_R, dt)

    # 2. Prepare Interaction Gates (Two Qubit)
    # These are uniform across the rail/rung (assuming J is constant)
    gate_rail_left = uqc.get_XYZ_gate(J_rail_left_xyz, dt)
    gate_rail_right = uqc.get_XYZ_gate(J_rail_right_xyz, dt)
    gate_rung = uqc.get_XYZ_gate(J_rung_xyz, dt)

    # --- B. Evolution Loop (lax.scan) ---
    # We use lax.scan for efficient JIT looping on accelerator (GPU/TPU) or CPU.
    
    def step_fn(curr_rho, _):
        """Single Trotter Step Kernel."""
        r = curr_rho
        
        # 1. Apply Local Fields (Commuting layer)
        # Apply L gates and R gates. They all commute with each other.
        for i in range(L_chain):
            r = uqc.apply_1q_gate(r, gates_XL[i], i, L_chain)             # Left Rail Site i
            r = uqc.apply_1q_gate(r, gates_XR[i], i + L_chain, L_chain)   # Right Rail Site i+L
            
        # 2. Apply Rungs (Commuting layer: (0,L), (1,L+1)... disjoint)
        for i in range(L_chain):
            r = uqc.apply_2q_gate(r, gate_rung, i, i + L_chain, L_chain)
            
        # 3. Apply Rails (Non-commuting Intra-rail bonds)
        # Split into Odd/Even bonds to ensure commutativity within sub-layer
        
        # Odd Layer: Bonds starting at 0, 2, 4... ((0,1), (2,3)...)
        for i in range(0, L_chain - 1, 2):
            r = uqc.apply_2q_gate(r, gate_rail_left, i, i+1, L_chain)           # Left Rail
            r = uqc.apply_2q_gate(r, gate_rail_right, i+L_chain, i+1+L_chain, L_chain) # Right Rail
            
        # Even Layer: Bonds starting at 1, 3, 5... ((1,2), (3,4)...)
        for i in range(1, L_chain - 1, 2):
            r = uqc.apply_2q_gate(r, gate_rail_left, i, i+1, L_chain)
            r = uqc.apply_2q_gate(r, gate_rail_right, i+L_chain, i+1+L_chain, L_chain)
            
        return r, None

    # Run the loop n_trotter times
    # lax.scan carries the state 'rho' through 'step_fn'
    final_rho, _ = lax.scan(step_fn, rho, None, length=n_trotter)
    return final_rho

@partial(jit, static_argnames=['L_chain', 'n_trotter'])
def evolve_trotter_many_body(rho, h_L, h_R, J_all_xyz, dt, L_chain, n_trotter, field_xyz):
    """
    Evolves rho using an All-to-All Trotter Decomposition.
    
    For All-to-All, splitting into commuting layers is complex (Graph Coloring).
    Here we implement a FIRST ORDER sequential swap:
    - Fields
    - Interactions (Pairwise sequential)
    
    Note on Accuracy: The non-commutativity error between pairs (i,j) and (j,k)
    means the order of loops matters. We use a fixed lexicographic order (i<j).
    """
    L_total = 2 * L_chain
    
    # 1. Gates
    # A. Fields
    fx, fy, fz = field_xyz
    f_vec_L = jnp.stack([h_L * fx, h_L * fy, h_L * fz], axis=1) # (L, 3)
    f_vec_R = jnp.stack([h_R * fx, h_R * fy, h_R * fz], axis=1) # (L, 3)
    f_vec_all = jnp.concatenate([f_vec_L, f_vec_R], axis=0)     # (2L, 3)
    
    gates_Field = vmap(uqc.get_Field_gate, in_axes=(0, None))(f_vec_all, dt)
    
    # B. Coupling
    gate_J = uqc.get_XYZ_gate(J_all_xyz, dt)
    
    # Generate list of pairs (i, j) with i < j
    pairs = []
    for i in range(L_total):
        for j in range(i + 1, L_total):
            pairs.append((i, j))
            
    def step_fn(curr_rho, _):
        r = curr_rho
        # 1. Fields
        for i in range(L_total):
             r = uqc.apply_1q_gate(r, gates_Field[i], i, L_chain)
        
        # 2. Couplings (Sequential)
        for (i, j) in pairs:
            r = uqc.apply_2q_gate(r, gate_J, i, j, L_chain)
            
        return r, None

    final_rho, _ = lax.scan(step_fn, rho, None, length=n_trotter)
    return final_rho

# ==============================================================================
# 2. EXACT DIAGONALIZATION (Pre-computed)
# ==============================================================================

@partial(jit)
def evolve_eig(rho, eig_vals, eig_vecs, t_evol):
    """
    Evolves rho using pre-computed eigenvalues and eigenvectors.
    This is extremely efficient for computing evolution at arbitrary time 't'
    once diagonalization is done.

    Theory:
    H = V D V†  (D diagonal expit values)
    U(t) = exp(-iHt) = V exp(-iDt) V†
    
    rho(t) = U(t) rho(0) U(t)†
           = (V exp(-iDt) V†) rho (V exp(iDt) V†)
           
    Args:
        rho (jnp.ndarray): Initial density matrix.
        eig_vals (jnp.ndarray): Eigenvalues of H (dim,).
        eig_vecs (jnp.ndarray): Eigenvectors of H (matrix V) (dim, dim).
        t_evol (float): Time duration.
        
    Returns:
        jnp.ndarray: Evolved rho.
        
    Performance Note:
    -----------------
    The operation `U @ rho @ U.conj().T` involves three $D \\times D$ matrices.
    Associativity does not impact complexity here:
    - $(U \\rho) U^\\dagger$: $O(D^3)$ then $O(D^3) \\to 2D^3$.
    - $U (\\rho U^\\dagger)$: $O(D^3)$ then $O(D^3) \\to 2D^3$.
    
    This contrasts with State Vector evolution ($|\\psi\\rangle$), where order is critical:
    - $(U A) |\\psi\\rangle$: $O(D^3)$ vs $U (A |\\psi\\rangle)$: $O(D^2)$.
    For Density Matrices, we are stuck with $O(D^3)$.
    """
    # 1. Construct time evolution operator diagonal matrix D(t)
    phases = jnp.exp(-1j * eig_vals * t_evol) # vector: [e^{-i E_0 t}, ...]

    # 2. Construct U(t) = V @ diag(phases) @ V†
    # Optimized: (V * phases) does row-wise scaling equivalent to V @ D
    U = (eig_vecs * phases[None, :]) @ jnp.conj(eig_vecs).T
    
    # 3. Evolve
    # 3. Evolve
    return U @ rho @ jnp.conj(U).T


@partial(jit)
def construct_unitary_eig(eig_vals, eig_vecs, t_evol):
    """
    Constructs the Unitary Evolution Operator U(t) from eigensystem.
    U(t) = V exp(-iDt) V^dagger.
    
    Useful for pre-computing U when t_evol is constant across many steps.
    """
    phases = jnp.exp(-1j * eig_vals * t_evol)
    U = (eig_vecs * phases[None, :]) @ jnp.conj(eig_vecs).T
    return U

@partial(jit)
def apply_dense_operator(rho, U, U_dag=None):
    """
    Applies a dense global operator U to rho:
    rho_new = U @ rho @ U^dagger
    
    Args:
        rho: Density Matrix
        U: Unitary Operator
        U_dag: (Optional) Pre-computed Conjugate Transpose of U.
    """
    if U_dag is None:
        U_dag = jnp.conj(U).T
    return U @ rho @ U_dag



@partial(jit, static_argnames=['L_chain', 'topology'])
def evolve_exact_expm(rho, h_L, h_R, J_rail_left_xyz, J_rail_right_xyz, J_rung_xyz, J_all_xyz, t_evol, L_chain, field_L_xyz, field_R_xyz, topology):
    """
    Evolves rho by computing exact Matrix Exponential of the FULL Hamiltonian.
    U = expm(-iHt).
    
    Note: Less efficient than `evolve_eig` if called repeatedly for different 't'
    because it recomputes expm each time. Used mainly for benchmarking.
    """
    # Construct Dense Hamiltonian
    if topology == 'ladder':
        H = uqc.construct_hamiltonian_ladder(h_L, h_R, J_rail_left_xyz, J_rail_right_xyz, J_rung_xyz, L_chain, field_L_xyz, field_R_xyz)
    elif topology == 'all_to_all':
        H = uqc.construct_hamiltonian_all2all(h_L, h_R, J_all_xyz, L_chain, field_L_xyz, field_R_xyz)
    else:
        H = jnp.zeros((2**(2*L_chain), 2**(2*L_chain)), dtype=jnp.complex128)

    # Compute U
    U = jax.scipy.linalg.expm(-1j * H * t_evol)
    
    # Evolve
    rho_new = U @ rho @ jnp.conj(U).T
    return rho_new


# ==============================================================================
# 3. RUNGE-KUTTA 4th ORDER INTEGRATION (RK4)
# ==============================================================================

from jax.experimental import sparse

@partial(jit, static_argnames=['L_chain', 'n_steps', 'topology'])
def evolve_rk4_dense(rho, h_L, h_R, J_rail_left_xyz, J_rail_right_xyz, J_rung_xyz, J_all_xyz, dt, L_chain, n_steps, field_L_xyz, field_R_xyz, topology):
    """
    Standard RK4 solver for d(rho)/dt = -i [H, rho].
    Uses dense Hamiltonian matrix.
    
    RK4 Steps:
    k1 = f(y)
    k2 = f(y + dt/2 k1)
    k3 = f(y + dt/2 k2)
    k4 = f(y + dt k3)
    y_next = y + dt/6 (k1 + 2k2 + 2k3 + k4)
    """
    # 1. Build H
    if topology == 'ladder':
        H = uqc.construct_hamiltonian_ladder(h_L, h_R, J_rail_left_xyz, J_rail_right_xyz, J_rung_xyz, L_chain, field_L_xyz, field_R_xyz)
    elif topology == 'all_to_all':
        H = uqc.construct_hamiltonian_all2all(h_L, h_R, J_all_xyz, L_chain, field_L_xyz, field_R_xyz)
    else:
        H = jnp.zeros((2**(2*L_chain), 2**(2*L_chain)), dtype=jnp.complex128)

    # Commutator Function: f(rho) = -i [H, rho]
    def commutator(r):
        return -1j * (H @ r - r @ H)

    def step_fn(curr_rho, _):
        k1 = commutator(curr_rho)
        k2 = commutator(curr_rho + 0.5 * dt * k1)
        k3 = commutator(curr_rho + 0.5 * dt * k2)
        k4 = commutator(curr_rho + dt * k3)
        
        next_rho = curr_rho + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return next_rho, None
        
    final_rho, _ = lax.scan(step_fn, rho, None, length=n_steps)
    return final_rho


@partial(jit, static_argnames=['n_steps'])
def evolve_rk4_sparse(rho, H_sparse, dt, n_steps):
    """
    RK4 Solver dealing with Sparse Hamiltonian (BCOO).
    Crucial for memory efficiency on larger systems.
    """
    
    def commutator(r):
        """
        Computes -i [H_sparse, rho_dense].
        JAX BCOO matrices support matrix multiplication with dense matrices.
        
        H @ rho: Sparse @ Dense -> Dense
        rho @ H: We use equality (rho @ H) = (H† @ rho†)† = (H @ rho†)† for Hermitian H.
        This allows using Sparse @ Dense logic again.
        """
        # 1. H @ rho
        Hr = H_sparse @ r
        
        # 2. rho @ H
        # Since H is Hermitian (real coeffs of Paulis, effectively), H = H_conj_transpose?
        # Actually H complex but Hermitian. H = H.conj().T
        # r @ H = (H @ r.conj().T).conj().T
        rH = jnp.conj((H_sparse @ jnp.conj(r).T).T)
        
        return -1j * (Hr - rH)

    def step_fn(curr_rho, _):
        k1 = commutator(curr_rho)
        k2 = commutator(curr_rho + 0.5 * dt * k1)
        k3 = commutator(curr_rho + 0.5 * dt * k2)
        k4 = commutator(curr_rho + dt * k3)
        
        next_rho = curr_rho + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return next_rho, None
        
    final_rho, _ = lax.scan(step_fn, rho, None, length=n_steps)
    return final_rho
