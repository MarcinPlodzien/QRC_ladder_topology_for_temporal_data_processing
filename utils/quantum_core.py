"""
Quantum Core Primitives (JAX)
=============================
Author: Marcin Plodzien


This module provides the fundamental quantum mechanical operations required for the QRC simulation.
It is built on JAX for high-performance automatic differentiation and JIT compilation.

Key functionalities include:
1.  **Quantum Gates**: Definitions of Pauli matrices (X, Y, Z, I) and rotation gates (Ry, ZZ, X-rot, Z-rot).
2.  **Hamiltonian Operators**: Functions to construct specialized Hamiltonians (e.g., Ladder, All-to-All)
    in both dense and sparse (BCOO) formats.
3.  **State Evolution**: Primitives for applying single and two-qubit gates (`apply_1q_gate`, `apply_2q_gate`).
4.  **State Manipulation**: Partial trace and state injection/reset logic.

Dependencies:
    - jax
    - jax.numpy
    - functools.partial

Usage Notes:
    - All functions operate with 64-bit precision (`jax_enable_x64=True`) for numerical stability.
    - Many functions are JIT-compatible and use `static_argnames` for configuration parameters.
    - Sparse Hamiltonian construction avoids large matrix allocations for N > 10.
"""

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

# Enable 64-bit precision for quantum stability (if not already enabled globally)
jax.config.update("jax_enable_x64", True)

# ==============================================================================
# 1. QUANTUM PRIMITIVES & GATES
# ==============================================================================

def get_pauli():
    """
    Returns the standard Pauli matrices I, X, Z.
    
    Y is typically constructed as -i * X @ Z or defined explicitly when needed.
    
    Returns:
        tuple[jnp.ndarray]: (I, X, Z) matrices of shape (2, 2).
    """
    I = jnp.eye(2, dtype=jnp.complex128)
    X = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
    Z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)
    return I, X, Z

# Pre-compute basic gates for efficiency (used in broadcasing if needed)
I2, X2, Z2 = get_pauli()

def get_Ry(theta):
    """
    Constructs a batched Single-Qubit Ry(theta) gate.
    
    Ry(theta) = [[cos(theta/2), -sin(theta/2)],
                 [sin(theta/2),  cos(theta/2)]]
                 
    Args:
        theta (float or jnp.ndarray): Rotation angle(s).
        
    Returns:
        jnp.ndarray: Matrix of shape (..., 2, 2).
    """
    c = jnp.cos(theta/2)
    s = jnp.sin(theta/2)
    # Shape: (..., 2, 2)
    return jnp.array([[c, -s], [s, c]], dtype=jnp.complex128).transpose(2, 0, 1)

def get_ZZ_gate(J, dt):
    r"""
    Constructs the Two-Qubit ZZ interaction gate.
    
    U_ZZ = exp(-i J Z \otimes Z dt)
    Diagonal in computational basis:
        |00>: exp(-i J dt)
        |01>: exp(+i J dt)
        |10>: exp(+i J dt)
        |11>: exp(-i J dt)
                 
    Args:
        J (float): Coupling strength.
        dt (float): Time step.
        
    Returns:
        jnp.ndarray: Diagonal matrix reshaped to (2, 2, 2, 2) tensor.
    """
    # Diagonal phases: [00, 01, 10, 11] -> [1, -1, -1, 1] sign pattern
    phases = jnp.exp(-1j * J * dt * jnp.array([1, -1, -1, 1], dtype=jnp.complex128))
    # Return as (2,2,2,2) for tensor contraction
    return jnp.diag(phases).reshape(2, 2, 2, 2)

def get_X_rot(h, dt):
    """
    Constructs Single-Qubit X rotation: U = exp(-i h X dt).
    
    Convention: H = h*X -> U = exp(-i H t)
    Analytical form: cos(h*dt)I - i*sin(h*dt)X
    
    Args:
        h (float): Field strength / Amplitude.
        dt (float): Time step.
        
    Returns:
        jnp.ndarray: (2, 2) matrix.
    """
    c = jnp.cos(h * dt)
    s = jnp.sin(h * dt)
    return c * I2 - 1j * s * X2

def get_Z_rot(h, dt):
    """
    Constructs Single-Qubit Z rotation: U = exp(-i h Z dt).
    
    Analytical form: cos(h*dt)I - i*sin(h*dt)Z
    
    Args:
        h (float): Field strength.
        dt (float): Time step.
        
    Returns:
        jnp.ndarray: (2, 2) matrix.
    """
    c = jnp.cos(h * dt)
    s = jnp.sin(h * dt)
    return c * I2 - 1j * s * Z2

def get_XYZ_gate(J_xyz, dt):
    """
    Constructs the General Heisenberg interaction gate (XX + YY + ZZ).
    
    U = exp(-i dt (Jx XX + Jy YY + Jz ZZ))
    
    Since XX, YY, and ZZ commute, the evolution can be factorized:
    U = exp(-i Jx XX dt) @ exp(-i Jy YY dt) @ exp(-i Jz ZZ dt)
    
    Args:
        J_xyz (tuple): (Jx, Jy, Jz) coupling strengths.
        dt (float): Time step.
        
    Returns:
        jnp.ndarray: (2, 2, 2, 2) unitary tensor.
    """
    Jx, Jy, Jz = J_xyz
    
    # 1. XX Interaction: R_xx(Jx)
    cx = jnp.cos(Jx * dt); sx = jnp.sin(Jx * dt)
    U_xx = jnp.array([
        [cx, 0, 0, -1j*sx],
        [0, cx, -1j*sx, 0],
        [0, -1j*sx, cx, 0],
        [-1j*sx, 0, 0, cx]
    ], dtype=jnp.complex128)
    
    # 2. YY Interaction: R_yy(Jy)
    cy = jnp.cos(Jy * dt); sy = jnp.sin(Jy * dt)
    U_yy = jnp.array([
        [cy, 0, 0, 1j*sy],
        [0, cy, -1j*sy, 0],
        [0, -1j*sy, cy, 0],
        [1j*sy, 0, 0, cy]
    ], dtype=jnp.complex128)
    
    # 3. ZZ Interaction: R_zz(Jz) (Diagonal)
    phases = jnp.exp(-1j * Jz * dt * jnp.array([1, -1, -1, 1], dtype=jnp.complex128))
    U_zz = jnp.diag(phases)
    
    # Combine (Commuting -> Order doesn't matter)
    U = U_xx @ U_yy @ U_zz
    return U.reshape(2, 2, 2, 2)

def get_Field_gate(h_xyz, dt):
    """
    Constructs a general Single-Qubit rotation for a field vector h=(hx, hy, hz).
    
    H = h . sigma = hx X + hy Y + hz Z
    U = exp(-i H dt) = cos(theta) I - i sin(theta) (n . sigma)
    where theta = |h| * dt, n = h / |h|.
    
    Args:
        h_xyz (tuple): Field components (hx, hy, hz).
        dt (float): Time step.
        
    Returns:
        jnp.ndarray: (2, 2) unitary matrix.
    """
    hx, hy, hz = h_xyz
    
    # Calculate Magnitude |h|
    h_mag = jnp.sqrt(hx**2 + hy**2 + hz**2)
    
    # Numerical stability: Avoid division by zero if |h| ~ 0
    Safe_h = jnp.where(h_mag < 1e-12, 1.0, h_mag)
    
    c = jnp.cos(h_mag * dt)
    s = jnp.sin(h_mag * dt)
    
    # Normalized direction vector n
    nx = hx / Safe_h
    ny = hy / Safe_h
    nz = hz / Safe_h
    
    # Construct Matrix Elements:
    # Diagonal term: c*I - i*s*nz*Z => [c - i*s*nz,  0          ]
    #                                  [0,           c + i*s*nz ]
    # Off-diag: -i*s*(nx*X + ny*Y)
    
    u00 = c - 1j * s * nz
    u01 = -s * ny - 1j * s * nx
    u10 = s * ny - 1j * s * nx
    u11 = c + 1j * s * nz
    
    U = jnp.array([[u00, u01], [u10, u11]], dtype=jnp.complex128)
    return U


# ==============================================================================
# 2. CORE EVOLUTION LOGIC (TENSOR CONTRACTION)
# ==============================================================================

@partial(jit, static_argnames=['i', 'j', 'L_chain'])
def apply_2q_gate(rho, U_tensor, i, j, L_chain):
    """
    Applies a two-qubit gate U to the density matrix rho.
    
    Operation: rho' = U rho U†
    Implemented via tensor contraction (einsum) for efficiency in JAX.
    
    Args:
        rho (jnp.ndarray): Density matrix of shape (D, D) where D=2^N_total.
        U_tensor (jnp.ndarray): 4-tensor gate of shape (2, 2, 2, 2).
        i (int): Index of the first target qubit.
        j (int): Index of the second target qubit.
        L_chain (int): Length of one rail (N_total = 2 * L_chain).
        
    Returns:
        jnp.ndarray: Updated density matrix rho'.
    """
    L_total = 2 * L_chain
    dim = 2**L_total
    
    # 1. Reshape to isolate target qubits i and j
    # This assumes i < j. Logic generalizes but order matters for reshaping.
    # Dimensions: (pre_i, i, mid, j, post)
    
    sz_pre = 2**i
    sz_mid = 2**(j - i - 1)
    sz_post = 2**(L_total - 1 - j)
    
    # Reshape rho to explicit tensor view: (Pre, 2_i, Mid, 2_j, Post,  Pre, 2_i, Mid, 2_j, Post)
    # The second half represents the "bra" (column) indices.
    rho_view = rho.reshape(sz_pre, 2, sz_mid, 2, sz_post,  sz_pre, 2, sz_mid, 2, sz_post)
    
    # 2. Apply U to Ket side (Indices 1 and 3)
    # U_{AB, xy} acts on rho_{...x...y...}
    # einsum indices: 'ABxy' (gate), 'pxmyq PXMYQ' (rho) -> 'pAmBq PXMYQ'
    rho_ket = jnp.einsum('ABxy, pxmyq P X M Y Q -> pAmBq P X M Y Q', U_tensor, rho_view)
    
    # 3. Apply U† to Bra side (Indices 6 and 8)
    # rho_{...X...Y...} U†_{XY, UV} -> rho_{...U...V...}
    # Note: U† is conjugate transpose. Here we use Conj(U) and contract properly.
    # einsum: 'UVXY' (gate conj), 'pAmBq PXMYQ' (rho_ket) -> 'pAmBq PUMVQ'
    rho_final = jnp.einsum('UVXY, pAmBq PXMYQ -> pAmBq PUMVQ', jnp.conj(U_tensor), rho_ket)
    
    return rho_final.reshape(dim, dim)


@partial(jit, static_argnames=['i', 'L_chain'])
def apply_1q_gate(rho, U, i, L_chain):
    """
    Applies a single-qubit gate U to qubit i.
    
    Args:
        rho (jnp.ndarray): Density matrix (D, D).
        U (jnp.ndarray): 2x2 unitary matrix.
        i (int): Target qubit index.
        L_chain (int): Half-system size.
        
    Returns:
        jnp.ndarray: Updated density matrix.
    """
    L_total = 2 * L_chain
    dim_total = 2**L_total
    
    sz_pre = 2**i
    sz_post = 2**(L_total - 1 - i)
    
    # Reshape: (Pre, Site, Post, Pre, Site, Post)
    rho_view = rho.reshape(sz_pre, 2, sz_post, sz_pre, 2, sz_post)
    
    # Ket Update: U_{ax} rho_{...x...}
    # 'ax': U indices (new, old)
    # 'pxq': rho ket indices (pre, site, post)
    rho_ket = jnp.einsum('ax, pxq P X Q -> paq P X Q', U, rho_view)
    
    # Bra Update: rho_{...X...} U†_{XA} -> rho_{...A...} U*_{AX}
    rho_final = jnp.einsum('AX, paq P X Q -> paq P A Q', jnp.conj(U), rho_ket)
    
    return rho_final.reshape(dim_total, dim_total)


# ==============================================================================
# 3. DENSE HAMILTONIAN CONSTRUCTION
# ==============================================================================

@partial(jit, static_argnames=['L_chain'])
def construct_hamiltonian_ladder(h_L, h_R, J_rail_left_xyz, J_rail_right_xyz, J_rung_xyz, L_chain, field_L_xyz, field_R_xyz):
    """
    Constructs the dense Hamiltonian matrix for the Ladder topology.
    
    Structure:
    - Two parallel rails of length L_chain (Total N = 2*L).
    - Sites 0..L-1 (Left Rail), Sites L..2L-1 (Right Rail).
    - Couplings: Intra-rail (Nearest Neighbor), Inter-rail (Rungs usually i to i+L).
    
    Args:
        h_L, h_R (array): Local disorder/control fields for L/R rails.
        J_rail_* (tuple): (Jx, Jy, Jz) couplings for rails.
        J_rung (tuple): (Jx, Jy, Jz) coupling for rungs.
        L_chain (int): Rail length.
        field_* (tuple): Global field direction vectors.
        
    Returns:
        jnp.ndarray: Hamiltonian matrix (2^N, 2^N).
    """
    N_total = 2 * L_chain
    dim = 2**N_total
    
    I, X, Z = get_pauli()
    Y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
    
    H = jnp.zeros((dim, dim), dtype=jnp.complex128)
    
    # Helper to create Full Operator from list of local ops
    def get_term(op_list, indices):
        """Builds tensor product O_i x O_j ... x I everywhere else."""
        res = jnp.eye(1, dtype=jnp.complex128)
        for k in range(N_total):
            if k in indices:
                idx = indices.index(k)
                op = op_list[idx]
            else:
                op = I
            res = jnp.kron(res, op)
        return res

    # --- 1. External Fields (Local Terms) ---
    fxL, fyL, fzL = field_L_xyz
    fxR, fyR, fzR = field_R_xyz
    
    # Left Rail
    for i in range(L_chain):
        # H_local = h_i * (fx X + fy Y + fz Z)
        op = h_L[i] * (fxL*X + fyL*Y + fzL*Z)
        H += get_term([op], [i])
        
    # Right Rail
    for i in range(L_chain):
        op = h_R[i] * (fxR*X + fyR*Y + fzR*Z)
        H += get_term([op], [i + L_chain])
        
    # --- 2. Interaction Terms (Couplings) ---
    Jx_L, Jy_L, Jz_L = J_rail_left_xyz
    Jx_R, Jy_R, Jz_R = J_rail_right_xyz
    Jx_rung, Jy_rung, Jz_rung = J_rung_xyz
    
    # Left Rail Intra-chain (i, i+1)
    for i in range(L_chain - 1):
        H += Jx_L * get_term([X, X], [i, i+1])
        H += Jy_L * get_term([Y, Y], [i, i+1])
        H += Jz_L * get_term([Z, Z], [i, i+1])
            
    # Right Rail Intra-chain
    for i in range(L_chain - 1):
        idx1, idx2 = i + L_chain, i + 1 + L_chain
        H += Jx_R * get_term([X, X], [idx1, idx2])
        H += Jy_R * get_term([Y, Y], [idx1, idx2])
        H += Jz_R * get_term([Z, Z], [idx1, idx2])
        
    # Rung Inter-chain (i, i+L)
    for i in range(L_chain):
        idx1, idx2 = i, i + L_chain
        H += Jx_rung * get_term([X, X], [idx1, idx2])
        H += Jy_rung * get_term([Y, Y], [idx1, idx2])
        H += Jz_rung * get_term([Z, Z], [idx1, idx2])
        
    return H

@partial(jit, static_argnames=['L_chain'])
def construct_hamiltonian_all2all(h_L, h_R, J_all_xyz, L_chain, field_L_xyz, field_R_xyz):
    """
    Constructs Full Hamiltonian for All-to-All topology.
    Every qubit interacts with every other qubit.
    
    Args:
        J_all_xyz (tuple): Global coupling strength (Jx, Jy, Jz) for all pairs.
        field_L_xyz (tuple): Field vector for Left Rail sites.
        field_R_xyz (tuple): Field vector for Right Rail sites.
    """
    N_total = 2 * L_chain
    dim = 2**N_total
    
    I, X, Z = get_pauli()
    Y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
    
    H = jnp.zeros((dim, dim), dtype=jnp.complex128)
    
    # helper
    def get_term(op_list, indices):
        res = jnp.eye(1, dtype=jnp.complex128)
        for k in range(N_total):
            if k in indices:
                idx = indices.index(k)
                op = op_list[idx]
            else:
                op = I
            res = jnp.kron(res, op)
        return res

    # 1. Fields
    fxL, fyL, fzL = field_L_xyz
    fxR, fyR, fzR = field_R_xyz
    
    for i in range(L_chain):
        op = h_L[i] * (fxL*X + fyL*Y + fzL*Z)
        H += get_term([op], [i])
        
    for i in range(L_chain):
        op = h_R[i] * (fxR*X + fyR*Y + fzR*Z)
        H += get_term([op], [i + L_chain])
        
    # 2. All-to-All Couplings
    Jx, Jy, Jz = J_all_xyz
    
    # Loop pairs i < j
    for i in range(N_total):
        for j in range(i + 1, N_total):
            H += Jx * get_term([X, X], [i, j])
            H += Jy * get_term([Y, Y], [i, j])
            H += Jz * get_term([Z, Z], [i, j])
            
    return H

# ==============================================================================
# 4. SPARSE HAMILTONIAN CONSTRUCTION (BCOO)
# ==============================================================================

from jax.experimental import sparse

def get_sparse_pauli_term(op_list, sites, N_total):
    """
    Generates (values, indices) for a Pauli string operator in sparse format.
    
    Mechanism: 
    - Pauli matrices are extremely sparse (max 1 non-zero per row/col).
    - Tensor product of Paulis is also 1-sparse (permutation matrix with phases).
    - Values are phases (+1, -1, i, -i).
    - Indices are determined by bitwise operations (X and Y flip bits).
    
    Returns:
        tuple: (values, indices) for BCOO construction.
    """
    dim = 2**N_total
    rows = jnp.arange(dim)
    
    # 1. Determine Column Indices (Bit flips)
    flip_mask = 0
    for op, site in zip(op_list, sites):
        if op in ['X', 'Y']:
            # X/Y flip the bit at position 'site'
            flip_mask |= (1 << (N_total - 1 - site))
            
    cols = rows ^ flip_mask
    indices = jnp.stack([rows, cols], axis=1)
    
    # 2. Determine Values (Phases)
    vals = jnp.ones(dim, dtype=jnp.complex128)
    
    for op, site in zip(op_list, sites):
        if op == 'Z':
            # Z|k>: if k-th bit is 1, phase -1. Else +1.
            bit = (rows >> (N_total - 1 - site)) & 1
            sign = jnp.where(bit == 0, 1.0, -1.0)
            vals *= sign
        elif op == 'Y':
            # Y = i X Z. (Bit flip handled in cols).
            # Phase depends on bit value BEFORE flip.
            # 0 -> 1: factor +i
            # 1 -> 0: factor -i
            bit = (rows >> (N_total - 1 - site)) & 1
            factor = jnp.where(bit == 0, 1.0j, -1.0j)
            vals *= factor
        # X: Phase is +1.
        
    return vals, indices

@partial(jit, static_argnames=['L_chain'])
def construct_hamiltonian_ladder_sparse(h_L, h_R, J_rail_left_xyz, J_rail_right_xyz, J_rung_xyz, L_chain, field_L_xyz, field_R_xyz):
    """
    Constructs sparse BCOO Hamiltonian for Ladder. 
    Memory efficient for larger N (e.g., N > 10).
    """
    N_total = 2 * L_chain
    dim = 2**N_total
    
    all_vals = []
    all_inds = []
    
    def add_term(coeff, ops, sites):
        v, ind = get_sparse_pauli_term(ops, sites, N_total)
        all_vals.append(v * coeff)
        all_inds.append(ind)
        
    # --- 1. Fields ---
    fxL, fyL, fzL = field_L_xyz
    fxR, fyR, fzR = field_R_xyz
    
    # Left
    for i in range(L_chain):
        add_term(h_L[i] * fxL, ['X'], [i])
        add_term(h_L[i] * fyL, ['Y'], [i])
        add_term(h_L[i] * fzL, ['Z'], [i])
        
    # Right
    for i in range(L_chain):
        idx = i + L_chain
        add_term(h_R[i] * fxR, ['X'], [idx])
        add_term(h_R[i] * fyR, ['Y'], [idx])
        add_term(h_R[i] * fzR, ['Z'], [idx])
        
    # --- 2. Couplings ---
    Jx_L, Jy_L, Jz_L = J_rail_left_xyz
    Jx_R, Jy_R, Jz_R = J_rail_right_xyz
    Jx_rung, Jy_rung, Jz_rung = J_rung_xyz
    
    # Left Rail
    for i in range(L_chain - 1):
        add_term(Jx_L, ['X','X'], [i, i+1])
        add_term(Jy_L, ['Y','Y'], [i, i+1])
        add_term(Jz_L, ['Z','Z'], [i, i+1])
        
    # Right Rail
    for i in range(L_chain - 1):
        s1, s2 = i+L_chain, i+1+L_chain
        add_term(Jx_R, ['X','X'], [s1, s2])
        add_term(Jy_R, ['Y','Y'], [s1, s2])
        add_term(Jz_R, ['Z','Z'], [s1, s2])
        
    # Rungs
    for i in range(L_chain):
        s1, s2 = i, i+L_chain
        add_term(Jx_rung, ['X','X'], [s1, s2])
        add_term(Jy_rung, ['Y','Y'], [s1, s2])
        add_term(Jz_rung, ['Z','Z'], [s1, s2])
        
    # Concatenate all sparse terms
    if not all_vals:
        return sparse.BCOO.fromdense(jnp.zeros((dim, dim), dtype=jnp.complex128))
    
    total_vals = jnp.concatenate(all_vals)
    total_inds = jnp.concatenate(all_inds)
    
    # Create Matrix and Sum Duplicates
    H = sparse.BCOO((total_vals, total_inds), shape=(dim, dim))
    H = H.sum_duplicates(nse=len(total_vals)) 
    
    return H


@partial(jit, static_argnames=['L_chain'])
def construct_hamiltonian_all2all_sparse(h_L, h_R, J_all_xyz, L_chain, field_L_xyz, field_R_xyz):
    """
    Constructs sparse BCOO Hamiltonian for All-to-All.
    """
    N_total = 2 * L_chain
    dim = 2**N_total
    
    all_vals = []
    all_inds = []
    
    def add_term(coeff, ops, sites):
        v, ind = get_sparse_pauli_term(ops, sites, N_total)
        all_vals.append(v * coeff)
        all_inds.append(ind)
        
    fxL, fyL, fzL = field_L_xyz
    fxR, fyR, fzR = field_R_xyz
    
    # Fields
    for i in range(N_total):
        # Assuming h_L/h_R map to linear index 0..N-1?
        # Let's assume h_vec is concatenated or handled separately.
        # Here we duplicate the logic from dense for safety.
        if i < L_chain:
            h_val = h_L[i]
            fx, fy, fz = fxL, fyL, fzL
        else:
            h_val = h_R[i - L_chain]
            fx, fy, fz = fxR, fyR, fzR
            
        add_term(h_val * fx, ['X'], [i])
        add_term(h_val * fy, ['Y'], [i])
        add_term(h_val * fz, ['Z'], [i])
        
    # All2All
    Jx, Jy, Jz = J_all_xyz
    for i in range(N_total):
        for j in range(i + 1, N_total):
            add_term(Jx, ['X','X'], [i, j])
            add_term(Jy, ['Y','Y'], [i, j])
            add_term(Jz, ['Z','Z'], [i, j])
            
    if not all_vals:
        return sparse.BCOO.fromdense(jnp.zeros((dim, dim), dtype=jnp.complex128))
        
    total_vals = jnp.concatenate(all_vals)
    total_inds = jnp.concatenate(all_inds)
    
    H = sparse.BCOO((total_vals, total_inds), shape=(dim, dim))
    H = H.sum_duplicates(nse=len(total_vals))
    return H

# ==============================================================================
# 5. STATE MANIPULATION & PARTIAL TRACE
# ==============================================================================

@partial(jit, static_argnames=['L_chain'])
def partial_trace_and_reset(rho_total, psi_new_L, L_chain):
    r"""
    Performs the QRC Reset Step:
    1. Traces out the Left Rail (computational/input qubits) from the total state.
    2. Replaces them with a fresh input state |psi_L>.
    3. Keeps the Right Rail (reservoir/memory qubits) intact.
    
    Mathematically:
    rho_new = |psi_L><psi_L| \otimes Tr_L(rho_total)
    
    Args:
        rho_total (jnp.ndarray): Flattened density matrix (D, D).
        psi_new_L (jnp.ndarray): New state vector for Left rail (d_L,).
        L_chain (int): Size of Left rail.
        
    Returns:
        jnp.ndarray: New density matrix (D, D).
    """
    dim_L = 2**L_chain
    dim_R = 2**L_chain
    
    # 1. Partial Trace L
    # We view rho as (DimL, DimR, DimL, DimR) tensor.
    # Trace means summing diagonal elements of L subsystem: rho_R = Sum_k <k|_L rho |k>_L
    # indices: (i_L, i_R, j_L, j_R). We execute rho_{i_L=j_L} -> sum over i_L.
    r_reshaped = rho_total.reshape(dim_L, dim_R, dim_L, dim_R)
    
    # Einstein sum: 'lilj -> ij' implies:
    # l, l are contracted (summed). i, j are free indices (Right rail).
    rho_R = jnp.einsum('lilj->ij', r_reshaped)
    
    # 2. Form new rho_L (Pure state density matrix)
    rho_L = jnp.outer(psi_new_L, psi_new_L.conj())
    
    # 3. Tensor Product: rho_new = rho_L x rho_R
    return jnp.kron(rho_L, rho_R)
