# Quantum Reservoir Computing (QRC) on Spin Ladders: Protocol & Implementation

## 1. Introduction

This project implements a high-performance simulation framework for **Quantum Reservoir Computing (QRC)** using spin-1/2 systems arranged in a **Ladder Topology**. The primary objective is to evaluate the memory and prediction capacity of quantum substrates processing temporal time-series information.

By leveraging **JAX** for GPU/CPU acceleration and automatic differentiation (vectorization), the framework allows for the efficient exploration of:
*   **Topological variations**: Tunable couplings between reservoir "rails" and input "rails".
*   **Input encoding strategies**: Single-qubit injection vs. Sliding-window (Batch) embedding.
*   **Disorder effects**: The role of Anderson localization in preserving information.

---

## 2. Physical System: The Spin Ladder

The quantum reservoir is modeled as a spin ladder consisting of two coupled 1D chains (rails) of length $L$. The total system size is $N = 2L$.

### 2.1. Topology
The system is divided into two distinct sub-systems:
1.  **Input Rail ($I$)**: A 1D chain where external information is injected.
2.  **Reservoir Rail ($R$)**: A 1D chain that acts as the primary processing substrate, coupled to the input rail.

The Hamiltonian is defined as:
$$ H = H_{rail}^{(I)} + H_{rail}^{(R)} + H_{rung} + H_{field} $$

### 2.2. Hamiltonian Interaction Terms
The couplings are generally of the Heisenberg or XXZ type:

$$ H_{rail}^{(I/R)} = \sum_{i=1}^{L-1} \sum_{\alpha \in \{x,y,z\}} J_{rail}^{\alpha} \sigma_i^\alpha \sigma_{i+1}^\alpha $$

$$ H_{rung} = \sum_{i=1}^{L} \sum_{\alpha \in \{x,y,z\}} J_{rung}^{\alpha} \sigma_{i, I}^\alpha \sigma_{i, R}^\alpha $$

$$ H_{field} = \sum_{i=1}^{N} \sum_{\alpha \in \{x,y,z\}} h_i^\alpha \sigma_i^\alpha $$

Where:
*   $\sigma^\alpha$ are Pauli matrices.
*   $J_{rail}$ controls intra-chain dynamics (transport).
*   $J_{rung}$ controls inter-chain information transfer.
*   $h_i$ represents external magnetic fields (Uniform or Disordered).

---

## 3. Operational Protocol

The QRC protocol operates in discrete time steps $k = 1, \dots, T$. Each step involves four distinct phases: **Reset**, **Injection**, **Evolution**, and **Measurement**.

### 3.1. Discrete Time Cycle

For each time step $k$:

1.  **Input Reset & Preparation**:
    The state of the **Input Rail** is traced out and replaced with a fresh product state encoding the input signal $u_k$. The Reservoir Rail is **untouched**, preserving its memory.
    $$ \rho_{total}^{(k)} = \rho_{input}(u_k) \otimes \text{Tr}_{input} \left( \rho_{total}^{(k-1)} \right) $$

2.  **Input Encoding ($\rho_{input}$)**:
    The input data $u_k$ is encoded into the input rail. Two strategies are implemented:
    
    *   **Single Data Input** (`single_data_input`):
        The scalar value $u_k$ is broadcasted to all sites on the input rail.
        $$ \ket{\psi_{in}} = \bigotimes_{i=1}^L \left( \sqrt{1-u_k}\ket{0} + \sqrt{u_k}\ket{1} \right) $$
        *(Note: Input $u_k$ is normalized to $[0, 1]$).*

    *   **Batch (Sliding Window) Input** (`batch_data_input`):
        A history window of length $L$ is encoded spatially across the input rail.
        $$ \vec{u}_k = [u_k, u_{k-1}, \dots, u_{k-L+1}] $$
        Site $j$ encodes $u_{k-j}$. This creates a **Spatial-Temporal Embedding**.

3.  **Unitary Evolution**:
    The coupled system evolves for a duration $t_{evol}$ under the full Hamiltonian $H$.
    $$ \rho_{total}^{(k)'} = e^{-i H t_{evol}} \rho_{total}^{(k)} e^{i H t_{evol}} $$

4.  **Measurement (Readout)**:
    Observable expectations are collected from the **Reservoir Rail** (and optionally the Input Rail).
    $$ x_{k, i}^\alpha = \text{Tr}(\sigma_i^\alpha \rho_{total}^{(k)'}) $$
    Common observables: $\langle Z_i \rangle, \langle X_i \rangle, \langle Z_i Z_{i+1} \rangle$.

---

## 4. Implementation Details

### 4.1. Scientific Stack
*   **Language**: Python 3.10+
*   **Core Logic**: `JAX` (Google's XLA framework) for vectorization (`vmap`) and compilation/optimization (`jit`).
*   **Data Handling**: `pandas` DataFrame for structured results, `pickle` for serialization.

### 4.2. Time Integration
The framework supports multiple integration schemes (`utils/time_integrator.py`):
1.  **Exact Diagonalization (`exact_eig`)**:
    *   Computes full eigendecomposition $H = U \Lambda U^\dagger$.
    *   Evolution is exact: $U(t) = U e^{-i \Lambda t} U^\dagger$.
    *   Best for small systems ($N \le 12$) or long evolution times.
2.  **Runge-Kutta 4 (`rk4`)**:
    *   Standard numerical integration for general time-dependent Hamiltonians.
3.  **Trotterization (`trotter`)**:
    *   Suzuki-Trotter decomposition for sparse evolution (not fully active in current config).

### 4.3. Parallelization
*   **Orchestration**: `00_runner_parallel_CPU.py`.
*   **Mechanism**: `concurrent.futures.ProcessPoolExecutor` distributes simulation tasks across CPU cores.
*   **JAX Config**: `JAX_PLATFORM_NAME='cpu'` is forced to avoid GPU context switching overheads when running many small parallel processes.
*   **Realization Averaging**: For disordered systems, $N_{real}$ separate realizations are run in parallel, and results are statistically aggregated (Mean $\pm$ Std).

---

## 5. Benchmark Task: Mackey-Glass Prediction

The standard benchmark is the prediction of the chaotic **Mackey-Glass** time series.

*   **Task**: Given inputs $u_0, \dots, u_t$, predict future value $u_{t+\tau}$.
*   **Process**:
    1.  **Harvesting**: Run simulation for $T$ steps, collecting reservoir state vectors $\mathbf{X}_t$.
    2.  **Training**: Fit a linear readout layer $\mathbf{W}_{out}$ using Ridge Regression (Tikhonov Regularization).
        $$ \mathbf{W}_{out} = \text{argmin}_{\mathbf{W}} || \mathbf{W}\mathbf{X} - \mathbf{Y}_{target} ||^2 + \alpha ||\mathbf{W}||^2 $$
    3.  **Evaluation**: Calculate **Memory Capacity** as the squared correlation coefficient ($R^2$) summed over delays $\tau$.
        $$ C_{total} = \sum_{\tau=1}^{\tau_{max}} C(\tau) = \sum_{\tau} \text{cov}^2(y_{pred}, y_{true}) $$

---

## 6. Key Mechanisms and Findings

### 6.1. The Role of Topological Disorder vs. Spatial Embedding
A key scientific finding from this implementation is the **Spatial-Temporal Trade-off**:

*   **Diffusive/Thermal Regime**: With **Single Data Input** (scalar injection) and **Uniform couplings**, the reservoir thermalizes/delocalizes rapidly. Information about past inputs dissipates into the bulk Hilbert space, leading to poor long-term memory ($C_{total}$ drops at large $t_{evol}$).
    *   *Solution*: Adding **Resultant Disorder** (Random Fields) creates **Anderson Localization**. Localized modes protect information from dissipation, preserving memory even at long evolution times.

*   **Spatial Embedding Regime**: With **Batch Data Input** (Sliding Window), history is explicitly encoded into the *spatial state* of the input rail ($I$).
    *   *Observation*: In this regime, **Disorder is redundant**. The reservoir "sees" the history directly at every step. Uniform fields perform equally well as disordered ones.
    *   *Conclusion*: Increasing input dimensionality (spatial width $L$) can substitute for the need for complex internal memory mechanisms (disorder/fading memory).

---

**Author**: Project Team (Quantum Reservoir Computing Group)
**Date**: February 2026
