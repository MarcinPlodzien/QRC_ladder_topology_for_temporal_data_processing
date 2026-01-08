"""
Hamiltonian Templates Configuration
===================================
Author: Marcin Plodzien

This module defines the template dictionaries for different Quantum Reservoir Topologies.
These templates serve as the SCHEMA for the simulation engine (`utils.engine`), ensuring
that the Hamiltonian construction functions receive all expected parameters.

Design Philosophy:
------------------
- **Separation of Concerns**: The `engine` expects a standardized dictionary. The `config` script populates it.
- **Flexibility**: New topologies can be added by defining a new template here and a corresponding 
  construction function in `quantum_core.py`.
- **Default Values**: Templates initialize couplings to (0.0, 0.0, 0.0) to avoid KeyErrors if a specific 
  coupling type is not relevant for a topology (e.g., `J_rung` in All-to-All).

Current Topologies:
-------------------
1.  **Ladder (`Top_Ladder`)**:
    -   Two parallel rails (Left, Right) of length L.
    -   Field applied to each site.
    -   Couplings: `J_rail_left`, `J_rail_right` (Intra-rail), `J_rung` (Inter-rail).
    
2.  **All-to-All (`Top_All2All`)**:
    -   Fully connected graph of N = 2*L qubits.
    -   Field applied to each site.
    -   Coupling: `J_all` applied to every pair (i, j).

"""

def get_hamiltonian_library():
    """
    Returns the library of Hamiltonian templates.
    
    These dictionaries define the REQUIRED SCHEMA for the simulation engine.
    Specific coupling values (J, field) are typically overridden by the configuration script 
    (`config_qrc_ladder.py`) before being passed to the engine.
    
    Returns:
        list[dict]: A list containing the template dictionaries for Ladder and All-to-All.
    """
    
    # ==========================================================================
    # 1. LADDER TOPOLOGY TEMPLATE
    # ==========================================================================
    TEMPLATE_LADDER = {
        'config_name': 'Top_Ladder',
        'topology': 'ladder',
        
        # Rail Couplings (Left and Right Rails)
        # Tuple format: (Jx, Jy, Jz)
        'J_rail_left':  (0.0, 0.0, 0.0), # Default: No coupling
        'J_rail_right': (0.0, 0.0, 0.0),
        
        # Rung Couplings (Between Rails at same index i)
        'J_rung':       (0.0, 0.0, 0.0),
        
        # Unused in Ladder (kept for schema consistency if needed, else ignore)
        'J_all':        (0.0, 0.0, 0.0),
        
        # External Field (Applied to all sites)
        # Tuple format: (hx, hy, hz) magnitude multiplier usually handled by 'h_mag'
        'field':        (0.0, 0.0, 0.0)
    }

    # ==========================================================================
    # 2. ALL-TO-ALL TOPOLOGY TEMPLATE
    # ==========================================================================
    TEMPLATE_ALL2ALL = {
        'config_name': 'Top_All2All',
        'topology': 'all_to_all',
        
        # All-to-All Coupling (J_ij for all i<j)
        # Uniform coupling strength between all pairs
        'J_all':        (0.0, 0.0, 0.0),
        
        # Unused in All-to-All (kept for compatibility)
        'J_rail_left':  (0.0, 0.0, 0.0),
        'J_rail_right': (0.0, 0.0, 0.0),
        'J_rung':       (0.0, 0.0, 0.0),
        
        # External Field
        'field':        (0.0, 0.0, 0.0)
    }

    return [TEMPLATE_LADDER, TEMPLATE_ALL2ALL]
