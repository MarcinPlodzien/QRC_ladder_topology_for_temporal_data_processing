def generate_result_filename(config, iter_idx):
    """
    Generates a structured filename for simulation results.
    
    Format:
    result_Topology#{name}_N#{n}_T#{t_evol}_Int#{method}_Jr#{jr}_JrL#{jrl}_JrR#{jrr}_FL#{fl}_FR#{fr}_In#{input}_Data#{data}_Iter#{iter}.pkl
    """
    # Helper to sanitize values
    def fmt(val):
#        return str(val).replace('.', 'p')
        return str(val)
        
    # Extract Parameters
    # Assumes 'param_names' dict exists in config, or falls back to 'name' parsing if needed.
    # We will ensure 'param_names' is populated in 01_config.
    
    params = config.get('param_names', {})
    
    topology = "Ladder" # Fixed for now or extracted if variable
    N = fmt(config.get('N', 0))
    t_evol = fmt(config.get('t_evol', 0))
    integrator = config.get('integration_method', 'Unknown')
    
    # Couplings
    jr = params.get('J_rungs', 'Unk')
    jrl = params.get('J_rail_left', 'Unk')
    jrr = params.get('J_rail_right', 'Unk')
    
    # Fields
    fl = params.get('field_L', 'Unk')
    fr = params.get('field_R', 'Unk')
    
    # Input/Data
    inp_state = config.get('input_state_type', 'Unk')
    data_mode = config.get('data_input_type', 'Unk')
    
    # Disorder
    is_disordered = config.get('field_disorder', True)
    dis_str = "Dis" if is_disordered else "Uni"
    
    # Construct parts
    parts = [
        "result",
        f"Topology#{topology}",
        f"N#{N}",
        f"T_evol#{t_evol}",
        f"Int#{integrator}",
        f"J_rung#{jr}",
        f"J_rail_left#{jrl}",
        f"J_rail_right#{jrr}",
        f"field_rail_Left#{fl}",
        f"field_rail_Right#{fr}",
        f"InputState#{inp_state}",
        f"DataMode#{data_mode}",
        f"Disorder#{dis_str}",
        f"Iter#{iter_idx}"
    ]
    
    filename = "_".join(parts) + ".pkl"
    return filename
