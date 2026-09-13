""" Source: https://gitlab.cc-asp.fraunhofer.de/gnn4gc/pandapower-env/-/merge_requests/72 (code from mohamed.hassouna) """

from pathlib import Path
import h5py
import numpy as np
import pandas as pd

def read_h5_data(save_path: Path):
    """
    Read all states and generator power deltas from the HDF5 file.

    Args:
        save_path (Path): Path to the HDF5 file.

    Returns:
        list[dict]: A list of dictionaries, one per stored group. Each dictionary contains:
            - 'state' (np.ndarray): The state array.
            - 'gen_p_delta' (np.ndarray): The generator power delta array.

    Example:
        samples = read_h5_data(Path("data.h5"))
        grid2op_env.observation_space.from_vect(samples[0]['state'])
    """
    results = []
    # Open the HDF5 file in read-only mode
    with h5py.File(save_path, 'r') as f:
        # Iterate over all groups and extract stored data
        for group_name in f.keys():
            group = f[group_name]
            results.append({
                'state': group['state'][:],          # Load state array into memory
                'gen_p_delta': group['gen_p_delta'][:]  # Load generator power delta
            })
    return results

def read_h5_ts(save_path: Path, env):
    """
    Read one or multiple HDF5 files of saved states and gen_p_deltas,
    reconstruct Grid2Op observations, and return stacked arrays.

    If `save_path` is a directory, all `.h5` files inside are read in
    alphabetical order and concatenated along the time dimension.

    Parameters
    ----------
    save_path : Path
        Path to a single HDF5 file or a directory containing multiple HDF5 files.
    env : grid2op.Environment.Environment
        Grid2Op environment (provides the observation space).

    Returns
    -------
    dict with keys:
        - load_p : array (T, n_load)
        - load_q : array (T, n_load)
        - gen_p : array (T, n_gen)
        - gen_v : array (T, n_gen)
        - gen_p_delta : array (T, n_gen)
        - sgen_p : array (T, n_sgen) or None
    """
    load_p_list, load_q_list, gen_p_list, gen_p_delta_list, gen_v_list, sgen_p_list = [], [], [], [], [], []

    # determine list of files to read
    if save_path.is_dir():
        files = sorted(save_path.glob("*.h5"))
    else:
        files = [save_path]

    for file in files:
        with h5py.File(file, "r") as f:
            for group_name in f.keys():
                group = f[group_name]

                # reconstruct obs from stored state
                obs = env.observation_space.from_vect(group["state"][:])
                gen_p_delta = group["gen_p_delta"][:].squeeze()

                # collect arrays
                load_p_list.append(obs.load_p)
                load_q_list.append(obs.load_q)
                gen_p_list.append(obs.gen_p)
                gen_v_list.append(obs.gen_v)

                gen_p_delta_list.append(gen_p_delta)

                # sgen is optional in grid2op
                if hasattr(obs, "sgen_p"):
                    sgen_p_list.append(obs.sgen_p)

    # stack to shape (T, n_x)
    load_p = np.vstack(load_p_list)
    load_q = np.vstack(load_q_list)
    gen_p = np.vstack(gen_p_list)
    gen_v = np.vstack(gen_v_list)

    gen_p_delta = np.vstack(gen_p_delta_list)

    sgen_p = np.vstack(sgen_p_list) if sgen_p_list else None

    return {
        "load_p": load_p,
        "load_q": load_q,
        "gen_p": gen_p,
        "gen_v": gen_v,
        "gen_p_delta": gen_p_delta,
        "sgen_p": sgen_p,
    }



def attach_profiles_from_ts(net, load_p, load_q, gen_p, gen_p_delta, sgen_p=None):
    """
    Create net.profiles from arrays of time series values.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network, must contain net.load, net.gen, net.sgen.
    load_p : array-like, shape (T, n_load)
        Active power demand profiles.
    load_q : array-like, shape (T, n_load)
        Reactive power demand profiles.
    gen_p : array-like, shape (T, n_gen)
        Generator active power profiles.
    gen_p_delta : array-like, shape (T, n_gen) or (n_gen,)
        Values to subtract from gen_p.
    sgen_p : array-like, shape (T, n_sgen), optional
        Static generator (renewables) active power profiles.

    Returns
    -------
    net : pandapowerNet
        Modified network with net.profiles attached.
    """

    # --- Assign profile names (use element name) ---
    net.load["profile"] = net.load["name"]
    net.gen["profile"] = net.gen["name"]
    if not net.sgen.empty:
        net.sgen["profile"] = net.sgen["name"]

    # Make sure inputs are numpy arrays
    load_p = np.asarray(load_p)
    load_q = np.asarray(load_q)
    gen_p = np.asarray(gen_p)
    gen_p_delta = np.asarray(gen_p_delta)
    if sgen_p is not None:
        sgen_p = np.asarray(sgen_p)

    T = load_p.shape[0]
    timesteps = range(T)

    # --- Load profiles ---
    profiles_load = {}
    for i, name in enumerate(net.load["profile"]):
        profiles_load[f"{name}_pload"] = load_p[:, i]
        profiles_load[f"{name}_qload"] = load_q[:, i]
    df_profiles_load = pd.DataFrame(profiles_load, index=timesteps)

    # --- Generator profiles (subtract delta) ---
    # Support gen_p_delta as (T, n_gen) or (n_gen,)
    if gen_p_delta.ndim == 1:
        gen_p_delta = np.tile(gen_p_delta, (T, 1))
    df_profiles_gen = pd.DataFrame(
        {
            name: gen_p[:, i] - gen_p_delta[:, i]
            for i, name in enumerate(net.gen["profile"])
        },
        index=timesteps
    )

    # --- Static generator (renewables) profiles ---
    if sgen_p is not None and not net.sgen.empty:
        df_profiles_sgen = pd.DataFrame(
            {
                name: sgen_p[:, i]
                for i, name in enumerate(net.sgen["profile"])
            },
            index=timesteps
        )
    else:
        df_profiles_sgen = pd.DataFrame(index=timesteps)

    # --- Attach to net ---
    net.profiles = {
        "load": df_profiles_load,
        "renewables": df_profiles_sgen,
        "powerplants": df_profiles_gen,
    }
    net.load["p_mw"] = 1
    net.load["q_mvar"] = 1
    net.gen["p_mw"] = 1
    return net
