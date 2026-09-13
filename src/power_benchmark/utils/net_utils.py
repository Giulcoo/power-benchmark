import pandapower as pp
import pandas as pd

# ---------- Net Equality -------------
def networks_equal(net1: None | pp.pandapowerNet, net2: None | pp.pandapowerNet, epsilon: float = None) -> bool:
    """
    Compare two pandapower networks for equality
    """
    if net1 is None or net2 is None:
        return net1 is net2
    elif not isinstance(net1, pp.pandapowerNet) or not isinstance(net2, pp.pandapowerNet):
        return False
    else:
        # See pandapower-env/environments/simulation_env.py function load_actions
        return _compare_closed_switches(net1, net2) and _compare_in_service_lines(net1, net2) and _compare_trafo_tap_positions(net1, net2, epsilon=epsilon)

def _compare_closed_switches(net1: pp.pandapowerNet, net2: pp.pandapowerNet) -> bool:
    """
        Compare switches of two pandapower networks for equality in open/closed switches.
        Only compares same switches (based on bus and element).
    """
    return _compare_dfs(net1.switch, net2.switch, id_columns=['bus', 'element'], columns_to_compare = ['closed'])

def _compare_in_service_lines(net1: pp.pandapowerNet, net2: pp.pandapowerNet) -> bool:
    """
        Compare lines of two pandapower networks for equality in "in service" lines.
        Only compares same lines (based on bus and element).
    """
    return _compare_dfs(net1.line, net2.line, id_columns=['from_bus', 'to_bus'], columns_to_compare = ['in_service'])

def _compare_trafo_tap_positions(net1: pp.pandapowerNet, net2: pp.pandapowerNet, epsilon: float = None) -> bool:
    """
        Compare transformers of two pandapower networks for equality in tap positions.
        Only compares same transformers (based on hv_bus and lv_bus).
    """
    return _compare_dfs(net1.trafo, net2.trafo, id_columns=['hv_bus', 'lv_bus'], columns_to_compare = ['tap_pos'], epsilon=epsilon)

def _compare_dfs(df1: pd.DataFrame, df2: pd.DataFrame, id_columns: list[str], columns_to_compare: list[str], epsilon: float = None) -> bool:
    # Check if one or both DataFrames are empty
    if df1.empty or df2.empty:
        return df1.empty and df2.empty

    # Drop irrelevant columns for comparison
    df1 = df1[id_columns + columns_to_compare]
    df2 = df2[id_columns + columns_to_compare]

    # Find common switches based on 'bus' and 'element'
    common_switches = pd.merge(df1, df2, on=id_columns, how='inner')

    # Split common switches by "_x" and "_y" suffixes
    df1_common = common_switches[[col + "_x" for col in columns_to_compare]]
    df2_common = common_switches[[col + "_y" for col in columns_to_compare]]

    # Rename columns to remove suffixes
    df1_common.columns = [col[:-2] for col in df1_common.columns]
    df2_common.columns = [col[:-2] for col in df2_common.columns]

    if epsilon is None:
        return df1_common.equals(df2_common)
    else:
        return (df1_common - df2_common).abs().le(epsilon).all().all()

# ---------- Net Changes -------------
def count_closed_switches_changes(net1: pp.pandapowerNet, net2: pp.pandapowerNet) -> int:
    """
        Compare switches of two pandapower networks for equality in open/closed switches.
        Only compares same switches (based on bus and element).
    """
    return _count_changes(net1.switch, net2.switch, id_columns=['bus', 'element'], columns_to_compare = ['closed'])

def count_in_service_lines_changes(net1: pp.pandapowerNet, net2: pp.pandapowerNet) -> int:
    """
        Compare lines of two pandapower networks for equality in "in service" lines.
        Only compares same lines (based on bus and element).
    """
    return _count_changes(net1.line, net2.line, id_columns=['from_bus', 'to_bus'], columns_to_compare = ['in_service'])

def count_trafo_tap_positions_changes(net1: pp.pandapowerNet, net2: pp.pandapowerNet, epsilon: float = None) -> int:
    """
        Compare transformers of two pandapower networks for equality in tap positions.
        Only compares same transformers (based on hv_bus and lv_bus).
    """
    return _count_changes(net1.trafo, net2.trafo, id_columns=['hv_bus', 'lv_bus'], columns_to_compare = ['tap_pos'], epsilon=epsilon)

def _count_changes(df1: pd.DataFrame, df2: pd.DataFrame, id_columns: list[str], columns_to_compare: list[str], epsilon: float = None) -> int:
    # Check if one or both DataFrames are empty
    if df1.empty or df2.empty:
        return 0

    # Drop irrelevant columns for comparison
    df1 = df1[id_columns + columns_to_compare]
    df2 = df2[id_columns + columns_to_compare]

    # Find common switches based on 'bus' and 'element'
    common_switches = pd.merge(df1, df2, on=id_columns, how='inner')

    # Split common switches by "_x" and "_y" suffixes
    df1_common = common_switches[[col + "_x" for col in columns_to_compare]]
    df2_common = common_switches[[col + "_y" for col in columns_to_compare]]

    # Rename columns to remove suffixes
    df1_common.columns = [col[:-2] for col in df1_common.columns]
    df2_common.columns = [col[:-2] for col in df2_common.columns]

    if epsilon is None:
        return (df1_common != df2_common).any(axis=1).sum()
    else:
        return ((df1_common - df2_common).abs().gt(epsilon)).any(axis=1).sum()


# ---------- Net Voltages -------------
def get_voltage_violations(
        net: pp.pandapowerNet,
        default_v_min_pu: float = 0.95,
        default_v_max_pu: float = 1.05
) -> pd.Series:
    """
    Get voltage violations per bus.

    Returns:
        pd.Series of voltage violations aligned with res_bus index.
        Zero if within limits, positive value indicating violation amount otherwise.
    """
    vm_pu = net.res_bus.vm_pu.dropna()
    v_min, v_max = _get_voltage_limits(net, default_v_min_pu, default_v_max_pu)

    # Align limits with available voltage measurements
    v_min = v_min.loc[vm_pu.index]
    v_max = v_max.loc[vm_pu.index]

    # Calculate violations
    violations = pd.Series(0.0, index=vm_pu.index)
    violations += (v_min - vm_pu).clip(lower=0)  # Under-voltage violations
    violations += (vm_pu - v_max).clip(lower=0)  # Over-voltage violations

    # Remove zero violations
    violations = violations[violations > 0]

    return violations

def _get_voltage_limits(
        net: pp.pandapowerNet,
        default_v_min_pu: float = 0.95,
        default_v_max_pu: float = 1.05,
) -> tuple[pd.Series, pd.Series]:
    """
    Get voltage limits per bus.

    Uses min_vm_pu/max_vm_pu from net.bus if available,
    otherwise falls back to default values from parameters.

    Args:
        net: pandapower network
        default_v_min_pu: default minimum voltage in pu if not specified in net.bus
        default_v_max_pu: default maximum voltage in pu if not specified in net.bus

    Returns:
        Tuple of (v_min, v_max) as pd.Series aligned with res_bus index.
    """
    bus_df = net.bus
    res_bus_index = net.res_bus.index

    # Get min voltage limits
    if 'min_vm_pu' in bus_df.columns:
        v_min = bus_df.loc[res_bus_index, 'min_vm_pu'].fillna(default_v_min_pu)
    else:
        v_min = pd.Series(default_v_min_pu, index=res_bus_index)

    # Get max voltage limits
    if 'max_vm_pu' in bus_df.columns:
        v_max = bus_df.loc[res_bus_index, 'max_vm_pu'].fillna(default_v_max_pu)
    else:
        v_max = pd.Series(default_v_max_pu, index=res_bus_index)

    return v_min, v_max

def get_voltage_deviations(
        net: pp.pandapowerNet,
        nominal_voltage: float = 1.0,
) -> pd.Series:
    """
    Get voltage deviations from limits per bus.

    Returns:
        pd.Series of voltage deviations aligned with res_bus index.
    """
    vm_pu = net.res_bus.vm_pu.dropna()
    return (vm_pu - 1.0).abs()


# ---------- Net Transformer Loadings -------------
def trafo_loading_max(net: pp.pandapowerNet) -> float:
    """Return the maximum transformer loading across all transformers."""
    max_loading = 0.0

    if len(net.res_trafo) > 0:
        loading = net.res_trafo.loading_percent.dropna()
        if len(loading) > 0:
            max_loading = max(max_loading, loading.max())

    if len(net.res_trafo3w) > 0:
        loading = net.res_trafo3w.loading_percent.dropna()
        if len(loading) > 0:
            max_loading = max(max_loading, loading.max())

    return max_loading

# ---------- Line Overloadings -------------
def amount_of_overloaded_lines(net: pp.pandapowerNet, threshold: float = 100.0) -> int:
    """Return the number of lines with loading above the specified threshold."""
    if len(net.res_line) == 0:
        return 0
    loading = net.res_line.loading_percent.dropna()
    overloaded_lines = loading[loading > threshold]
    return len(overloaded_lines)

def amount_of_overloaded_regions(net: pp.pandapowerNet, threshold: float = 100.0) -> int:
    """Return the number of unique regions with at least one line overloaded above the specified threshold."""
    if len(net.res_line) == 0 or 'region' not in net.line.columns:
        return 0
    loading = net.res_line.loading_percent.dropna()
    overloaded_lines = net.line.loc[loading[loading > threshold].index]
    return _find_amount_of_regions(overloaded_lines)

def _find_amount_of_regions(lines: pd.DataFrame) -> int:
    """Receives a list of lines and looks for connected components, returning the number of regions found."""
    from collections import defaultdict, deque

    # Build adjacency list
    adjacency = defaultdict(set)
    for _, line in lines.iterrows():
        from_bus = line['from_bus']
        to_bus = line['to_bus']
        adjacency[from_bus].add(to_bus)
        adjacency[to_bus].add(from_bus)

    visited = set()
    region_count = 0

    # BFS to find connected components
    for bus in adjacency:
        if bus not in visited:
            region_count += 1
            queue = deque([bus])
            visited.add(bus)

            while queue:
                current_bus = queue.popleft()
                for neighbor in adjacency[current_bus]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

    return region_count


def max_possible_regions(net: pp.pandapowerNet) -> int:
    """
    Return the maximum number of non-adjacent regions possible in the network.

    Two regions are adjacent if there exists a line connecting a bus from
    one region to a bus from another region. Non-adjacent regions must have
    at least one "buffer bus" between them.
    """
    import networkx as nx

    if len(net.line) == 0:
        return 0

    # Build extended line graph:
    # - Nodes = lines
    # - Edges = lines that share a bus OR lines whose buses are directly connected
    line_graph = nx.Graph()
    line_graph.add_nodes_from(net.line.index)

    # Map: bus -> list of line indices
    bus_to_lines = {}
    for idx, line in net.line.iterrows():
        for bus in [line['from_bus'], line['to_bus']]:
            bus_to_lines.setdefault(bus, []).append(idx)

    # Connect lines that share a bus or are one hop apart
    for idx, line in net.line.iterrows():
        from_bus, to_bus = line['from_bus'], line['to_bus']

        neighbors = set(bus_to_lines.get(from_bus, [])) | set(bus_to_lines.get(to_bus, []))
        neighbors.discard(idx)

        for neighbor_idx in neighbors:
            line_graph.add_edge(idx, neighbor_idx)

    # Maximum Independent Set
    max_independent_set = nx.approximation.maximum_independent_set(line_graph)

    return len(max_independent_set)


def max_possible_regions_lower_bound(net: pp.pandapowerNet) -> int:
    """
    Lower bound using Caro-Wei theorem. Tighter than n/(Δ+1), still O(L) runtime.
    """
    if len(net.line) == 0:
        return 0

    # Count lines per bus
    bus_line_count = {}
    for _, line in net.line[['from_bus', 'to_bus']].iterrows():
        bus_line_count[line['from_bus']] = bus_line_count.get(line['from_bus'], 0) + 1
        bus_line_count[line['to_bus']] = bus_line_count.get(line['to_bus'], 0) + 1

    # Caro-Wei sum
    caro_wei_sum = 0.0
    for _, line in net.line[['from_bus', 'to_bus']].iterrows():
        degree = bus_line_count[line['from_bus']] + bus_line_count[line['to_bus']] - 2
        caro_wei_sum += 1.0 / (degree + 1)

    return int(caro_wei_sum)  # Floor

def has_gens(net: pp.pandapowerNet) -> bool:
    return len(net.gen) > 0

def has_sgens(net: pp.pandapowerNet) -> bool:
    return len(net.sgen) > 0

def has_trafo_any_type(net: pp.pandapowerNet) -> bool:
    """ Checks if net has 2 or 3 winding transformers. """
    return has_trafos(net) or has_trafos3w(net)

def has_trafos(net: pp.pandapowerNet) -> bool:
    return len(net.trafo) > 0

def has_trafos3w(net: pp.pandapowerNet) -> bool:
    return len(net.trafo3w) > 0