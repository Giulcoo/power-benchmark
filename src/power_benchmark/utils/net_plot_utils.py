import pandapower as pp
import pandapower.topology as top
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import json
from matplotlib.patches import Patch
import pandas as pd

import json
from matplotlib.patches import Patch

GEN_COLOR = 'dimgray'
SGEN_COLOR = 'lightgreen'
LOAD_COLOR = 'lightcoral'
BUS_COLOR = 'skyblue'

def plot_net(net: pp.pandapowerNet, size=(5, 5), min_distance=0.05):
    mg = top.create_nxgraph(net)
    plt.figure(figsize=size)

    # Extract geo positions from net.bus
    pos = {}
    for idx, row in net.bus.iterrows():
        if pd.notna(row['geo']):
            if isinstance(row['geo'], str):
                geo_data = json.loads(row['geo'])
            else:
                geo_data = row['geo']

            coords = geo_data['coordinates']
            pos[idx] = np.array([coords[0], coords[1]])

    # Space out nodes with similar positions
    if len(pos) > 0 and min_distance is not None:
        pos = space_out_close_nodes(pos, min_distance)

    gen_buses = set(net.gen['bus'].values) if len(net.gen) > 0 else set()
    sgen_buses = set(net.sgen['bus'].values) if len(net.sgen) > 0 else set()
    load_buses = set(net.load['bus'].values) if len(net.load) > 0 else set()

    node_colors = []
    for node in mg.nodes():
        if node in sgen_buses:
            node_colors.append(SGEN_COLOR)
        elif node in gen_buses:
            node_colors.append(GEN_COLOR)
        elif node in load_buses:
            node_colors.append(LOAD_COLOR)
        else:
            node_colors.append(BUS_COLOR)

    if len(pos) == 0:
        pos = nx.spring_layout(mg)

    nx.draw(mg, pos, with_labels=True, node_color=node_colors, node_size=500,
            edge_color='gray', font_size=8, font_color='black')

    legend_elements = [
        Patch(facecolor=GEN_COLOR, label='Generator'),
        Patch(facecolor=SGEN_COLOR, label='Renewable Generator'),
        Patch(facecolor=LOAD_COLOR, label='Load'),
        Patch(facecolor=BUS_COLOR, label='Bus')
    ]
    plt.legend(handles=legend_elements, loc='best')
    plt.axis('equal')
    plt.show()


def space_out_close_nodes(pos, min_distance=0.05, max_iterations=50):
    """
    Adjust positions of nodes that are too close to each other.

    Args:
        pos: Dictionary of node positions {node_id: np.array([x, y])}
        min_distance: Minimum distance between nodes
        max_iterations: Maximum number of adjustment iterations

    Returns:
        Adjusted positions dictionary
    """
    pos_adjusted = {k: v.copy() for k, v in pos.items()}
    nodes = list(pos_adjusted.keys())

    for iteration in range(max_iterations):
        moved = False

        for i, node1 in enumerate(nodes):
            for node2 in nodes[i + 1:]:
                diff = pos_adjusted[node1] - pos_adjusted[node2]
                distance = np.linalg.norm(diff)

                if distance < min_distance and distance > 0:
                    # Calculate push direction
                    direction = diff / distance
                    # Push nodes apart
                    push = direction * (min_distance - distance) / 2
                    pos_adjusted[node1] += push
                    pos_adjusted[node2] -= push
                    moved = True
                elif distance == 0:
                    # Nodes at exactly the same position - add random offset
                    offset = np.random.randn(2) * min_distance / 2
                    pos_adjusted[node1] += offset
                    pos_adjusted[node2] -= offset
                    moved = True

        if not moved:
            break

    return pos_adjusted

def plot_bottleneck_net(net: pp.pandapowerNet, size=None, gen_spacing=0.3, bottleneck_spacing=0.15, load_spacing=0.3):
    # Create position dictionary
    pos = {}

    # Categorize buses
    gen_buses: list[tuple[int, str]] = []
    bottleneck_layers = {}  # Dictionary: layer_idx -> list of bus indices
    load_buses = []

    for bus_idx in net.bus.index:
        bus_name = net.bus.at[bus_idx, 'name']
        if 'Generator' in bus_name and 'Static' not in bus_name:
            gen_buses.append((bus_idx, "gen"))
        if 'Static Generator' in bus_name:
            gen_buses.append((bus_idx, "sgen"))
        elif 'Layer' in bus_name:
            # Extract layer number from name (e.g., "Layer_0_Node_3" -> 0)
            layer_idx = int(bus_name.split('_')[1])
            if layer_idx not in bottleneck_layers:
                bottleneck_layers[layer_idx] = []
            bottleneck_layers[layer_idx].append(bus_idx)
        elif 'Load' in bus_name:
            load_buses.append(bus_idx)

    if size is None:
        n_load = len(load_buses)
        n_gen = len(gen_buses)
        n_bottleneck_layers = len(bottleneck_layers)
        max_bottleneck_size = max(len(buses) for buses in bottleneck_layers.values()) if bottleneck_layers else 0
        max_height = max(n_load, n_gen, max_bottleneck_size)
        height = max_height * max(gen_spacing, bottleneck_spacing, load_spacing) + 1
        width = (n_bottleneck_layers + 2) * (max_height * 0.03) + 5
        size = (width, height)

    mg = top.create_nxgraph(net)
    plt.figure(figsize=size)

    # Sort layers by index
    sorted_layers = sorted(bottleneck_layers.keys())
    num_bottleneck_layers = len(sorted_layers)

    # Position GEN and SGEN nodes in first column (x=0)
    y_positions = get_centered_positions(len(gen_buses), spacing=gen_spacing)
    for i, (bus, gen_type) in enumerate(gen_buses):
        pos[bus] = (0, y_positions[i])

    # Position each bottleneck layer in its own column
    for col_idx, layer_idx in enumerate(sorted_layers):
        layer_buses = bottleneck_layers[layer_idx]
        x_pos = col_idx + 1  # Start at x=1 after generators

        y_positions = get_centered_positions(len(layer_buses), spacing=bottleneck_spacing)
        for i, bus in enumerate(layer_buses):
            pos[bus] = (x_pos, y_positions[i])

    # Position load nodes in last column
    x_load = num_bottleneck_layers + 1
    y_positions = get_centered_positions(len(load_buses), spacing=load_spacing)
    for i, bus in enumerate(load_buses):
        pos[bus] = (x_load, y_positions[i])

    # Create node colors with different shades for each bottleneck layer
    # Generate color palette for bottleneck layers
    bottleneck_colors = plt.cm.Oranges(np.linspace(0.4, 0.8, num_bottleneck_layers))

    node_colors = []
    for node in mg.nodes():
        if (node, "gen") in gen_buses:
            node_colors.append(GEN_COLOR)
        elif (node, "sgen") in gen_buses:
            node_colors.append(SGEN_COLOR)
        elif node in load_buses:
            node_colors.append(LOAD_COLOR)
        else:
            # Find which layer this node belongs to
            for col_idx, layer_idx in enumerate(sorted_layers):
                if node in bottleneck_layers[layer_idx]:
                    node_colors.append(bottleneck_colors[col_idx])
                    break
            else:
                node_colors.append(BUS_COLOR)

    # Draw the network
    nx.draw(mg, pos, with_labels=True, node_color=node_colors,
            node_size=500, edge_color='gray', font_size=8,
            font_color='black', arrows=True, arrowsize=10, width=1.5)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=GEN_COLOR, label='Generators'),
        Patch(facecolor=SGEN_COLOR, label='Renewable Generators')
    ]

    for col_idx, layer_idx in enumerate(sorted_layers):
        legend_elements.append(
            Patch(facecolor=bottleneck_colors[col_idx], label=f'Bottleneck Layer {layer_idx}')
        )

    legend_elements.append(Patch(facecolor=LOAD_COLOR, label='Loads'))

    plt.legend(handles=legend_elements, loc='upper right')

    plt.title("Bottleneck Network Topology")
    plt.tight_layout()
    plt.show()

def get_centered_positions(n_nodes, spacing=0.15):
        """
        Calculate y-positions for nodes, centered vertically with equal spacing.

        Parameters:
        -----------
        n_nodes : int
            Number of nodes to position
        spacing : float
            Desired spacing between nodes (adjusted if necessary)

        Returns:
        --------
        list of float
            Y-positions for each node, centered around 0.5
        """
        if n_nodes == 1:
            return [0.5]

        # Calculate total height needed
        total_height = (n_nodes - 1) * spacing

        # Adjust spacing if it exceeds available space (with margin)
        max_height = 0.9  # Leave 10% margin (5% top, 5% bottom)
        if total_height > max_height:
            spacing = max_height / (n_nodes - 1)
            total_height = max_height

        # Center around y=0.5
        y_start = 0.5 - total_height / 2

        return [y_start + i * spacing for i in range(n_nodes)]