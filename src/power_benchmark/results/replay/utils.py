import json

import matplotlib
import matplotlib.colors as mcolors
import pandapower as pp


class CoordinateHelper:
    """Handles coordinate extraction from GeoJSON, Shapely geometries, or raw lists."""

    @staticmethod
    def extract_coords(geo_value) -> list[list[float]]:
        """Extract coordinate pairs from a GeoJSON string, Shapely geometry, or raw list."""
        if isinstance(geo_value, str):
            return json.loads(geo_value)["coordinates"]
        if hasattr(geo_value, "coords"):  # Shapely geometry
            return list(geo_value.coords)
        return geo_value  # already a list of pairs

    @staticmethod
    def get_bus_coords(net: pp.pandapowerNet, bus: int):
        """Parse coordinates from the geo column of net.bus."""
        if bus not in net.bus.index:
            return None, None

        geo_val = net.bus.at[bus, "geo"]
        if geo_val is None:
            return None, None

        try:
            geo_dict = json.loads(geo_val)
            coords = geo_dict["coordinates"]  # [x, y]
            return coords[0], coords[1]
        except (json.JSONDecodeError, KeyError, TypeError, IndexError):
            return None, None


class ColorMapHelper:
    """Handles matplotlib colormap creation and conversion to Plotly colorscales."""

    @staticmethod
    def mpl_cmap_to_plotly(cmap_name: str, n: int = 10) -> list:
        """Convert a registered matplotlib colormap to a Plotly colorscale."""
        cmap = matplotlib.colormaps[cmap_name]
        colorscale = []
        for i in range(n):
            pos = i / (n - 1)
            r, g, b, a = cmap(pos)
            colorscale.append([
                pos,
                f"rgba({float(r) * 255:.1f},{float(g) * 255:.1f},{float(b) * 255:.1f},{float(a):.2f})"
            ])
        return colorscale

    @staticmethod
    def get_line_loading_cmap(cmax: float = 200) -> str:
        """Create and register a custom line-loading colormap, returning its name."""
        p0 = 0.0
        p50 = 50 / cmax
        p100 = 100 / cmax
        p_mid = (100 + cmax) / (2 * cmax)
        p_max = 1.0

        color_stops = [
            (p0, "green"),
            (p50, "gold"),
            (p100, "orange"),
            (p_mid, "red"),
            (p_max, "black"),
        ]

        cmap_name = f"line_loading_cmap_{int(cmax)}"
        cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, color_stops)

        try:
            matplotlib.colormaps.register(cmap, name=cmap_name)
        except ValueError:
            pass  # already registered

        return cmap_name