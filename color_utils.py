import numpy as np
import matplotlib.colors as mcolors  # Import matplotlib colors
import plotly.express as px


def get_distinct_colors(n):
    """Generate n distinct colors, using a cycling method if n exceeds the base palette size."""
    base_colors = px.colors.qualitative.Dark24  # This is a palette of dark colors
    if n <= len(base_colors):
        return base_colors[:n]
    else:
        # Extend the color palette by repeating and modifying slightly
        colors = []
        cycle_count = int(np.ceil(n / len(base_colors)))
        for i in range(cycle_count):
            for color in base_colors:
                modified_color = lighten_color(color, amount=0.1 * i)
                colors.append(modified_color)
                if len(colors) == n:
                    return colors
    return colors

def lighten_color(color, amount=0.5):
    """Lighten color by a given amount. Amount > 0 to lighten, < 0 to darken."""
    try:
        c = mcolors.to_rgb(color)
        c = mcolors.rgb_to_hsv(c)
        c = (c[0], c[1], max(0, min(1, c[2] * (1 + amount))))
        c = mcolors.hsv_to_rgb(c)
        return mcolors.to_hex(c)
    except:
        print('Error: Invalid color: ', color)
        return color



