#!/usr/bin/env python3
"""
View the exploration map saved by smart_explore.py
"""

import numpy as np
import sys

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def view_map(map_file='/home/bo/robot_pet/exploration_map.npy'):
    print(f"Loading map from {map_file}...")

    try:
        grid = np.load(map_file)
    except FileNotFoundError:
        print("Error: Map file not found. Run smart_explore.py first.")
        return

    print(f"Grid size: {grid.shape}")
    print(f"Unknown cells: {np.sum(grid == 0)}")
    print(f"Free cells: {np.sum(grid == 1)}")
    print(f"Obstacle cells: {np.sum(grid == 2)}")

    # Calculate explored area
    resolution = 0.05  # 5cm per cell
    explored_area = np.sum(grid > 0) * (resolution ** 2)
    print(f"Explored area: {explored_area:.2f} m²")

    if not HAS_MATPLOTLIB:
        print("\nMatplotlib not available. Showing ASCII map:")
        print_ascii_map(grid)
        return

    # Create visualization
    print("\nGenerating map image...")

    # Color map: black=unknown, white=free, red=obstacle
    rgb_map = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
    rgb_map[grid == 0] = [50, 50, 50]    # Unknown = dark gray
    rgb_map[grid == 1] = [255, 255, 255]  # Free = white
    rgb_map[grid == 2] = [255, 0, 0]      # Obstacle = red

    # Mark center (start position)
    center = grid.shape[0] // 2
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            rgb_map[center+dx, center+dy] = [0, 255, 0]  # Green = start

    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_map, origin='lower')
    plt.title('Exploration Map\n(Green=Start, White=Free, Red=Obstacle, Gray=Unknown)')
    plt.xlabel('X (grid cells)')
    plt.ylabel('Y (grid cells)')

    # Add scale bar
    scale_cells = int(1.0 / resolution)  # 1 meter in cells
    plt.plot([10, 10 + scale_cells], [10, 10], 'b-', linewidth=3)
    plt.text(10 + scale_cells//2, 15, '1m', ha='center', color='blue')

    output_file = '/home/bo/robot_pet/exploration_map.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_file}")

    plt.close()

def print_ascii_map(grid, downsample=4):
    """Print ASCII representation of map."""
    h, w = grid.shape
    chars = {0: '·', 1: ' ', 2: '#'}

    print("\nASCII Map (· = unknown, space = free, # = obstacle):")
    print("-" * (w // downsample + 2))

    for y in range(h - 1, -1, -downsample):
        row = "|"
        for x in range(0, w, downsample):
            # Sample this region
            region = grid[max(0, x):min(w, x+downsample),
                         max(0, y-downsample+1):min(h, y+1)]
            if np.any(region == 2):
                row += '#'
            elif np.any(region == 1):
                row += ' '
            else:
                row += '·'
        row += "|"
        print(row)

    print("-" * (w // downsample + 2))

if __name__ == "__main__":
    map_file = sys.argv[1] if len(sys.argv) > 1 else '/home/bo/robot_pet/exploration_map.npy'
    view_map(map_file)
