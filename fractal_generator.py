"""
Advanced Fractal Generator with Force Field

Author: OBC

Description:
This script generates complex fractal patterns using recursive functions,
incorporating a force field that influences the growth direction of branches.
It includes randomness, attractor points, and color mapping.
Supports both 2D and 3D visualizations.

Requirements:
- Python 3.x
- NumPy
- Matplotlib
- Shapely
- mpl_toolkits.mplot3d (for 3D visualization)
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
from shapely.affinity import rotate, translate
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

# Global list to store all line segments
line_list = []

def generate_fractal(start_point, direction, length, depth, max_depth, params):
    """
    Recursive function to generate fractal patterns.

    Parameters:
    - start_point: NumPy array [x, y, z], starting coordinate.
    - direction: NumPy array [dx, dy, dz], direction vector.
    - length: Float, length of the current line segment.
    - depth: Int, current recursion depth.
    - max_depth: Int, maximum recursion depth.
    - params: Dict, contains parameters like angle_change, length_scaling_factor, force_field.
    """
    if depth > max_depth or length < 0.1:
        return

    # Normalize the direction vector
    direction_norm = np.linalg.norm(direction)
    if direction_norm == 0:
        return
    direction_unit = direction / direction_norm

    # Calculate the end point
    end_point = start_point + direction_unit * length

    # Create a line segment (projected onto 2D for Shapely)
    line = LineString([start_point[:2], end_point[:2]])
    line_list.append((line, depth))

    # Update length and depth for recursion
    new_length = length * params['length_scaling_factor']
    next_depth = depth + 1

    # Introduce randomness in angle and length scaling
    angle_variation = random.uniform(-params['angle_randomness'], params['angle_randomness'])
    length_variation = random.uniform(-params['length_randomness'], params['length_randomness'])

    angle_change = params['angle_change'] + angle_variation
    length_scaling_factor = params['length_scaling_factor'] + length_variation

    # Calculate new directions for branches
    num_branches = params['num_branches']
    angle_between_branches = params['max_branch_angle'] * 2 / max(1, num_branches - 1)

    for i in range(num_branches):
        # Calculate branch angle
        branch_angle = -params['max_branch_angle'] + i * angle_between_branches

        # Apply angle change and randomness
        total_angle = branch_angle + angle_change

        # Convert angle to radians
        rad = math.radians(total_angle)

        # Create rotation matrices for 3D rotation around a random axis
        axis = np.array([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)])
        axis = axis / np.linalg.norm(axis)
        cos_theta = math.cos(rad)
        sin_theta = math.sin(rad)
        one_minus_cos = 1 - cos_theta

        x, y, z = axis
        rotation_matrix = np.array([
            [cos_theta + x*x*one_minus_cos,     x*y*one_minus_cos - z*sin_theta, x*z*one_minus_cos + y*sin_theta],
            [y*x*one_minus_cos + z*sin_theta,   cos_theta + y*y*one_minus_cos,   y*z*one_minus_cos - x*sin_theta],
            [z*x*one_minus_cos - y*sin_theta,   z*y*one_minus_cos + x*sin_theta, cos_theta + z*z*one_minus_cos   ]
        ])

        new_direction = rotation_matrix @ direction_unit

        # Apply force field
        if params['use_force_field']:
            force = params['force_field_function'](end_point)
            new_direction = new_direction + force * params['force_field_strength']
            new_direction = new_direction / np.linalg.norm(new_direction)

        # Recursive call
        generate_fractal(
            end_point,
            new_direction,
            new_length,
            next_depth,
            max_depth,
            params
        )

def force_field_function(position):
    """
    Defines the force field affecting the growth direction.

    Parameters:
    - position: NumPy array [x, y, z], current position.

    Returns:
    - force: NumPy array [fx, fy, fz], force vector at the given position.
    """
    # Example: Radial force field towards the origin
    # force = -position / np.linalg.norm(position) ** 2

    # Example: Sinusoidal force field creating wave patterns
    k = 0.1  # Wave number
    force = np.array([
        math.sin(k * position[1]),
        math.sin(k * position[0]),
        0.0
    ])

    return force

def main():
    # Parameters
    start_point = np.array([0.0, 0.0, 0.0])
    initial_direction = np.array([0.0, 1.0, 0.0])  # Facing upwards
    initial_length = 10.0
    max_recursion_depth = 9

    params = {
        'angle_change': 0.0,                # Base angle change
        'angle_randomness': 20.0,           # Max angle variation in degrees
        'length_scaling_factor': 0.75,
        'length_randomness': 0.05,          # Max length scaling variation
        'num_branches': 3,                  # Number of branches at each node
        'max_branch_angle': 60.0,           # Maximum angle from the central direction
        'use_force_field': True,
        'force_field_function': force_field_function,
        'force_field_strength': 0.25,        # Strength of the force field
    }

    # Clear the line list
    line_list.clear()

    # Generate the fractal
    generate_fractal(
        start_point,
        initial_direction,
        initial_length,
        depth=0,
        max_depth=max_recursion_depth,
        params=params
    )

    # Visualization
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Normalize depths for coloring
    depths = [depth for (_, depth) in line_list]
    max_depth = max(depths) if depths else 1

    for line, depth in line_list:
        x, y = line.xy
        z = np.zeros_like(x)  # Assuming z=0 for 2D lines

        # Assign colors based on depth
        color = plt.cm.viridis(depth / max_depth)

        # 3D plotting
        ax.plot(x, y, z, color=color, linewidth=1)

    # Customize the plot
    ax.set_aspect('auto')
    ax.set_axis_off()
    ax.view_init(elev=60, azim=45)  # Adjust viewing angle
    plt.tight_layout()
    plt.show()

    # Save the figure
    fig.savefig('images/example.png', dpi=600, bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":
    main()
