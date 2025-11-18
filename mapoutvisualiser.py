# Navmesh Visualizer
# This script reads a binary map file (.vpa), parses its navigation mesh data,
# and visualizes it in a 3D plot using matplotlib.
#
# How to run:
# 1. Make sure you have Python installed.
# 2. Install the required library: pip install matplotlib numpy
# 3. Run the script from your terminal: python your_script_name.py
# 4. A file dialog will open, asking you to select your .vpa file.

import struct
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import os

# --- Data Structures based on the Hexpat ---

class Vertex:
    """Represents a 3D vertex."""
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Vertex(x={self.x}, y={self.y}, z={self.z})"

class Tri:
    """Represents a navigation mesh triangle."""
    def __init__(self, v_indices, n_indices, data):
        self.vertex_indices = v_indices
        self.neighbour_indices = n_indices
        self.data = data
        # Decode the bitfield data
        self.passable = data & 0b01111111 # Lower 7 bits
        self.battle = (data >> 7) & 0b11
        self.location = (data >> 11) & 0b11
        self.soundType = (data >> 15) & 0b11


    def __repr__(self):
        return f"Tri(v_indices={self.vertex_indices}, passable={self.passable})"

# --- Color Mapping for Visualization ---

# Maps the 'Passability' enum to a color for plotting.
# You can customize these colors.
PASSABILITY_COLORS = {
    0: 'green',      # Pass
    1: 'red',        # Block
    2: 'orange',     # BlockNPC
    14: 'yellow',    # BlockPlayer
    # Add a range for the script values 48-63
    **{i: 'purple' for i in range(48, 64)} # Script
}
DEFAULT_COLOR = 'gray' # For unmapped passability values

def get_passability_color(pass_val):
    """Returns a color based on the passability value."""
    return PASSABILITY_COLORS.get(pass_val, DEFAULT_COLOR)

# --- File Parsing Logic ---

def parse_map_file(file_path):
    """
    Parses the binary map file to extract navmesh data.
    """
    try:
        with open(file_path, 'rb') as f:
            content = f.read()

            # --- 1. Read Header ---
            # struct Header based on the hexpat
            # magic (4s), padding (12x), u32, geometry_ptr (I), navmesh_ptr (I), ...
            header_format = '<4s 12x I I I 7I I'
            header_size = struct.calcsize(header_format)
            header_data = struct.unpack_from(header_format, content, 0)

            magic = header_data[0]
            if magic != b'MAP1':
                print(f"Error: Invalid magic number. Expected 'MAP1', got {magic}")
                return None, None
            
            # CORRECTED: The navmesh pointer is the 3rd integer value in the tuple (index 3),
            # not the 2nd. header_data[0] is the magic string.
            navmesh_base_addr = header_data[3]
            print(f"Found NavMesh base address at: 0x{navmesh_base_addr:X}")

            # --- 2. Read NavMesh Struct ---
            # padding (10x), vertex_count (H), scale (f), padding (8x), vertices_ptr (I), tri_info_ptr (I)
            navmesh_format = '<10x H f 8x I I'
            navmesh_data = struct.unpack_from(navmesh_format, content, navmesh_base_addr)
            
            vertex_count = navmesh_data[0]
            scale = navmesh_data[1]
            vertices_ptr = navmesh_data[2]
            tri_info_ptr = navmesh_data[3]

            print(f"Vertex Count: {vertex_count}, Scale: {scale}")
            print(f"Vertices pointer: 0x{vertices_ptr:X}, TriInfo pointer: 0x{tri_info_ptr:X}")

            # --- 3. Read Vertices ---
            vertices = []
            vertex_format = '<hhh2x' # s16 x, y, z; padding[2]
            vertex_size = struct.calcsize(vertex_format)
            
            # The pointer is relative to the navmesh base address (cba)
            vertices_abs_addr = navmesh_base_addr + vertices_ptr
            
            for i in range(vertex_count):
                offset = vertices_abs_addr + i * vertex_size
                v_data = struct.unpack_from(vertex_format, content, offset)
                # Apply the scale to vertex positions
                vertices.append(Vertex(v_data[0] * scale, v_data[1] * scale, v_data[2] * scale))
            
            print(f"Successfully parsed {len(vertices)} vertices.")

            # --- 4. Read TriInfo and Tris ---
            # padding (8x), tri_count (I), tris_ptr (I)
            tri_info_format = '<8x I I'
            tri_info_abs_addr = navmesh_base_addr + tri_info_ptr
            tri_info_data = struct.unpack_from(tri_info_format, content, tri_info_abs_addr)

            tri_count = tri_info_data[0]
            tris_ptr = tri_info_data[1]
            tris_abs_addr = navmesh_base_addr + tris_ptr
            
            print(f"Triangle Count: {tri_count}")

            tris = []
            # u16[3] vertex_indices, s16[3] neighbour_indices, u32 data
            tri_format = '<3H 3h I' 
            tri_size = struct.calcsize(tri_format)

            for i in range(tri_count):
                offset = tris_abs_addr + i * tri_size
                t_data = struct.unpack_from(tri_format, content, offset)
                vertex_indices = t_data[0:3]
                neighbour_indices = t_data[3:6]
                tri_data_bitfield = t_data[6]
                tris.append(Tri(vertex_indices, neighbour_indices, tri_data_bitfield))
            
            print(f"Successfully parsed {len(tris)} triangles.")
            
            return vertices, tris

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None, None
    except struct.error as e:
        print(f"Error parsing the file structure: {e}")
        print("The file might not match the expected format or is corrupted.")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None


# --- Visualization Logic ---

def visualize_navmesh(vertices, tris, filename):
    """
    Creates and displays a 3D plot of the navmesh.
    """
    if not vertices or not tris:
        print("No data to visualize.")
        return

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    polygons = []
    colors = []

    for tri in tris:
        # Get the vertices for this triangle
        points = [vertices[i] for i in tri.vertex_indices]
        
        # Check for invalid indices
        if any(i >= len(vertices) for i in tri.vertex_indices):
            print(f"Warning: Triangle {tri} has an out-of-bounds vertex index. Skipping.")
            continue
            
        # Create a polygon from the vertex coordinates
        polygon = [(p.x, p.y, p.z) for p in points]
        polygons.append(polygon)
        
        # Assign color based on passability
        colors.append(get_passability_color(tri.passable))

    # Create a collection of polygons and add to the plot
    poly_collection = Poly3DCollection(polygons, facecolors=colors, linewidths=1, edgecolors='k', alpha=0.75)
    ax.add_collection3d(poly_collection)

    # --- Auto-scaling the plot view ---
    all_x = [v.x for v in vertices]
    all_y = [v.y for v in vertices]
    all_z = [v.z for v in vertices]
    
    if not all_x: # Handle case with no vertices
        return

    max_range = np.array([max(all_x)-min(all_x), max(all_y)-min(all_y), max(all_z)-min(all_z)]).max() / 2.0

    mid_x = (max(all_x)+min(all_x)) * 0.5
    mid_y = (max(all_y)+min(all_y)) * 0.5
    mid_z = (max(all_z)+min(all_z)) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # --- Create a legend for colors ---
    legend_elements = [plt.Rectangle((0, 0), 1, 1, color=color, label=label) for label, color in 
                       [('Passable', 'green'), ('Blocked', 'red'), ('Block NPC', 'orange'), 
                        ('Block Player', 'yellow'), ('Scripted', 'purple'), ('Other', 'gray')]]
    ax.legend(handles=legend_elements, loc='upper right')

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title(f'Navmesh Visualization for: {os.path.basename(filename)}')

    plt.show()


# --- Main Execution ---

def main():
    """
    Main function to run the application.
    """
    # Use Tkinter to open a file dialog
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    file_path = filedialog.askopenfilename(
        title="Select a VPA file",
        filetypes=(("VPA files", "*.vpa"), ("All files", "*.*"))
    )

    if not file_path:
        print("No file selected. Exiting.")
        return

    print(f"Loading file: {file_path}")
    vertices, tris = parse_map_file(file_path)

    if vertices and tris:
        visualize_navmesh(vertices, tris, file_path)
    else:
        print("Failed to parse map file. Cannot visualize.")

if __name__ == '__main__':
    main()
