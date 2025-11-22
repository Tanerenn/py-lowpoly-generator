import numpy as np
import os
import sys
from scipy.spatial import Delaunay
from skimage import io, color, feature, transform
import matplotlib.pyplot as plt

def generate_low_poly_svg(image_path, output_svg="output.svg", num_points=1500, resize_width=800):
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        return

    img = io.imread(image_path)
    
    if img.shape[2] == 4:
        img = color.rgba2rgb(img)

    if img.shape[1] > resize_width:
        scale_factor = resize_width / img.shape[1]
        h = int(img.shape[0] * scale_factor)
        img = transform.resize(img, (h, resize_width), anti_aliasing=True)
        img = (img * 255).astype(np.uint8)

    gray = color.rgb2gray(img)
    h, w = gray.shape

    coords = feature.corner_peaks(feature.corner_harris(gray), min_distance=5, threshold_rel=0.02)
    
    if len(coords) > num_points:
        indices = np.random.choice(len(coords), num_points, replace=False)
        coords = coords[indices]
    
    random_points = np.random.rand(int(num_points / 4), 2) * [h, w]
    corners = np.array([[0, 0], [0, w], [h, 0], [h, w]])
    points = np.concatenate((coords, random_points, corners))

    tri = Delaunay(points)

    svg_content = f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">\n'
    
    for simplex in tri.simplices:
        pts = points[simplex]
        center = np.mean(pts, axis=0).astype(int)
        cx = np.clip(center[0], 0, h-1)
        cy = np.clip(center[1], 0, w-1)
        
        r, g, b = img[cx, cy]
        hex_color = '#{:02x}{:02x}{:02x}'.format(r, g, b)
        
        p_str = " ".join([f"{p[1]:.2f},{p[0]:.2f}" for p in pts])
        svg_content += f'  <polygon points="{p_str}" fill="{hex_color}" stroke="{hex_color}" stroke-width="1"/>\n'

    svg_content += '</svg>'
    
    with open(output_svg, "w", encoding="utf-8") as f:
        f.write(svg_content)
    
    print(f"Successfully generated: {output_svg}")

if __name__ == "__main__":
    input_image = "input.jpg" 
    generate_low_poly_svg(input_image, output_svg="result.svg", num_points=2000)
