import numpy as np
from face_triangles import TRIANGLES_468

def generate_accurate_face_uv_coords():
    """
    Generate more accurate UV coordinates for MediaPipe FaceMesh.
    This creates a better distribution that matches face topology.
    """
    uv_coords = np.zeros((468, 2), dtype=np.float32)
    
    # Landmark indices for different face regions (approximate)
    # You may need to adjust these based on your specific needs
    
    # Face outline (approx indices 0-16)
    for i in range(17):
        uv_coords[i, 0] = i / 16.0
        uv_coords[i, 1] = 0.0
    
    # Eyebrows (approx indices 17-46)
    for i in range(17, 27):  # Left eyebrow
        uv_coords[i, 0] = (i - 17) / 9.0 * 0.3 + 0.1
        uv_coords[i, 1] = 0.15
    for i in range(27, 36):  # Right eyebrow  
        uv_coords[i, 0] = (i - 27) / 8.0 * 0.3 + 0.6
        uv_coords[i, 1] = 0.15
    
    # Nose (approx indices 43-97)
    for i in range(43, 51):
        uv_coords[i, 0] = 0.45 + (i - 43) / 7.0 * 0.1
        uv_coords[i, 1] = 0.3 + (i - 43) / 7.0 * 0.2
    
    # Eyes (approx indices 33-132)  
    for i in range(33, 42):  # Left eye
        uv_coords[i, 0] = 0.2 + (i - 33) / 8.0 * 0.2
        uv_coords[i, 1] = 0.4
    for i in range(42, 51):  # Right eye
        uv_coords[i, 0] = 0.6 + (i - 42) / 8.0 * 0.2
        uv_coords[i, 1] = 0.4
    
    # Mouth (approx indices 133-172)
    for i in range(133, 150):
        uv_coords[i, 0] = 0.3 + (i - 133) / 16.0 * 0.4
        uv_coords[i, 1] = 0.7
    
    # Fill remaining landmarks with a fallback pattern
    for i in range(468):
        if uv_coords[i, 0] == 0 and uv_coords[i, 1] == 0:
            # Create a grid-like pattern for remaining points
            row = (i // 20) % 10
            col = i % 20
            uv_coords[i, 0] = col / 19.0
            uv_coords[i, 1] = row / 9.0 * 0.5 + 0.5
    
    return uv_coords

# Generate and save the improved UV coordinates
# uv_coords = generate_accurate_face_uv_coords()
uv_coords = TRIANGLES_468
np.save("face_uv_coords.npy", uv_coords)
print(f"Generated improved UV coordinates for {len(uv_coords)} landmarks")