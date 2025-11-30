import cv2
import numpy as np

# ---------------------------
# Import your data here
# ---------------------------
from face_model_landmarks_triangles import uv as UV_COORDS
from face_model_landmarks_triangles import triangles as TRIANGLES

# ---------------------------
# Settings
# ---------------------------
SIZE = 1024           # Size of output texture (SIZE x SIZE)
DOT_SIZE = 2          # Landmark dot size
LINE_THICKNESS = 1    # Triangle line thickness
DRAW_POINTS = True
FILL_TRIANGLES = False  # True = solid mesh visualization

# ---------------------------
# Convert UV (0-1) to pixel coords
# ---------------------------
pts = np.array(UV_COORDS, dtype=np.float32)
pts[:, 0] *= SIZE   # u → x
pts[:, 1] *= SIZE   # v → y

# Flip vertically for visual consistency (optional)
# pts[:, 1] = SIZE - pts[:, 1]

# ---------------------------
# Create blank texture
# ---------------------------
texture = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)

# ---------------------------
# Draw triangles
# ---------------------------
for tri in TRIANGLES:
    p1 = tuple(pts[tri[0]].astype(int))
    p2 = tuple(pts[tri[1]].astype(int))
    p3 = tuple(pts[tri[2]].astype(int))

    if FILL_TRIANGLES:
        cv2.fillConvexPoly(texture, np.array([p1, p2, p3]), (100, 200, 255))
    else:
        cv2.polylines(texture, [np.array([p1, p2, p3])], True, (255, 255, 255), LINE_THICKNESS)

# ---------------------------
# Draw points
# ---------------------------
if DRAW_POINTS:
    for p in pts:
        cv2.circle(texture, tuple(p.astype(int)), DOT_SIZE, (0, 255, 0), -1)

# ---------------------------
# Save and show
# ---------------------------
cv2.imwrite("uv_template.png", texture)
print("Saved uv_template.png")

cv2.imshow("UV Template", texture)
cv2.waitKey(0)
