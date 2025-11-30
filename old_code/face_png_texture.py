import cv2
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION

mp_face_mesh = mp.solutions.face_mesh

# ----------------------------------------------------------------------------
# Alpha blending function (supports RGBA textures)
# ----------------------------------------------------------------------------
def alpha_blend_rgba_over_bgr(fg_rgba, bg_bgr):
    """
    fg_rgba: warped RGBA texture (H x W x 4)
    bg_bgr:  target BGR frame (H x W x 3)
    """
    # Extract alpha channel, convert to float
    alpha = fg_rgba[:, :, 3] / 255.0
    alpha = alpha[:, :, np.newaxis]

    # Convert fg to BGR
    fg_bgr = fg_rgba[:, :, :3].astype(np.float32)
    bg_bgr = bg_bgr.astype(np.float32)

    # Blend
    blended = fg_bgr * alpha + bg_bgr * (1 - alpha)
    return blended.astype(np.uint8)

# ----------------------------------------------------------------------------
# Load texture (PNG with or without transparency)
# ----------------------------------------------------------------------------
texture = cv2.imread("face_uv_texture.png", cv2.IMREAD_UNCHANGED)

if texture is None:
    raise ValueError("Could not load texture file.")

has_alpha = (texture.shape[2] == 4)
print("Loaded texture with alpha:", has_alpha)

# ----------------------------------------------------------------------------
# Load UV coordinates
# ----------------------------------------------------------------------------
try:
    uv_coords = list(FACEMESH_TESSELATION) # np.load("face_uv_coords.npy")
    print(f"Loaded UV coordinates for {len(uv_coords)} landmarks")
except FileNotFoundError:
    print("face_uv_coords.npy not found. Please generate it first.")
    exit()

# ----------------------------------------------------------------------------
# Define triangle indices (MediaPipe FaceMesh tesselation)
# ----------------------------------------------------------------------------

# Filter only triangles (connections with 3 points)
triangles = []
for connection in np.load("face_uv_coords.npy"):
    if len(connection) == 3:  # Only take triangles
        triangles.append(connection)

print(f"Using {len(triangles)} triangles for texture mapping")

# ----------------------------------------------------------------------------
# Fixed warp_texture_to_face function
# ----------------------------------------------------------------------------
def warp_texture_to_face(frame, landmarks, texture, uv_coords, triangles):
    h, w = frame.shape[:2]
    tw, th = texture.shape[1], texture.shape[0]

    # Initialize warped texture with same channels as input texture
    warped_texture = np.zeros((h, w, texture.shape[2]), dtype=np.uint8)

    for tri_indices in triangles:
        idx0, idx1, idx2 = tri_indices

        # Get face triangle points - ensure proper shape (3, 2)
        pt1 = [landmarks[idx0].x * w, landmarks[idx0].y * h]
        pt2 = [landmarks[idx1].x * w, landmarks[idx1].y * h]
        pt3 = [landmarks[idx2].x * w, landmarks[idx2].y * h]
        face_triangle = np.array([pt1, pt2, pt3], dtype=np.float32)

        # Get UV triangle points (scaled to texture size) - ensure proper shape (3, 2)
        uv1 = [uv_coords[idx0][0] * (tw - 1), uv_coords[idx0][1] * (th - 1)]
        uv2 = [uv_coords[idx1][0] * (tw - 1), uv_coords[idx1][1] * (th - 1)]
        uv3 = [uv_coords[idx2][0] * (tw - 1), uv_coords[idx2][1] * (th - 1)]
        texture_triangle = np.array([uv1, uv2, uv3], dtype=np.float32)

        # Verify triangle areas to avoid degenerate triangles
        face_area = cv2.contourArea(face_triangle)
        texture_area = cv2.contourArea(texture_triangle)
        
        if face_area < 1.0 or texture_area < 1.0:
            continue  # Skip degenerate triangles

        try:
            # Compute affine transform
            M = cv2.getAffineTransform(texture_triangle, face_triangle)
            
            # Warp the texture triangle to face triangle
            warped_tri = cv2.warpAffine(
                texture,
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_TRANSPARENT
            )

            # Create mask for the triangle region
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, face_triangle.astype(np.int32), 255)

            # Blend the warped triangle into the final texture
            for c in range(texture.shape[2]):
                warped_texture[:, :, c] = np.where(
                    mask == 255,
                    warped_tri[:, :, c],
                    warped_texture[:, :, c]
                )

        except cv2.error as e:
            continue

    return warped_texture

# ----------------------------------------------------------------------------
# Alternative implementation using a different approach
# ----------------------------------------------------------------------------
def warp_texture_to_face_alternative(frame, landmarks, texture, uv_coords, triangles):
    """
    Alternative implementation that's more robust
    """
    h, w = frame.shape[:2]
    tw, th = texture.shape[1], texture.shape[0]

    # Create empty output
    warped_texture = np.zeros((h, w, texture.shape[2]), dtype=np.uint8)
    
    # Process triangles in batches for better performance
    valid_triangles = 0
    
    for tri_indices in triangles:
        idx0, idx1, idx2 = tri_indices

        # Source points (texture space)
        src_pts = np.array([
            [uv_coords[idx0][0] * tw, uv_coords[idx0][1] * th],
            [uv_coords[idx1][0] * tw, uv_coords[idx1][1] * th],
            [uv_coords[idx2][0] * tw, uv_coords[idx2][1] * th]
        ], dtype=np.float32)

        # Destination points (face space)
        dst_pts = np.array([
            [landmarks[idx0].x * w, landmarks[idx0].y * h],
            [landmarks[idx1].x * w, landmarks[idx1].y * h],
            [landmarks[idx2].x * w, landmarks[idx2].y * h]
        ], dtype=np.float32)

        # Skip if any point is outside reasonable bounds
        if (np.any(dst_pts < 0) or np.any(dst_pts[:, 0] > w) or 
            np.any(dst_pts[:, 1] > h) or np.any(src_pts < 0) or
            np.any(src_pts[:, 0] > tw) or np.any(src_pts[:, 1] > th)):
            continue

        # Calculate affine transform
        try:
            M = cv2.getAffineTransform(src_pts, dst_pts)
            
            # Warp the texture
            warped_tri = cv2.warpAffine(
                texture, M, (w, h), flags=cv2.INTER_LINEAR, 
                borderMode=cv2.BORDER_TRANSPARENT
            )
            
            # Create mask
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, dst_pts.astype(np.int32), 255)
            
            # Apply mask
            mask_bool = mask.astype(bool)
            for c in range(texture.shape[2]):
                warped_texture[mask_bool, c] = warped_tri[mask_bool, c]
            
            valid_triangles += 1
            
        except Exception as e:
            continue

    print(f"Processed {valid_triangles} valid triangles")
    return warped_texture

# ----------------------------------------------------------------------------
# Simple implementation for testing
# ----------------------------------------------------------------------------
def warp_texture_simple(frame, landmarks, texture, uv_coords, triangles):
    """
    Simple implementation that processes all valid triangles
    """
    h, w = frame.shape[:2]
    tw, th = texture.shape[1], texture.shape[0]
    
    warped_texture = np.zeros((h, w, texture.shape[2]), dtype=np.uint8)
    valid_count = 0
    
    for tri in triangles:
        if len(tri) != 3:
            continue
            
        idx0, idx1, idx2 = tri
        
        # Source points in texture space
        src_points = np.float32([
            [uv_coords[idx0][0] * tw, uv_coords[idx0][1] * th],
            [uv_coords[idx1][0] * tw, uv_coords[idx1][1] * th],
            [uv_coords[idx2][0] * tw, uv_coords[idx2][1] * th]
        ])
        
        # Destination points in screen space
        dst_points = np.float32([
            [landmarks[idx0].x * w, landmarks[idx0].y * h],
            [landmarks[idx1].x * w, landmarks[idx1].y * h],
            [landmarks[idx2].x * w, landmarks[idx2].y * h]
        ])
        
        try:
            # Get affine transform
            M = cv2.getAffineTransform(src_points, dst_points)
            
            # Warp texture
            warped_tri = cv2.warpAffine(
                texture, M, (w, h), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_TRANSPARENT
            )
            
            # Create mask
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, dst_points.astype(np.int32), 255)
            
            # Apply to output
            mask_3d = mask[:, :, np.newaxis] if texture.shape[2] == 3 else mask
            if texture.shape[2] == 4:
                mask_3d = np.repeat(mask[:, :, np.newaxis], 4, axis=2)
            else:
                mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                
            warped_texture = np.where(mask_3d > 0, warped_tri, warped_texture)
            valid_count += 1
            
        except Exception as e:
            continue
    
    print(f"Successfully processed {valid_count} triangles")
    return warped_texture

# ----------------------------------------------------------------------------
# Main capture loop
# ----------------------------------------------------------------------------
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark

            # Use the simple implementation
            warped = warp_texture_to_face(
                frame,
                face_landmarks,
                texture,
                uv_coords,
                triangles
            )

            # If texture has alpha → blend
            if has_alpha:
                frame = alpha_blend_rgba_over_bgr(warped, frame)
            else:
                # Create a mask where the texture is not black
                mask = np.any(warped > 0, axis=2)
                for c in range(3):  # Only for BGR channels
                    frame[:, :, c] = np.where(mask, warped[:, :, c], frame[:, :, c])

        cv2.imshow("Texture Mapped Face", frame)
        if cv2.waitKey(1) == 27:  # ESC key
            break

cap.release()
cv2.destroyAllWindows()