import cv2
import numpy as np
import mediapipe as mp


# ---------------------------
# Load MediaPipe FaceMesh
# ---------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------------------------
# Load Texture (PNG with alpha)
# ---------------------------
texture = cv2.imread("face_uv_texture.png", cv2.IMREAD_UNCHANGED)
if texture is None:
    raise FileNotFoundError("Could not load texture.png")

tex_h, tex_w = texture.shape[:2]


# ---------------------------
# Utility: Alpha blend PNG onto frame
# ---------------------------
def blend_transparent(background, overlay):
    # overlay must have 4 channels (BGRA)
    bgr = overlay[:, :, :3]
    alpha = overlay[:, :, 3] / 255.0

    alpha = alpha[..., None]  # shape (h,w,1)

    return (bgr * alpha + background * (1 - alpha)).astype(np.uint8)


# ---------------------------
# Main Loop (Webcam)
# ---------------------------
cap = cv2.VideoCapture(0)

# Landmark IDs for stable facial coordinates:
#   10  = forehead (center)
#  234  = left temple area
#  454  = right temple area
#  152  = chin
LANDMARKS = [234, 454, 10, 152]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]

        # Extract required landmark coordinates
        pts = []
        for idx in LANDMARKS:
            lm = face.landmark[idx]
            pts.append([int(lm.x * w), int(lm.y * h)])
        pts = np.array(pts, dtype=np.float32)

        # Define where the texture should map:
        # Using the same number of control points as pts.
        # Adjust texture control points if you want to reshape the fit.
        tex_pts = np.array([
            [0, 0],
            [tex_w - 1, 0],
            [tex_w // 2, tex_h // 3],
            [tex_w // 2, tex_h - 1],
        ], dtype=np.float32)

        # Compute homography (projective transform)
        H, _ = cv2.findHomography(tex_pts, pts)

        # Warp the texture into the frame space
        warped = cv2.warpPerspective(texture, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_DEFAULT)

        # Split channels to detect alpha
        if warped.shape[2] == 4:
            # Separate frame BGR and overlay BGRA
            overlay = warped
            # Create a mask of where the PNG has alpha > 0
            alpha_mask = overlay[:, :, 3] > 0

            # Extract only where alpha is present
            frame_region = frame.copy()

            # Blend only on pixels where alpha exists
            roi = frame_region * 1  # copy
            blended = blend_transparent(roi, overlay)

            # Replace only alpha areas
            frame[alpha_mask] = blended[alpha_mask]

    cv2.imshow("Face Texture Overlay", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
