#!/usr/bin/env python
import sys
import os
import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)

# Try to import mediapipe; fail with clear message if not installed
try:
    import mediapipe as mp
    from mediapipe.python.solutions import face_mesh as mp_face_mesh_module
    mp_face_mesh = mp.solutions.face_mesh
except Exception as e:
    logging.debug("Error importing MediaPipe. Make sure 'mediapipe' is installed: pip install mediapipe")
    raise

# Optional: scipy for better Delaunay (fallback)
try:
    from scipy.spatial import Delaunay
    _has_scipy = True
except Exception:
    _has_scipy = False

# Path to the UV texture template provided earlier
DEFAULT_TEXTURE_PATH = "face_uv_template.png"

# Put the path to your texture here (square PNG/JPG)
TEXTURE_PATH = DEFAULT_TEXTURE_PATH

# Drawing & processing parameters
SHOW_DEBUG = True  # show intermediate landmarks/triangles
SAVE_OUTPUT_DIR = "face_texture_outputs"
os.makedirs(SAVE_OUTPUT_DIR, exist_ok=True)

# Try to retrieve UV map and triangles from mediapipe if available
uv_map = None          # list of (u, v) pairs normalized [0,1]
triangles = None       # list of triplets of vertex indices (i0, i1, i2)

# Attempt 1: try canonical names (may exist in some mediapipe versions)
try:
    # Some builds expose FACEMESH_UV_MAP and a triangulation list
    from mediapipe.python.solutions.face_mesh import FACEMESH_TESSELATION  # noqa
    uv_map = FACEMESH_TESSELATION
    logging.debug("Loaded FACEMESH_UV_MAP from mediapipe.")
except Exception:
    try:
        # load uv_map from json uv_map.json
        import json
        with open("uv_map.json", "r") as f:
            uv_map_json = json.load(f)
        # convert uv_map_json to list of (a,b,c) tuples
        uv_map = [(uv[0], uv[1], uv[2]) for uv in uv_map_json]
        logging.debug("Loaded uv_map from uv_map.json.")
    except Exception:
        uv_map = None

# Attempt 2: try the tessellation (may be edges); use when it's triangles
try:
    from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION  # noqa
    tess = FACEMESH_TESSELATION
    # Check whether tess contains triplets (triangles) or pairs (edges)
    if tess and len(tess) > 0 and isinstance(tess[0], (tuple, list)):
        if len(tess[0]) == 3:
            triangles = tess
            logging.debug("Using FACEMESH_TESSELATION as triangle list from mediapipe.")
        else:
            # it's likely edges; we'll not use it directly here
            logging.debug("FACEMESH_TESSELATION contains pairs/edges; will build triangles later if needed.")
except Exception:
    tess = None

# If uv_map still None, we will attempt to read it from a bundled file.
# NOTE: For robust operation you can paste the 468-entry UV array into this script.
# For this example we will *not* include the full hard-coded uv map; instead we rely
# on the mediapipe-provided map. If your Mediapipe does not have FACEMESH_UV_MAP,
# you should provide it manually by editing this file and setting `uv_map = [...]`.
if uv_map is None:
    logging.debug("FACEMESH_UV_MAP not available in this mediapipe build. The script will attempt a fallback.")
    # Fallback: We will infer UVs using the normalized landmark (x,y) coordinates per frame.
    # This is a pragmatic fallback for prototyping but *not* the same as canonical UVs.
    # For best results, provide the true uv_map exported from MediaPipe.
    use_uv_fallback = True
else:
    use_uv_fallback = False

def landmarks_to_points(landmarks, image_shape):
    """Convert mediapipe normalized landmarks to pixel coordinates (x,y)."""
    h, w = image_shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    return pts

def uv_to_texture_coords(uv, tex_w, tex_h):
    """Convert normalized UV (u,v) in [0,1] to pixel coords on texture."""
    return [(int(u * tex_w), int(v * tex_h)) for (u, v) in uv]

def get_triangle_list_from_uvs(uvs):
    """Build a triangle list using Delaunay on the UV coordinate set.
    Returns a list of (i0, i1, i2) indices into the uv array.
    Requires scipy.spatial.Delaunay if available; otherwise uses OpenCV subdiv2D.
    """
    coords = np.array(uvs)
    if coords.shape[0] < 3:
        return []

    # Use scipy Delaunay if available for robust triangulation
    if _has_scipy:
        try:
            # Delaunay expects 2D points
            tri = Delaunay(coords)
            triangles_idx = [tuple(tr) for tr in tri.simplices]
            logging.debug(f"Built Delaunay triangulation (scipy). Triangles: {len(triangles_idx)}")
            return triangles_idx
        except Exception as e:
            logging.debug("Scipy Delaunay failed:", e)

    # Fallback to OpenCV Subdiv2D (pixel coordinates expected)
    try:
        # Convert normalized to a bounding pixel plane
        pts = (coords * 1000).astype(np.int32)  # scale to avoid degenerate points
        rect = (0, 0, 1000, 1000)
        subdiv = cv2.Subdiv2D(rect)
        for p in pts:
            subdiv.insert((int(p[0]), int(p[1])))
        edge_list = subdiv.getTriangleList().reshape(-1, 6)
        triangles_idx = []
        # Map triangle vertex coordinates back to nearest indices
        for t in edge_list:
            tri_pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
            idxs = []
            for tp in tri_pts:
                # find nearest original point
                dists = np.sum((pts - np.array(tp))**2, axis=1)
                idxs.append(int(np.argmin(dists)))
            triangles_idx.append(tuple(idxs))
        logging.debug(f"Built triangulation using OpenCV Subdiv2D. Triangles: {len(triangles_idx)}")
        return triangles_idx
    except Exception as e:
        logging.debug("Falling back: triangulation failed:", e)
        return []

def warp_texture_to_face(image, texture, landmarks, uv_map_for_frame=None, triangle_list=None):
    """
    Core function: warp texture onto face region defined by landmarks.
    - landmarks: list of mediapipe landmarks (each has .x,.y)
    - uv_map_for_frame: list of normalized (u,v) pairs matching the texture layout.
                        If None and use_uv_fallback True, we use normalized landmark (x,y) as UVs.
    - triangle_list: list of triplets of indices into landmarks/uv_map.
    """
    h, w = image.shape[:2]
    tex_h, tex_w = texture.shape[:2]

    # Convert landmarks to pixel points on the image
    pts_img = [(lm.x * w, lm.y * h) for lm in landmarks]
    pts_img = np.array(pts_img, dtype=np.float32)

    # UV coords either from provided uv_map_for_frame or fallback to landmark normalized x,y
    if uv_map_for_frame is None:
        # fallback: use the normalized landmark xy as UVs (not canonical)
        uv = [(lm.x, lm.y) for lm in landmarks]
    else:
        uv = uv_map_for_frame

    # Convert to texture pixel coordinates
    pts_tex = np.array([(u * tex_w, v * tex_h) for (u, v) in uv], dtype=np.float32)

    # Acquire triangle list: try provided, otherwise build from UVs
    if triangle_list is None or len(triangle_list) == 0:
        # Build triangulation on UVs (so triangles correspond to texture topology)
        triangle_list = get_triangle_list_from_uvs(uv)

    out_img = image.copy()

    # For each triangle, compute affine mapping from texture patch -> image triangle
    for tri in triangle_list:
        try:
            i0, i1, i2 = tri
            # source triangle in texture
            src_tri = np.float32([pts_tex[i0], pts_tex[i1], pts_tex[i2]])
            # destination triangle in image
            dst_tri = np.float32([pts_img[i0], pts_img[i1], pts_img[i2]])

            # Compute bounding rects
            r_src = cv2.boundingRect(src_tri)
            r_dst = cv2.boundingRect(dst_tri)

            # Skip degenerate rects
            if r_src[2] <= 0 or r_src[3] <= 0 or r_dst[2] <= 0 or r_dst[3] <= 0:
                continue

            # Offset points by top-left corner of the respective rectangles
            src_tri_offset = []
            dst_tri_offset = []
            for j in range(3):
                src_tri_offset.append(((src_tri[j][0] - r_src[0]), (src_tri[j][1] - r_src[1])))
                dst_tri_offset.append(((dst_tri[j][0] - r_dst[0]), (dst_tri[j][1] - r_dst[1])))

            # Crop source patch from texture
            src_patch = texture[r_src[1]:r_src[1]+r_src[3], r_src[0]:r_src[0]+r_src[2]]

            if src_patch.shape[0] == 0 or src_patch.shape[1] == 0:
                continue

            # Compute affine transform
            warp_mat = cv2.getAffineTransform(np.float32(src_tri_offset), np.float32(dst_tri_offset))

            # Warp source patch to destination size
            warped_patch = cv2.warpAffine(src_patch, warp_mat, (r_dst[2], r_dst[3]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

            # Create mask by filling triangle
            mask = np.zeros((r_dst[3], r_dst[2], 3), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.int32(dst_tri_offset), (255, 255, 255), cv2.LINE_AA)

            # Paste warped patch into output image region using mask
            roi = out_img[r_dst[1]:r_dst[1]+r_dst[3], r_dst[0]:r_dst[0]+r_dst[2]]
            # Ensure shapes match
            if roi.shape[0] != warped_patch.shape[0] or roi.shape[1] != warped_patch.shape[1]:
                # skip if mismatch
                continue
            roi[:] = roi * (1 - mask/255) + warped_patch * (mask/255)
        except Exception as e:
            # Keep going even if one triangle fails
            # if SHOW_DEBUG:
            #     logging.debug("Triangle warp failed for tri", tri, "error:", e)
            continue

    return out_img

def main():
    # Load texture
    if not os.path.exists(TEXTURE_PATH):
        logging.debug("Texture not found at:", TEXTURE_PATH)
        logging.debug("Please place your UV texture at TEXTURE_PATH and rerun.")
        return
    texture = cv2.imread(TEXTURE_PATH, cv2.IMREAD_UNCHANGED)
    if texture is None:
        logging.debug("Failed to read texture. Make sure it's a valid image file.")
        return
    tex_h, tex_w = texture.shape[:2]
    logging.debug(f"Texture loaded: {TEXTURE_PATH} size: {tex_w} x {tex_h}")

    # Prepare mediapipe face mesh
    with mp_face_mesh.FaceMesh(static_image_mode=False,
                               max_num_faces=1,
                               refine_landmarks=True,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as face_mesh:

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logging.debug("Cannot open webcam. You can modify the script to process an image file instead.")
            return

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.debug("Failed to read frame from webcam.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            output = frame.copy()
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Optionally use canonical uv_map if available
                    uv_for_frame = None
                    if not use_uv_fallback and uv_map is not None:
                        uv_for_frame = uv_map  # assumes a list of 468 (u,v) pairs

                    # use triangles if we loaded them earlier
                    triangle_list = triangles

                    # Perform texture warping
                    output = warp_texture_to_face(output, texture, face_landmarks.landmark, uv_for_frame, triangle_list)

                    if SHOW_DEBUG:
                        # draw landmarks
                        mp.solutions.drawing_utils.draw_landmarks(
                            output,
                            face_landmarks,
                            mp_face_mesh.FACEMESH_TESSELATION
                        )

            cv2.imshow("Face Texture Mapper", output)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # save frame
                outpath = os.path.join(SAVE_OUTPUT_DIR, f"frame_{frame_idx:04d}.png")
                cv2.imwrite(outpath, output)
                logging.debug("Saved:", outpath)
            frame_idx += 1

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
