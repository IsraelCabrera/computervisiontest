#!/usr/bin/env python3
"""
Face texture overlay using MediaPipe FaceMesh + Delaunay triangulation.

Features:
 - Uses Mediapipe FaceMesh for 468 landmarks
 - Builds Delaunay triangulation on sampled landmarks each frame
 - Warps texture PNG (with alpha) onto each triangle with affine transforms
 - Multiple textures toggled with keyboard (n / p / 1..9)
 - Optional GPU acceleration via OpenCV CUDA (if available) with --use-gpu
 - Mac users: script detects CUDA absence and falls back to CPU
"""

import cv2
import numpy as np
import mediapipe as mp
import argparse
import os
import sys
import math
import platform
from glob import glob

# ------------
# Arguments
# ------------
parser = argparse.ArgumentParser(description="Face texture overlay with Delaunay triangulation")
parser.add_argument("--textures", "-t", required=True, help="Path to a texture PNG or a directory containing PNGs")
parser.add_argument("--camera", "-c", type=int, default=0, help="Camera index (default 0)")
parser.add_argument("--use-gpu", action="store_true", help="Attempt to use OpenCV CUDA (if available)")
parser.add_argument("--out", type=str, default=None, help="Optional output path to save video (mp4)")
parser.add_argument("--sample-step", type=int, default=2, help="Sample step for landmarks (1 = all 468, 2 = every 2nd landmark, etc.)")
args = parser.parse_args()

# ------------
# Texture loader
# ------------
def load_textures(path):
    files = []
    if os.path.isdir(path):
        files = sorted(glob(os.path.join(path, "*.png")))
    elif os.path.isfile(path):
        files = [path]
    else:
        raise FileNotFoundError(f"No such file or directory: {path}")

    if not files:
        raise FileNotFoundError("No PNG textures found in the provided path")

    texs = []
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: couldn't load {f}, skipping")
            continue
        if img.shape[2] != 4:
            print(f"Warning: texture {f} does not have alpha channel (RGBA). Converting with full alpha.")
            bgr = img
            alpha = np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.uint8)
            img = np.concatenate([bgr, alpha], axis=2)
        texs.append((os.path.basename(f), img))
    if not texs:
        raise FileNotFoundError("No valid RGBA PNG textures loaded")
    return texs

textures = load_textures(args.textures)
current_texture_index = 0

# ------------
# GPU availability check
# ------------
use_gpu = False
cuda_available = False
if args.use_gpu:
    if hasattr(cv2, "cuda"):
        try:
            cnt = cv2.cuda.getCudaEnabledDeviceCount()
            if cnt > 0:
                cuda_available = True
                use_gpu = True
            else:
                # cv2.cuda exists but no devices
                cuda_available = False
                use_gpu = False
        except Exception:
            cuda_available = False
            use_gpu = False
    else:
        cuda_available = False
        use_gpu = False

if args.use_gpu and not use_gpu:
    print("GPU requested but OpenCV CUDA isn't available or no CUDA devices found. Falling back to CPU.")
    if platform.system() == "Darwin":
        print("Note: On macOS, CUDA support is typically not available. Consider CPU mode or build OpenCV with Metal/OpenCL support.")
else:
    if use_gpu:
        print("Using OpenCV CUDA acceleration (best-effort).")
    else:
        print("Running in CPU mode.")

# ------------
# Mediapipe setup
# ------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ------------
# Helper utilities
# ------------
def alpha_blend(img_bg, img_fg):
    """
    Alpha-blend foreground RGBA (img_fg) over background BGR (img_bg).
    img_bg: HxWx3 BGR
    img_fg: HxWx4 BGRA
    Returns combined image (HxWx3 BGR)
    """
    if img_fg is None:
        return img_bg
    fg_bgr = img_fg[..., :3].astype(np.float32)
    alpha = img_fg[..., 3].astype(np.float32) / 255.0
    alpha = alpha[..., None]
    bg = img_bg.astype(np.float32)
    out = fg_bgr * alpha + bg * (1 - alpha)
    return out.astype(np.uint8)

def rect_from_points(pts):
    xs = pts[:, 0]
    ys = pts[:, 1]
    min_x = int(xs.min())
    min_y = int(ys.min())
    max_x = int(xs.max())
    max_y = int(ys.max())
    return (min_x, min_y, max_x - min_x, max_y - min_y)

def inside_rect(pt, rect):
    x, y = pt
    rx, ry, rw, rh = rect
    return (x >= rx - 1) and (x <= rx + rw + 1) and (y >= ry - 1) and (y <= ry + rh + 1)

# ------------
# Delaunay triangulation helpers
# ------------
def build_delaunay(width, height, points):
    """
    Build Delaunay triangulation for given points.
    Returns list of triangles as indices into `points`.
    """
    rect = (0, 0, width, height)
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert((float(p[0]), float(p[1])))

    triangleList = subdiv.getTriangleList()  # returns flat triangles as coordinates
    triangles = []
    # For each triangle, find indices of the vertices
    for t in triangleList:
        pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        idx = []
        good = True
        for vx, vy in pts:
            # find the nearest point in our list
            dists = np.linalg.norm(points - np.array([vx, vy]), axis=1)
            i = int(np.argmin(dists))
            if dists[i] > 1.0:  # numeric tolerance
                # If too far, mark not good (this can happen at borders)
                good = False
                break
            idx.append(i)
        if good and len(idx) == 3:
            triangles.append(tuple(idx))
    # Remove duplicates
    triangles = list(set(tuple(sorted(t)) for t in triangles))
    # But above sorting loses orientation. We need oriented triangles (3 indices).
    # Instead we will re-create oriented triangles by using the original triangleList mapping to indices:
    oriented = []
    for t in triangleList:
        pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        idx = []
        valid = True
        for vx, vy in pts:
            dists = np.linalg.norm(points - np.array([vx, vy]), axis=1)
            i = int(np.argmin(dists))
            if dists[i] > 1.0:
                valid = False
                break
            idx.append(i)
        if valid and len(idx) == 3:
            oriented.append(tuple(idx))
    # Remove approximate duplicates while preserving orientation
    unique = []
    seen = set()
    for tri in oriented:
        key = tuple(sorted(tri))
        if key not in seen:
            seen.add(key)
            unique.append(tri)
    return unique

def affine_warp_triangle(src_img, dst_img, src_tri, dst_tri, use_gpu_local=False):
    """
    Warp triangular region from src_img to dst_img using affine transform.
    src_tri and dst_tri are lists/arrays of 3 (x,y) points.
    This function warps and blends using alpha channel from src_img (if present).
    """
    # Bounding rect for destination triangle
    dst_rect = cv2.boundingRect(np.float32([dst_tri]))
    dst_x, dst_y, dst_w, dst_h = dst_rect
    if dst_w == 0 or dst_h == 0:
        return

    # Offset points to local coordinates
    dst_tri_offset = []
    for i in range(3):
        dst_tri_offset.append(((dst_tri[i][0] - dst_x), (dst_tri[i][1] - dst_y)))
    src_rect = cv2.boundingRect(np.float32([src_tri]))
    src_x, src_y, src_w, src_h = src_rect
    if src_w == 0 or src_h == 0:
        return
    src_tri_offset = []
    for i in range(3):
        src_tri_offset.append(((src_tri[i][0] - src_x), (src_tri[i][1] - src_y)))

    # Crop the source patch
    src_patch = src_img[src_y:src_y + src_h, src_x:src_x + src_w].copy()
    if src_patch.size == 0:
        return

    # Compute affine
    m = cv2.getAffineTransform(np.float32(src_tri_offset), np.float32(dst_tri_offset))

    # Warp the source patch to the size of the destination rect
    try:
        if use_gpu_local and use_gpu and cuda_available:
            # GPU path: upload patch, warp, download
            gpu_src = cv2.cuda_GpuMat()
            gpu_src.upload(src_patch)
            gpu_dst = cv2.cuda.warpAffine(gpu_src, m, (dst_w, dst_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
            warped = gpu_dst.download()
        else:
            warped = cv2.warpAffine(src_patch, m, (dst_w, dst_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    except Exception:
        # In case gpu calls fail for some reason - fallback to cpu
        warped = cv2.warpAffine(src_patch, m, (dst_w, dst_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

    # Build mask from the warped alpha (if present)
    if warped.shape[2] == 4:
        alpha = warped[:, :, 3] / 255.0
        alpha = alpha[..., None]
        # Destination ROI
        dst_roi = dst_img[dst_y:dst_y + dst_h, dst_x:dst_x + dst_w]
        if dst_roi.shape[0] != warped.shape[0] or dst_roi.shape[1] != warped.shape[1]:
            # size mismatch safety
            return
        # Blend channels
        for c in range(3):
            dst_roi[:, :, c] = (warped[:, :, c] * alpha[:, :, 0] + dst_roi[:, :, c] * (1 - alpha[:, :, 0])).astype(np.uint8)
        dst_img[dst_y:dst_y + dst_h, dst_x:dst_x + dst_w] = dst_roi
    else:
        # No alpha: simple copy of warped BGR
        dst_roi = dst_img[dst_y:dst_y + dst_h, dst_x:dst_x + dst_w]
        mask = np.any(warped[:, :, :3] != 0, axis=2)
        dst_roi[mask] = warped[:, :, :3][mask]
        dst_img[dst_y:dst_y + dst_h, dst_x:dst_x + dst_w] = dst_roi

# ------------
# Main loop
# ------------
cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    print("Error: cannot open camera", args.camera)
    sys.exit(1)

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or math.isnan(fps):
    fps = 30.0

writer = None
if args.out:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.out, fourcc, fps, (frame_w, frame_h))

print("Controls: n = next texture, p = previous, 1..9 = jump to texture, q / ESC = quit")

sample_step = max(1, int(args.sample_step))
# Choose landmark indices to sample. We'll sample across the 0..467 range with step
landmark_indices = list(range(0, 468, sample_step))

# Ensure we include some stable reference points (chin, left/right temple, forehead) if not in sample
stable_indices = [10, 234, 454, 152]
for s in stable_indices:
    if s not in landmark_indices:
        landmark_indices.append(s)
landmark_indices = sorted(set(landmark_indices))

while True:
    ok, frame = cap.read()
    if not ok:
        print("Failed to read frame from camera")
        break
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    out_frame = frame.copy()

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        pts = []
        for idx in landmark_indices:
            lm = face_landmarks.landmark[idx]
            pts.append([int(lm.x * w), int(lm.y * h)])
        pts = np.array(pts, dtype=np.int32)

        # Bounding rect of the face in frame space
        rx, ry, rw_rect, rh_rect = rect_from_points(pts)

        # Map face landmarks to texture source coordinates by normalizing within face bounding box
        tex_name, tex_img = textures[current_texture_index]
        tex_h, tex_w = tex_img.shape[:2]

        # Source points: mapping pts in face rect -> texture coordinates
        src_points = []
        for (px, py) in pts:
            # normalized inside face rect
            if rw_rect > 0:
                u = (px - rx) / float(rw_rect)
            else:
                u = 0.5
            if rh_rect > 0:
                v = (py - ry) / float(rh_rect)
            else:
                v = 0.5
            sx = np.clip(u * tex_w, 0, tex_w - 1)
            sy = np.clip(v * tex_h, 0, tex_h - 1)
            src_points.append([sx, sy])
        src_points = np.array(src_points, dtype=np.float32)

        # Destination points are the pts themselves (converted to float)
        dst_points = pts.astype(np.float32)

        # Build Delaunay on destination points (frame coordinates)
        triangles = build_delaunay(w, h, dst_points)

        # For better performance we can pre-crop a large ROI of dst frame and texture
        # But for correctness, we will warp triangle-by-triangle and blend into out_frame
        for tri in triangles:
            ia, ib, ic = tri
            src_tri = [tuple(src_points[ia]), tuple(src_points[ib]), tuple(src_points[ic])]
            dst_tri = [tuple(dst_points[ia]), tuple(dst_points[ib]), tuple(dst_points[ic])]

            # Skip degenerate triangles
            area = abs( (dst_tri[0][0]*(dst_tri[1][1]-dst_tri[2][1]) +
                         dst_tri[1][0]*(dst_tri[2][1]-dst_tri[0][1]) +
                         dst_tri[2][0]*(dst_tri[0][1]-dst_tri[1][1])) / 2.0)
            if area < 1.0:
                continue

            # Warp triangle
            affine_warp_triangle(tex_img, out_frame, src_tri, dst_tri, use_gpu_local=True)

    # Draw current texture name on screen
    cv2.putText(out_frame, f"Texture [{current_texture_index+1}/{len(textures)}]: {textures[current_texture_index][0]}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

    cv2.imshow("Face Texture Delaunay", out_frame)
    if writer is not None:
        writer.write(out_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break
    elif key == ord('n'):
        current_texture_index = (current_texture_index + 1) % len(textures)
    elif key == ord('p'):
        current_texture_index = (current_texture_index - 1) % len(textures)
    elif ord('1') <= key <= ord('9'):
        idx = (key - ord('1'))
        if idx < len(textures):
            current_texture_index = idx

cap.release()
if writer:
    writer.release()
cv2.destroyAllWindows()
