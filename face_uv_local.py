#!/usr/bin/env python3
"""
face_uv_wrap_local.py

Applies per-triangle affine warp from texture UV to detected face landmarks so the texture
follows rotations and deformations correctly (provided the texture is authored in that UV layout).

Usage:
    python face_uv_wrap_local.py --textures ./textures --camera 0 [--use-gpu] [--out out.mp4]

Controls:
    n: next texture
    p: previous texture
    1..9: jump to texture
    q / ESC: quit
"""

import cv2
import numpy as np
import mediapipe as mp
import argparse
import os
import sys
import math
from glob import glob
import platform

# -----------------------
# Arguments
# -----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--textures", "-t", required=True,
                    help="Path to texture PNG or directory containing PNGs (RGBA)")
parser.add_argument("--camera", "-c", type=int, default=0, help="Camera index")
parser.add_argument("--use-gpu", action="store_true", help="Try to use OpenCV CUDA (if available)")
parser.add_argument("--out", help="Optional output mp4 path")
parser.add_argument("--sample-step", type=int, default=1,
                    help="Sampling step for Delaunay fallback (1 = all landmarks)")
args = parser.parse_args()

# -----------------------
# Load canonical UV & triangles from local files
# -----------------------
canonical_uv = None
canonical_triangles = None
try:
    from face_model_landmarks_triangles import uv as CANONICAL_UV
    from face_model_landmarks_triangles import triangles as CANONICAL_TRIANGLES
    canonical_uv = np.array(CANONICAL_UV, dtype=np.float32)  # shape (468,2)
    canonical_triangles = [tuple(t) for t in CANONICAL_TRIANGLES]
    print(f"[ok] Loaded canonical UV ({len(canonical_uv)}) and {len(canonical_triangles)} triangles.")
except Exception as e:
    canonical_uv = None
    canonical_triangles = None
    print("[warn] Could not import canonical_uv/canonical_triangles locally:", e)
    print("[warn] Will fall back to Delaunay triangulation if face detected (less ideal).")

# -----------------------
# Helpers: load textures
# -----------------------
def load_textures(path):
    files = []
    if os.path.isdir(path):
        files = sorted(glob(os.path.join(path, "*.png")))
    elif os.path.isfile(path):
        files = [path]
    else:
        raise FileNotFoundError(f"No such file or directory: {path}")
    texs = []
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[warn] couldn't read {f} -- skipping")
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        if img.shape[2] == 3:
            alpha = np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.uint8)
            img = np.concatenate([img, alpha], axis=2)
        texs.append((os.path.basename(f), img))
    if not texs:
        raise FileNotFoundError("No valid PNG textures found (RGBA).")
    return texs

textures = load_textures(args.textures)
cur_tex_idx = 0

# -----------------------
# GPU check
# -----------------------
use_gpu = False
cuda_available = False
if args.use_gpu:
    if hasattr(cv2, "cuda"):
        try:
            cnt = cv2.cuda.getCudaEnabledDeviceCount()
            cuda_available = cnt > 0
            use_gpu = cuda_available
        except Exception:
            cuda_available = False
            use_gpu = False
    else:
        cuda_available = False
        use_gpu = False

if args.use_gpu and not use_gpu:
    print("[info] GPU requested but OpenCV CUDA not available. Falling back to CPU.")
    if platform.system() == "Darwin":
        print("[note] macOS typically lacks CUDA support; CPU path will be used.")
else:
    if use_gpu:
        print("[info] Using OpenCV CUDA where possible.")

# -----------------------
# MediaPipe FaceMesh
# -----------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# -----------------------
# Geometry utilities
# -----------------------
def build_delaunay(width, height, points):
    """Returns oriented triangles as index tuples into points"""
    rect = (0, 0, width, height)
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert((float(p[0]), float(p[1])))
    triangleList = subdiv.getTriangleList()
    triangles = []
    pts = np.array(points)
    for t in triangleList:
        pts_coords = [(t[0],t[1]),(t[2],t[3]),(t[4],t[5])]
        idx = []
        ok = True
        for (vx,vy) in pts_coords:
            dists = np.linalg.norm(pts - np.array([vx,vy]), axis=1)
            i = int(np.argmin(dists))
            if dists[i] > 1.5:
                ok = False
                break
            idx.append(i)
        if ok and len(idx) == 3:
            triangles.append(tuple(idx))
    # remove duplicates (keep orientation from first occurrence)
    seen = set(); unique = []
    for tri in triangles:
        key = tuple(sorted(tri))
        if key not in seen:
            seen.add(key)
            unique.append(tri)
    return unique

def affine_warp_triangle(src_img, dst_img, src_tri, dst_tri, use_gpu_local=False):
    """
    Warp triangular region from src_img (RGBA) to dst_img (BGR) using affine transform.
    src_tri and dst_tri are lists of three (x,y) coordinate tuples in their respective spaces.
    """
    dst_rect = cv2.boundingRect(np.float32([dst_tri]))
    dx, dy, dw, dh = dst_rect
    if dw == 0 or dh == 0:
        return
    src_rect = cv2.boundingRect(np.float32([src_tri]))
    sx, sy, sw, sh = src_rect
    if sw == 0 or sh == 0:
        return

    # Offset points by rect top-left
    dst_tri_offset = [ (dst_tri[i][0] - dx, dst_tri[i][1] - dy) for i in range(3) ]
    src_tri_offset = [ (src_tri[i][0] - sx, src_tri[i][1] - sy) for i in range(3) ]

    src_patch = src_img[sy:sy+sh, sx:sx+sw].copy()
    if src_patch.size == 0:
        return

    M = cv2.getAffineTransform(np.float32(src_tri_offset), np.float32(dst_tri_offset))
    try:
        if use_gpu_local and use_gpu and cuda_available:
            gpu_src = cv2.cuda_GpuMat()
            gpu_src.upload(src_patch)
            gpu_dst = cv2.cuda.warpAffine(gpu_src, M, (dw, dh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
            warped = gpu_dst.download()
        else:
            warped = cv2.warpAffine(src_patch, M, (dw, dh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    except Exception:
        warped = cv2.warpAffine(src_patch, M, (dw, dh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

    # Blend using alpha if present
    if warped.shape[2] == 4:
        alpha = warped[:,:,3].astype(np.float32) / 255.0
        alpha = alpha[..., None]
        dst_roi = dst_img[dy:dy+dh, dx:dx+dw].astype(np.float32)
        fg = warped[:,:,:3].astype(np.float32)
        out = fg * alpha + dst_roi * (1 - alpha)
        dst_img[dy:dy+dh, dx:dx+dw] = out.astype(np.uint8)
    else:
        dst_roi = dst_img[dy:dy+dh, dx:dx+dw]
        mask = np.any(warped[:,:,:3] != 0, axis=2)
        dst_roi[mask] = warped[:,:,:3][mask]
        dst_img[dy:dy+dh, dx:dx+dw] = dst_roi

# -----------------------
# Camera + writer
# -----------------------
cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    print("[error] cannot open camera", args.camera)
    sys.exit(1)

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0 or math.isnan(fps):
    fps = 30.0

writer = None
if args.out:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.out, fourcc, fps, (frame_w, frame_h))

print("Controls: n=next, p=prev, 1..9=jump, q/ESC=quit")

sample_step = max(1, int(args.sample_step))
landmark_indices = list(range(0, 468, sample_step))
for s in [10, 234, 454, 152]:
    if s not in landmark_indices:
        landmark_indices.append(s)
landmark_indices = sorted(set(landmark_indices))

# -----------------------
# Main loop
# -----------------------
while True:
    ok, frame = cap.read()
    if not ok:
        print("[error] failed to read camera frame")
        break
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    out_frame = frame.copy()

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # Collect landmark pixel coordinates (2D)
        lm_coords = []
        for lm in face_landmarks.landmark:
            x_px = float(np.clip(lm.x * w, 0, w-1))
            y_px = float(np.clip(lm.y * h, 0, h-1))
            lm_coords.append((x_px, y_px))
        lm_coords = np.array(lm_coords, dtype=np.float32)  # shape (468,2)

        # Decide triangle set:
        if canonical_triangles is not None and canonical_uv is not None:
            triangles = canonical_triangles
        else:
            # fallback: make Delaunay on sampled landmarks (indices refer to sampled array)
            sampled_pts = lm_coords[landmark_indices]
            dela = build_delaunay(w, h, sampled_pts)
            # map indices back to full 468 indices
            triangles = []
            for tri in dela:
                triangles.append((landmark_indices[tri[0]], landmark_indices[tri[1]], landmark_indices[tri[2]]))

        # Source texture and its uv mapping
        tex_name, tex_img = textures[cur_tex_idx]
        th, tw = tex_img.shape[:2]

        # Source points per landmark: if we have canonical_uv, use those directly.
        if canonical_uv is not None:
            # canonical_uv are normalized UVs in range [0,1]
            src_points = np.stack([canonical_uv[:,0] * (tw - 1), canonical_uv[:,1] * (th - 1)], axis=1)
        else:
            # fallback: approximate mapping by normalizing landmark positions to face bbox
            # (less accurate but will still map)
            xs = lm_coords[:,0]; ys = lm_coords[:,1]
            minx, miny = xs.min(), ys.min()
            maxx, maxy = xs.max(), ys.max()
            rw_box = max(1.0, maxx - minx)
            rh_box = max(1.0, maxy - miny)
            src_points = np.zeros_like(lm_coords)
            for i,(px,py) in enumerate(lm_coords):
                u = (px - minx) / rw_box
                v = (py - miny) / rh_box
                sx = np.clip(u * (tw - 1), 0, tw - 1)
                sy = np.clip(v * (th - 1), 0, th - 1)
                src_points[i] = (sx, sy)

        # Warp all triangles
        for tri in triangles:
            a,b,c = tri
            # validate indices
            if a < 0 or b < 0 or c < 0 or a >= len(lm_coords) or b >= len(lm_coords) or c >= len(lm_coords):
                continue
            dst_tri = [tuple(lm_coords[a]), tuple(lm_coords[b]), tuple(lm_coords[c])]
            src_tri = [tuple(src_points[a]), tuple(src_points[b]), tuple(src_points[c])]

            # skip degenerate
            area = abs((dst_tri[0][0] * (dst_tri[1][1] - dst_tri[2][1]) +
                        dst_tri[1][0] * (dst_tri[2][1] - dst_tri[0][1]) +
                        dst_tri[2][0] * (dst_tri[0][1] - dst_tri[1][1])) / 2.0)
            if area < 0.5:
                continue

            affine_warp_triangle(tex_img, out_frame, src_tri, dst_tri, use_gpu_local=True)

    # UI text
    cv2.putText(out_frame, f"Texture [{cur_tex_idx+1}/{len(textures)}]: {textures[cur_tex_idx][0]}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

    cv2.imshow("Face UV Wrap (local)", out_frame)
    if writer:
        writer.write(out_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break
    elif key == ord('n'):
        cur_tex_idx = (cur_tex_idx + 1) % len(textures)
    elif key == ord('p'):
        cur_tex_idx = (cur_tex_idx - 1) % len(textures)
    elif ord('1') <= key <= ord('9'):
        idx = key - ord('1')
        if idx < len(textures):
            cur_tex_idx = idx

cap.release()
if writer:
    writer.release()
cv2.destroyAllWindows()
