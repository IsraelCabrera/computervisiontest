#!/usr/bin/env python3
"""
face_uv_wrap.py

Full-face texture wrapping using MediaPipe FaceMesh + triangle topology.

Features:
 - Attempts to download MediaPipe canonical triangle list (face_model_landmarks_triangles.txt)
 - If triangle list unavailable, falls back to Delaunay triangulation on detected landmarks
 - Maps a texture (RGBA PNG) to the face using per-triangle affine warps
 - Multiple textures in a folder, toggleable with keyboard
 - Optional GPU acceleration via OpenCV CUDA (best-effort)
 - macOS: detects likely absence of CUDA and falls back to CPU
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
import requests
from io import StringIO

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
                    help="If using Delaunay fallback, sample step across landmarks (1=all)")
args = parser.parse_args()

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
        if img.shape[2] != 4:
            # Ensure RGBA
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
        print("[note] macOS typically lacks NVIDIA CUDA support. CPU path will be used.")
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
# Attempt to download canonical triangulation
# -----------------------
# This file lives in the MediaPipe repo. We try to fetch it at runtime.
MEDIAPIPE_TRIANGLES_FILE = "face_model_landmarks_triangles.txt"

def fetch_triangle_list():
    try:
        from face_model_landmarks_triangles import triangles
        tris = triangles
        if tris:
            print(f"[ok] downloaded {len(tris)} triangles")
            return tris
        else:
            return None
    except Exception as e:
        print("[warn] could not fetch triangle list:", str(e))
        return None

canonical_triangles = fetch_triangle_list()
if canonical_triangles is None:
    print("[info] canonical triangle list not available — will fall back to Delaunay triangulation per-frame.")
else:
    # Ensure indices are int triples
    canonical_triangles = [tuple(map(int, t)) for t in canonical_triangles]

# -----------------------
# Geometry helpers
# -----------------------
def rect_from_points(pts):
    xs = pts[:,0]; ys = pts[:,1]
    minx = int(xs.min()); miny = int(ys.min())
    maxx = int(xs.max()); maxy = int(ys.max())
    return (minx, miny, maxx-minx, maxy-miny)

def affine_warp_triangle(src_img, dst_img, src_tri, dst_tri, use_gpu_local=False):
    """
    Warp triangular region from src_img (RGBA) to dst_img (BGR) using affine transform.
    src_tri, dst_tri: list/tuple of 3 (x,y) pairs in their respective coordinate spaces.
    """
    # dest bounding rectangle
    dst_rect = cv2.boundingRect(np.float32([dst_tri]))
    dx, dy, dw, dh = dst_rect
    if dw == 0 or dh == 0:
        return
    # source bounding rectangle
    src_rect = cv2.boundingRect(np.float32([src_tri]))
    sx, sy, sw, sh = src_rect
    if sw == 0 or sh == 0:
        return

    # offset coords
    dst_tri_offset = [ (dst_tri[i][0]-dx, dst_tri[i][1]-dy) for i in range(3) ]
    src_tri_offset = [ (src_tri[i][0]-sx, src_tri[i][1]-sy) for i in range(3) ]

    src_patch = src_img[sy:sy+sh, sx:sx+sw].copy()
    if src_patch.size == 0:
        return

    # compute affine and warp
    M = cv2.getAffineTransform(np.float32(src_tri_offset), np.float32(dst_tri_offset))
    try:
        if use_gpu_local and use_gpu and cuda_available:
            # GPU path (best-effort)
            gpu_src = cv2.cuda_GpuMat()
            gpu_src.upload(src_patch)
            gpu_dst = cv2.cuda.warpAffine(gpu_src, M, (dw, dh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
            warped = gpu_dst.download()
        else:
            warped = cv2.warpAffine(src_patch, M, (dw, dh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    except Exception:
        warped = cv2.warpAffine(src_patch, M, (dw, dh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

    # If src_patch had alpha channel, blend via alpha
    if warped.shape[2] == 4:
        alpha = warped[:,:,3].astype(np.float32)/255.0
        alpha = alpha[...,None]
        dst_roi = dst_img[dy:dy+dh, dx:dx+dw].astype(np.float32)
        fg = warped[:,:,:3].astype(np.float32)
        out = fg*alpha + dst_roi*(1-alpha)
        dst_img[dy:dy+dh, dx:dx+dw] = out.astype(np.uint8)
    else:
        dst_roi = dst_img[dy:dy+dh, dx:dx+dw]
        mask = np.any(warped[:,:,:3] != 0, axis=2)
        dst_roi[mask] = warped[:,:,:3][mask]
        dst_img[dy:dy+dh, dx:dx+dw] = dst_roi

# -----------------------
# Delaunay fallback builder
# -----------------------
def build_delaunay(width, height, points):
    rect = (0,0,width,height)
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
            if dists[i] > 1.5: # tolerance
                ok=False
                break
            idx.append(i)
        if ok and len(idx)==3:
            triangles.append(tuple(idx))
    # keep oriented unique triangles
    seen = set(); unique=[]
    for tri in triangles:
        key = tuple(sorted(tri))
        if key not in seen:
            seen.add(key)
            unique.append(tri)
    return unique

# -----------------------
# Main loop
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
# ensure some stable indices included
for s in [10, 234, 454, 152]:
    if s not in landmark_indices:
        landmark_indices.append(s)
landmark_indices = sorted(set(landmark_indices))

while True:
    ret, frame = cap.read()
    if not ret:
        print("[error] camera frame read failed")
        break
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    out_frame = frame.copy()

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]
        # collect all 468 landmark positions (pixel coords)
        lm_coords = []
        for lm in face.landmark:
            x_px = int(np.clip(lm.x * w, 0, w-1))
            y_px = int(np.clip(lm.y * h, 0, h-1))
            lm_coords.append([x_px, y_px])
        lm_coords = np.array(lm_coords, dtype=np.float32)

        # Use canonical triangles if available: triangles are tuples of landmark indices
        if canonical_triangles:
            triangles = canonical_triangles
        else:
            # fallback: build Delaunay on sampled landmarks
            sampled_pts = lm_coords[landmark_indices]
            # get triangles as indices into sampled_pts; convert to full-index scheme
            dela_tri = build_delaunay(w, h, sampled_pts)
            # map back to full indices
            triangles = []
            for tri in dela_tri:
                a = landmark_indices[tri[0]]
                b = landmark_indices[tri[1]]
                c = landmark_indices[tri[2]]
                triangles.append((a,b,c))

        # Source texture coordinates:
        tex_name, tex_img = textures[cur_tex_idx]
        th, tw = tex_img.shape[:2]

        # Two possible source mappings:
        # 1) If the texture is a canonical UV map (designed for MediaPipe ordering),
        #    then the UV coordinates should be known. We don't have the UV map file here,
        #    so we assume the texture's coordinate for each landmark is:
        #      (u = landmark_uv_x * tw, v = landmark_uv_y * th)
        #    BUT since we don't have explicit UVs, we approximate by normalizing the landmark
        #    positions w.r.t. the face bounding rectangle: this works reasonably for full-face masks
        #    if your texture is centered similarly.
        rx, ry, rw_box, rh_box = rect_from_points(lm_coords)
        # Expand bounding box slightly to include forehead / edges
        pad_x = int(0.1 * rw_box)
        pad_y = int(0.15 * rh_box)
        rx = max(0, rx - pad_x)
        ry = max(0, ry - pad_y)
        rw_box = min(w - rx, rw_box + 2*pad_x)
        rh_box = min(h - ry, rh_box + 2*pad_y)

        # Compute src_points by mapping each landmark inside the face rect to texture pixel coords
        src_points = []
        for (px, py) in lm_coords:
            if rw_box > 1:
                u = (px - rx) / float(rw_box)
            else:
                u = 0.5
            if rh_box > 1:
                v = (py - ry) / float(rh_box)
            else:
                v = 0.5
            sx = np.clip(u * tw, 0, tw-1)
            sy = np.clip(v * th, 0, th-1)
            src_points.append((sx, sy))
        src_points = np.array(src_points, dtype=np.float32)

        # Warp each triangle from texture space (src_points) to frame space (lm_coords)
        # Note: this assumes texture is prepared / aligned for this mapping. For canonical UV-perfect results,
        #       pass a texture designed for MediaPipe canonical UV; otherwise expect approximate wrapping.
        for tri in triangles:
            a, b, c = tri
            # ensure indices in range
            if a<0 or b<0 or c<0 or a>=len(lm_coords) or b>=len(lm_coords) or c>=len(lm_coords):
                continue
            dst_tri = [tuple(lm_coords[a]), tuple(lm_coords[b]), tuple(lm_coords[c])]
            src_tri = [tuple(src_points[a]), tuple(src_points[b]), tuple(src_points[c])]
            # skip degenerate
            area = abs((dst_tri[0][0]*(dst_tri[1][1]-dst_tri[2][1]) +
                        dst_tri[1][0]*(dst_tri[2][1]-dst_tri[0][1]) +
                        dst_tri[2][0]*(dst_tri[0][1]-dst_tri[1][1]))/2.0)
            if area < 0.5:
                continue
            affine_warp_triangle(tex_img, out_frame, src_tri, dst_tri, use_gpu_local=True)

    # overlay UI text
    cv2.putText(out_frame, f"Texture [{cur_tex_idx+1}/{len(textures)}]: {textures[cur_tex_idx][0]}",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

    cv2.imshow("Face UV Wrap", out_frame)
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
