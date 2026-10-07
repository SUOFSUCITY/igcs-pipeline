"""Geometric, shape and intensity features for location classification."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import parameter, validate_parameters

import SimpleITK as sitk
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import multiprocessing
from scipy import ndimage
import warnings
import json

warnings.filterwarnings("ignore")


def get_brain_mask(img_sitk):
    img_arr = sitk.GetArrayFromImage(img_sitk)
    mask = (img_arr > parameter("localization.brain_hu_min")) & (
        img_arr < parameter("localization.brain_hu_max")
    )
    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    sizes = ndimage.sum(mask, labeled, range(num_features + 1))
    max_label = np.argmax(sizes[1:]) + 1
    brain_mask = (labeled == max_label).astype(np.uint8)
    struct = ndimage.generate_binary_structure(3, 2)
    brain_mask = ndimage.binary_closing(
        brain_mask,
        structure=struct,
        iterations=parameter("localization.closing_iterations"),
    )
    brain_mask = ndimage.binary_fill_holes(brain_mask)
    return brain_mask.astype(np.uint8)


def compute_pca_features(mask_arr, spacing):
    points = np.argwhere(mask_arr > 0)
    if len(points) < 4:
        return [0] * 6
    coords = points.astype(float)
    coords[:, 0] *= spacing[2]
    coords[:, 1] *= spacing[1]
    coords[:, 2] *= spacing[0]
    mean_coords = np.mean(coords, axis=0)
    coords -= mean_coords
    cov = np.cov(coords.T)
    evals, evecs = np.linalg.eigh(cov)
    idx = evals.argsort()[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    eig1, eig2, eig3 = np.sqrt(evals + 1e-06) * 2
    elongation = eig2 / (eig1 + 1e-06)
    flatness = eig3 / (eig1 + 1e-06)
    v1 = evecs[:, 0]
    z_axis = np.array([1, 0, 0])
    cos_theta = np.abs(np.dot(v1, z_axis))
    angle_z = np.degrees(np.arccos(np.clip(cos_theta, 0, 1)))
    return [eig1, eig2, eig3, elongation, flatness, angle_z]


def process_case(item):
    img_path = item["img"]
    mask_path = item["msk"]
    case_id = item["id"]
    label = item["label"]
    subset = item["split"]
    try:
        img_sitk = sitk.ReadImage(str(img_path))
        mask_sitk = sitk.ReadImage(str(mask_path))
        spacing = img_sitk.GetSpacing()
        origin = img_sitk.GetOrigin()
        img_arr = sitk.GetArrayFromImage(img_sitk)
        mask_arr = sitk.GetArrayFromImage(mask_sitk)
        mask_arr = (mask_arr > 0).astype(np.uint8)
        if np.sum(mask_arr) == 0:
            print(f"Warning: Empty mask for {case_id}, returning default 0 features.")
            return {
                "id": case_id,
                "subset": subset,
                "label": label,
                "rel_x": 0.0,
                "rel_y": 0.0,
                "rel_z": 0.0,
                "center_dist": 0.0,
                "midline_dist": 0.0,
                "surface_dist": 0.0,
                "surface_dist_min": 0.0,
                "surface_dist_10p": 0.0,
                "volume": 0.0,
                "eig1": 0.0,
                "eig2": 0.0,
                "eig3": 0.0,
                "elongation": 0.0,
                "flatness": 0.0,
                "angle_z": 0.0,
                "hu_mean": 0.0,
                "hu_std": 0.0,
                "peri_hu_mean": 0.0,
            }
        brain_mask_arr = get_brain_mask(img_sitk)
        ich_indices = np.argwhere(mask_arr > 0)
        ich_centroid_idx = np.mean(ich_indices, axis=0)
        brain_indices = np.argwhere(brain_mask_arr > 0)
        if len(brain_indices) > 0:
            brain_centroid_idx = np.mean(brain_indices, axis=0)
            z_min, y_min, x_min = np.min(brain_indices, axis=0)
            z_max, y_max, x_max = np.max(brain_indices, axis=0)
            brain_size = np.array([z_max - z_min, y_max - y_min, x_max - x_min]) + 1e-06
        else:
            brain_centroid_idx = np.array(img_arr.shape) / 2
            brain_size = np.array(img_arr.shape)
            z_min, z_max = (0, img_arr.shape[0])
            y_min = x_min = 0
        spacing_zyx = np.array(spacing[::-1])
        rel_pos = (ich_centroid_idx - np.array([z_min, y_min, x_min])) / brain_size
        rel_z, rel_y, rel_x = rel_pos
        dist_vec = (ich_centroid_idx - brain_centroid_idx) * spacing_zyx
        center_dist = np.linalg.norm(dist_vec)
        midline_dist = np.abs(ich_centroid_idx[2] - brain_centroid_idx[2]) * spacing[0]
        if len(brain_indices) > 0:
            dist_map = ndimage.distance_transform_edt(
                brain_mask_arr, sampling=spacing_zyx
            )
            cz, cy, cx = ich_centroid_idx.astype(int)
            cz = np.clip(cz, 0, dist_map.shape[0] - 1)
            cy = np.clip(cy, 0, dist_map.shape[1] - 1)
            cx = np.clip(cx, 0, dist_map.shape[2] - 1)
            surface_dist = dist_map[cz, cy, cx]
            iz = ich_indices[:, 0]
            iy = ich_indices[:, 1]
            ix = ich_indices[:, 2]
            iz = np.clip(iz, 0, dist_map.shape[0] - 1)
            iy = np.clip(iy, 0, dist_map.shape[1] - 1)
            ix = np.clip(ix, 0, dist_map.shape[2] - 1)
            dist_values = dist_map[iz, iy, ix]
            if len(dist_values) > 0:
                surface_dist_min = float(np.min(dist_values))
                surface_dist_10p = float(np.percentile(dist_values, 10))
            else:
                surface_dist_min = float(surface_dist)
                surface_dist_10p = float(surface_dist)
        else:
            surface_dist = 0.0
            surface_dist_min = 0.0
            surface_dist_10p = 0.0
        eig1, eig2, eig3, elongation, flatness, angle_z = compute_pca_features(
            mask_arr, spacing
        )
        volume = np.sum(mask_arr) * np.prod(spacing)
        ich_pixels = img_arr[mask_arr > 0]
        hu_mean = np.mean(ich_pixels)
        hu_std = np.std(ich_pixels)
        iter_dilate = int(parameter("localization.ring_mm") / np.mean(spacing))
        struct = ndimage.generate_binary_structure(3, 1)
        dilated_mask = ndimage.binary_dilation(
            mask_arr, structure=struct, iterations=iter_dilate
        )
        ring_mask = dilated_mask ^ mask_arr
        ring_mask = ring_mask & brain_mask_arr
        if np.sum(ring_mask) > 0:
            peri_hu_mean = np.mean(img_arr[ring_mask > 0])
        else:
            peri_hu_mean = hu_mean
        feat = {
            "id": case_id,
            "subset": subset,
            "label": label,
            "rel_x": rel_x,
            "rel_y": rel_y,
            "rel_z": rel_z,
            "center_dist": center_dist,
            "midline_dist": midline_dist,
            "surface_dist": surface_dist,
            "surface_dist_min": surface_dist_min,
            "surface_dist_10p": surface_dist_10p,
            "volume": volume,
            "eig1": eig1,
            "eig2": eig2,
            "eig3": eig3,
            "elongation": elongation,
            "flatness": flatness,
            "angle_z": angle_z,
            "hu_mean": hu_mean,
            "hu_std": hu_std,
            "peri_hu_mean": peri_hu_mean,
        }
        return feat
    except Exception as e:
        print(f"Error processing {case_id}: {e}")
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Location feature extraction")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    validate_parameters(
        "localization.brain_hu_min",
        "localization.brain_hu_max",
        "localization.closing_iterations",
        "localization.ring_mm",
    )
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    json_path = args.tasks
    if not Path(json_path).exists():
        print(f"Error: {json_path} not found.")
        return
    with open(json_path, "r") as f:
        tasks = json.load(f)
    print(f"Loaded {len(tasks)} tasks from {json_path}")
    results = []
    num_p = max(1, multiprocessing.cpu_count() - 2)
    print(f"Starting extraction with {num_p} processes...")
    with multiprocessing.Pool(processes=num_p) as pool:
        for res in tqdm(pool.imap_unordered(process_case, tasks), total=len(tasks)):
            if res is not None:
                results.append(res)
    df = pd.DataFrame(results)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} features to {args.output}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
