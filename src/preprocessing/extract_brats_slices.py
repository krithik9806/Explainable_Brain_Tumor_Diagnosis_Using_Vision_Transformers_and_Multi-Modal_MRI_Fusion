import os
import sys
import glob
import random
import time
import pandas as pd
import numpy as np
import nibabel as nib

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROJECT_DIR = r"c:\PROJECTS\Explainable_Brain_Tumor_Diagnosis_Using_Vision_Transformers_and_Multi-Modal_MRI_Fusion"
RAW_BRATS_DIR = os.path.join(PROJECT_DIR, "data", "raw", "brats", "BraTS2020_TrainingData", "MICCAI_BraTS2020_TrainingData")
NAME_MAPPING_CSV = os.path.join(RAW_BRATS_DIR, "name_mapping.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "data", "processed", "brats_slices")

# Config parameters
TARGET_PATIENT_COUNT = 140  # 120-150 patient cases
SLICES_PER_PATIENT = 28     # 25-30 middle axial slices

def normalize_intensity(slice_img):
    """Min-max intensity normalization to [0, 1] range per slice."""
    min_val = slice_img.min()
    max_val = slice_img.max()
    if max_val > min_val:
        return ((slice_img - min_val) / (max_val - min_val)).astype(np.float32)
    return np.zeros_like(slice_img, dtype=np.float32)

def main():
    print("=== BraTS 2D Slice Extraction Script ===")
    print(f"Random Seed: {SEED}")
    print(f"Target Patient Subsample Count: {TARGET_PATIENT_COUNT}")
    print(f"Slices per Patient: {SLICES_PER_PATIENT}")
    print(f"Source Raw Directory: {RAW_BRATS_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("-" * 60)

    if not os.path.exists(NAME_MAPPING_CSV):
        print(f"Error: {NAME_MAPPING_CSV} not found!")
        sys.exit(1)

    df_mapping = pd.read_csv(NAME_MAPPING_CSV)
    print(f"Total available patients in name_mapping.csv: {len(df_mapping)}")
    print("Grade Distribution:\n", df_mapping['Grade'].value_counts())

    # Stratified Sampling across HGG and LGG
    # HGG ratio: ~79.4%, LGG ratio: ~20.6% -> 111 HGG, 29 LGG
    hgg_df = df_mapping[df_mapping['Grade'] == 'HGG']
    lgg_df = df_mapping[df_mapping['Grade'] == 'LGG']

    sample_hgg = hgg_df.sample(n=111, random_state=SEED)
    sample_lgg = lgg_df.sample(n=29, random_state=SEED)

    selected_df = pd.concat([sample_hgg, sample_lgg]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print(f"\nStratified Selected Subsample: {len(selected_df)} patients ({len(sample_hgg)} HGG, {len(sample_lgg)} LGG)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    manifest_records = []
    total_slices_saved = 0
    start_time = time.time()

    for idx, row in selected_df.iterrows():
        patient_id = row['BraTS_2020_subject_ID']
        grade = row['Grade']
        patient_dir = os.path.join(RAW_BRATS_DIR, patient_id)

        if not os.path.exists(patient_dir):
            print(f"Warning: Directory for {patient_id} not found at {patient_dir}, skipping.")
            continue

        # File paths for 4 modalities + seg mask
        flair_path = os.path.join(patient_dir, f"{patient_id}_flair.nii")
        t1_path = os.path.join(patient_dir, f"{patient_id}_t1.nii")
        t1ce_path = os.path.join(patient_dir, f"{patient_id}_t1ce.nii")
        t2_path = os.path.join(patient_dir, f"{patient_id}_t2.nii")

        # Segmentation mask file name handling (including BraTS20_Training_355 anomaly)
        seg_path = os.path.join(patient_dir, f"{patient_id}_seg.nii")
        if not os.path.exists(seg_path):
            alt_seg = glob.glob(os.path.join(patient_dir, "*[S|s]eg*.nii"))
            if alt_seg:
                seg_path = alt_seg[0]
            else:
                print(f"Warning: Seg mask for {patient_id} not found, skipping.")
                continue

        try:
            flair_vol = nib.load(flair_path).get_fdata().astype(np.float32)
            t1_vol = nib.load(t1_path).get_fdata().astype(np.float32)
            t1ce_vol = nib.load(t1ce_path).get_fdata().astype(np.float32)
            t2_vol = nib.load(t2_path).get_fdata().astype(np.float32)
            seg_vol = nib.load(seg_path).get_fdata().astype(np.uint8)
        except Exception as e:
            print(f"Error loading volumes for {patient_id}: {e}")
            continue

        # Find axial z-slices containing tumor voxels
        tumor_z_indices = np.where(seg_vol.sum(axis=(0, 1)) > 0)[0]

        if len(tumor_z_indices) > 0:
            center_z = int(np.median(tumor_z_indices))
        else:
            center_z = seg_vol.shape[2] // 2

        half_window = SLICES_PER_PATIENT // 2
        start_z = max(0, center_z - half_window)
        end_z = min(seg_vol.shape[2], start_z + SLICES_PER_PATIENT)

        # Adjust start_z if near boundary
        if (end_z - start_z) < SLICES_PER_PATIENT:
            start_z = max(0, end_z - SLICES_PER_PATIENT)

        patient_out_dir = os.path.join(OUTPUT_DIR, patient_id)
        os.makedirs(patient_out_dir, exist_ok=True)

        for z in range(start_z, end_z):
            flair_s = normalize_intensity(flair_vol[:, :, z])
            t1_s = normalize_intensity(t1_vol[:, :, z])
            t1ce_s = normalize_intensity(t1ce_vol[:, :, z])
            t2_s = normalize_intensity(t2_vol[:, :, z])
            mask_s = seg_vol[:, :, z]

            has_tumor = int(mask_s.sum() > 0)

            slice_filename = f"slice_{z:03d}.npz"
            slice_filepath = os.path.join(patient_out_dir, slice_filename)

            np.savez_compressed(
                slice_filepath,
                flair=flair_s,
                t1=t1_s,
                t1ce=t1ce_s,
                t2=t2_s,
                mask=mask_s
            )

            manifest_records.append({
                'patient_id': patient_id,
                'grade': grade,
                'z_index': z,
                'has_tumor': has_tumor,
                'file_path': os.path.relpath(slice_filepath, PROJECT_DIR)
            })
            total_slices_saved += 1

        if (idx + 1) % 20 == 0 or (idx + 1) == len(selected_df):
            elapsed = time.time() - start_time
            print(f"Processed {idx + 1}/{len(selected_df)} patients | Slices saved: {total_slices_saved} | Elapsed: {elapsed:.1f}s")

    # Save manifest CSV
    manifest_df = pd.DataFrame(manifest_records)
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)
    print(f"\nManifest saved to: {manifest_path}")

    # Calculate output size
    total_size_bytes = 0
    total_file_count = 0
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for f in files:
            fp = os.path.join(root, f)
            total_size_bytes += os.path.getsize(fp)
            total_file_count += 1

    size_mb = total_size_bytes / (1024**2)
    size_gb = total_size_bytes / (1024**3)

    print("\n" + "="*60)
    print("=== EXTRACTION COMPLETE ===")
    print(f"Processed Patients: {len(selected_df)}")
    print(f"Total Extracted 2D Slice Files: {total_slices_saved:,}")
    print(f"Total Files in data/processed/brats_slices/: {total_file_count:,}")
    print(f"Total Output Size: {size_mb:.2f} MB ({size_gb:.4f} GB)")
    print("="*60)

if __name__ == "__main__":
    main()
