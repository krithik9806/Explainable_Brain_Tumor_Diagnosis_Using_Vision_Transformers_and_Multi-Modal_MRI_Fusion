"""
Skull Stripping Module for Raw 3D NIfTI Brain MRI Scans.

This module provides reusable functionality to remove non-brain tissue (skull, scalp, dura)
from raw 3D NIfTI (.nii / .nii.gz) MRI volumes using HD-BET as the primary state-of-the-art
deep-learning-based skull stripper, with FSL BET as a fallback option.

Note on Pipeline Usage:
    BraTS dataset volumes are distributed pre-skull-stripped by the dataset providers.
    Therefore, this preprocessing step is skipped in the main BraTS processing pipeline.
    This script is provided for reusability when ingesting raw, unstripped MRI scans in future datasets.

Dependencies (Optional):
    - HD-BET: https://github.com/MIC-DKFZ/HD-BET (`pip install hd-bet`)
    - FSL BET: FMRIB Software Library (requires FSL installed on Linux/macOS or WSL)
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


def skull_strip_nifti(
    input_path: str,
    output_path: str,
    method: str = "hd-bet",
    device: str = "0",
    overwrite: bool = True,
    save_mask: bool = False,
) -> str:
    """
    Perform skull stripping on a 3D raw NIfTI MRI volume using HD-BET or FSL BET.

    Args:
        input_path (str): Path to the raw input NIfTI file (.nii or .nii.gz).
        output_path (str): Destination path for the skull-stripped NIfTI file.
        method (str): Skull-stripping tool to use ('hd-bet' or 'fsl-bet'). Default is 'hd-bet'.
        device (str): GPU device index for HD-BET (e.g., '0' or 'cpu'). Default is '0'.
        overwrite (bool): If True, overwrite existing output files. Default is True.
        save_mask (bool): If True, saves the brain extraction binary mask file.

    Returns:
        str: Absolute path to the output skull-stripped file.

    Raises:
        FileNotFoundError: If `input_path` does not exist.
        RuntimeError: If neither HD-BET nor FSL BET is available or if execution fails.
    """
    input_p = Path(input_path).resolve()
    output_p = Path(output_path).resolve()

    if not input_p.exists():
        raise FileNotFoundError(f"Input file not found: {input_p}")

    if output_p.exists() and not overwrite:
        print(f"[INFO] Output file already exists, skipping: {output_p}")
        return str(output_p)

    output_p.parent.mkdir(parents=True, exist_ok=True)

    method = method.lower()
    if method == "hd-bet":
        success = _run_hd_bet(input_p, output_p, device=device, save_mask=save_mask)
        if not success:
            print("[WARNING] HD-BET execution failed or tool unavailable. Attempting FSL BET fallback...")
            success = _run_fsl_bet(input_p, output_p, save_mask=save_mask)
    elif method == "fsl-bet":
        success = _run_fsl_bet(input_p, output_p, save_mask=save_mask)
    else:
        raise ValueError(f"Unsupported method '{method}'. Choose 'hd-bet' or 'fsl-bet'.")

    if not success:
        raise RuntimeError(
            "Skull stripping failed. Please ensure HD-BET (pip install hd-bet) or FSL BET "
            "is properly installed and available in system PATH."
        )

    print(f"[SUCCESS] Skull-stripped volume saved to: {output_p}")
    return str(output_p)


def _run_hd_bet(input_p: Path, output_p: Path, device: str = "0", save_mask: bool = False) -> bool:
    """Helper to invoke HD-BET command-line interface or python package."""
    hd_bet_cmd = shutil.which("hd-bet")
    
    if hd_bet_cmd is not None:
        cmd = [hd_bet_cmd, "-i", str(input_p), "-o", str(output_p), "-device", device]
        if not save_mask:
            cmd.extend(["-s", "0"])
        try:
            print(f"[INFO] Executing HD-BET CLI: {' '.join(cmd)}")
            res = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return True
        except (subprocess.CalledProcessError, Exception) as e:
            print(f"[ERROR] HD-BET CLI error: {e}")
            return False

    # Attempt Python package execution if CLI executable is not on PATH directly
    try:
        from HD_BET.run import run_hd_bet
        print(f"[INFO] Executing HD-BET via Python API...")
        run_hd_bet(
            input_files=str(input_p),
            output_files=str(output_p),
            mode="fast",
            device=device,
            postprocessing=True,
            do_overwrite=True,
        )
        return True
    except ImportError:
        print("[INFO] HD-BET Python package is not installed.")
        return False
    except Exception as e:
        print(f"[ERROR] HD-BET Python execution failed: {e}")
        return False


def _run_fsl_bet(input_p: Path, output_p: Path, save_mask: bool = False) -> bool:
    """Helper to invoke FSL BET command-line interface."""
    bet_cmd = shutil.which("bet")
    if bet_cmd is None:
        print("[INFO] FSL 'bet' command line utility not found in PATH.")
        return False

    cmd = [bet_cmd, str(input_p), str(output_p)]
    if save_mask:
        cmd.append("-m")

    try:
        print(f"[INFO] Executing FSL BET CLI: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return True
    except (subprocess.CalledProcessError, Exception) as e:
        print(f"[ERROR] FSL BET execution failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Skull stripping utility for 3D raw NIfTI MRI volumes using HD-BET / FSL BET.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_path",
        "-i",
        required=True,
        type=str,
        help="Path to raw input 3D NIfTI file (.nii or .nii.gz).",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        required=True,
        type=str,
        help="Path for output skull-stripped NIfTI file.",
    )
    parser.add_argument(
        "--method",
        choices=["hd-bet", "fsl-bet"],
        default="hd-bet",
        help="Skull-stripping algorithm to use.",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="GPU device index or 'cpu' for HD-BET.",
    )
    parser.add_argument(
        "--save_mask",
        action="store_true",
        help="Save the calculated brain mask alongside output.",
    )

    args = parser.parse_args()

    try:
        skull_strip_nifti(
            input_path=args.input_path,
            output_path=args.output_path,
            method=args.method,
            device=args.device,
            save_mask=args.save_mask,
        )
    except Exception as err:
        print(f"[FATAL] Skull stripping process failed: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
