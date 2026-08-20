"""
Multi-Modal MRI Fusion Module for Brain Tumor Diagnosis.

This module implements input-level early fusion (channel-stacking) for multi-modal MRI scans
(BraTS 2020 dataset) as well as a pass-through handler for single-modality images (Kaggle dataset).

Rationale for Input-Level Early Fusion:
--------------------------------------
Early fusion concatenates co-registered MRI sequences (T1, T1ce, T2, FLAIR) along the channel dimension
prior to feature extraction, forming a single multi-channel tensor of shape [4, H, W] (e.g., [4, 224, 224]).

Key Architectural Advantages & Design Justification (per README.md):
1. Cross-Modal Joint Feature Learning:
   By presenting all four MRI modalities simultaneously at the patch embedding stage of the
   Swin Transformer backbone, early fusion allows joint spatial and cross-modality self-attention
   learning from the very first layer.
2. Synergistic Tissue Contrast Representation:
   Each MRI sequence captures complementary biological and anatomical characteristics of brain tissue:
   - T1 (Channel 0): Visualizes basic anatomical structures and tissue boundaries.
   - T1ce (Channel 1): Highlights vascular disruption, active tumor borders, and contrast enhancement.
   - T2 (Channel 2): Emphasizes hyperintense fluid accumulation and peritumoral edema regions.
   - FLAIR (Channel 3): Suppresses cerebrospinal fluid (CSF) signals to make parenchymal lesions prominent.
   Combining these 4 modalities into a fixed channel stack gives the Vision Transformer a complete,
   high-dimensional diagnostic context.
3. Computational & Parameter Efficiency:
   Early fusion requires only a single backbone model with a modified 4-channel input stem layer,
   avoiding the heavy parameter duplication and latency overhead of multi-stream late-fusion networks.
4. Spatial Co-Registration Compliance:
   Because BraTS dataset 3D volumes are rigid-body co-registered to a common anatomical coordinate frame,
   voxel-to-voxel spatial alignment across modalities is guaranteed, making pixel-level channel stacking
   structurally valid and optimal.

Channel Ordering Specification (strictly aligned with configs/brats_fusion_config.yaml):
--------------------------------------------------------------------------------------
Channel 0 (Index 0): T1    (Native T1-weighted MRI)
Channel 1 (Index 1): T1ce  (T1 post-contrast enhancement)
Channel 2 (Index 2): T2    (T2-weighted MRI)
Channel 3 (Index 3): FLAIR (Fluid Attenuated Inversion Recovery)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


class BraTSEarlyFusion(nn.Module):
    """
    Early Fusion module for BraTS 4-modality MRI scans.
    
    Stacks 4 co-registered 2D MRI modalities (T1, T1ce, T2, FLAIR) into a 4-channel tensor of shape [4, H, W].
    
    Expected Channel Ordering:
        Channel 0: T1
        Channel 1: T1ce
        Channel 2: T2
        Channel 3: FLAIR
    """

    # Strictly defined channel order matching configs/brats_fusion_config.yaml
    CHANNEL_ORDER = ("t1", "t1ce", "t2", "flair")

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        return_tensor: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            target_size (Tuple[int, int]): Expected spatial resolution [H, W]. Default: (224, 224).
            return_tensor (bool): If True, returns PyTorch torch.Tensor. If False, returns NumPy array.
            dtype (torch.dtype): Output tensor data type (default: torch.float32).
        """
        super().__init__()
        self.target_size = target_size
        self.return_tensor = return_tensor
        self.dtype = dtype
        self.expected_channels = len(self.CHANNEL_ORDER)

    def forward(
        self,
        modalities: Optional[Union[Dict[str, Union[np.ndarray, torch.Tensor]], str, Path, Tuple, List]] = None,
        **kwargs,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Fuses 4 BraTS modalities into a single [4, H, W] tensor or numpy array.

        Args:
            modalities: Input data which can be:
                - Dictionary containing keys ('t1', 't1ce', 't2', 'flair') mapped to arrays/tensors.
                - Path (str or Path) to a compressed .npz file containing the modality keys.
                - Tuple or List of 4 modality arrays/tensors in exact order (t1, t1ce, t2, flair).
                - Individual keyword arguments (t1=..., t1ce=..., t2=..., flair=...).

        Returns:
            torch.Tensor or np.ndarray: Stacked tensor of shape [4, 224, 224] (or [4, H, W]).

        Raises:
            KeyError: If any required modality key is missing.
            ValueError: If input dimensions, shapes, or contents contain NaNs/Infs.
            FileNotFoundError: If a file path is provided that does not exist.
        """
        modalities_dict: Dict[str, Union[np.ndarray, torch.Tensor]] = {}

        # Case 1: Path to .npz file
        if isinstance(modalities, (str, Path)):
            file_path = Path(modalities)
            if not file_path.exists():
                raise FileNotFoundError(f"BraTS slice file not found at: {file_path}")
            npz_data = np.load(file_path)
            for m in self.CHANNEL_ORDER:
                if m in npz_data:
                    modalities_dict[m] = npz_data[m]
                else:
                    raise KeyError(f"Key '{m}' not found in .npz file: {file_path}")

        # Case 2: Dictionary input
        elif isinstance(modalities, dict):
            modalities_dict = modalities

        # Case 3: Tuple or List input of length 4
        elif isinstance(modalities, (tuple, list)):
            if len(modalities) != 4:
                raise ValueError(f"Expected 4 modalities in tuple/list input, got {len(modalities)}")
            modalities_dict = {
                name: modalities[idx] for idx, name in enumerate(self.CHANNEL_ORDER)
            }

        # Case 4: Keyword arguments
        elif kwargs:
            modalities_dict = kwargs
        else:
            raise ValueError(
                "No valid input provided. Pass a dict, file path, tuple of 4 modalities, or kwargs."
            )

        # Verify all required keys exist
        missing_keys = [m for m in self.CHANNEL_ORDER if m not in modalities_dict]
        if missing_keys:
            raise KeyError(
                f"Missing required modality keys for early fusion: {missing_keys}. "
                f"Required channel order: {self.CHANNEL_ORDER}"
            )

        # Extract and format each channel
        channels: List[np.ndarray] = []
        for mod_name in self.CHANNEL_ORDER:
            val = modalities_dict[mod_name]

            # Detach tensor if needed
            if isinstance(val, torch.Tensor):
                arr = val.detach().cpu().numpy()
            else:
                arr = np.asarray(val, dtype=np.float32)

            # Ensure 2D spatial shape [H, W]
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr.squeeze(0)
            elif arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr.squeeze(-1)

            if arr.ndim != 2:
                raise ValueError(
                    f"Modality '{mod_name}' must be a 2D slice [H, W], but got shape {arr.shape}"
                )

            # Verify spatial dimensions match target size
            if self.target_size is not None and arr.shape != self.target_size:
                raise ValueError(
                    f"Modality '{mod_name}' spatial shape {arr.shape} does not match target size {self.target_size}"
                )

            channels.append(arr.astype(np.float32))

        # Stack into [4, H, W] along axis 0
        fused_stack = np.stack(channels, axis=0)  # Shape: [4, H, W]

        # NaN / Inf validation
        if np.isnan(fused_stack).any():
            raise ValueError("Fused BraTS modality stack contains NaN values!")
        if np.isinf(fused_stack).any():
            raise ValueError("Fused BraTS modality stack contains Inf values!")

        # Format output
        if self.return_tensor:
            tensor_output = torch.from_numpy(fused_stack).to(dtype=self.dtype)
            return tensor_output

        return fused_stack


class KagglePassThrough(nn.Module):
    """
    Pass-through fusion module for Kaggle single-modality MRI images.
    
    Provides a uniform interface by accepting single-modality image data and formatting it
    into a 3-channel tensor of shape [3, H, W].
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        return_tensor: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            target_size (Tuple[int, int]): Expected spatial resolution [H, W]. Default: (224, 224).
            return_tensor (bool): If True, returns PyTorch torch.Tensor. If False, returns NumPy array.
            dtype (torch.dtype): Output tensor data type (default: torch.float32).
        """
        super().__init__()
        self.target_size = target_size
        self.return_tensor = return_tensor
        self.dtype = dtype
        self.expected_channels = 3

    def forward(
        self,
        image: Union[np.ndarray, torch.Tensor, str, Path, Dict[str, Union[np.ndarray, torch.Tensor]]],
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Formats single-modality Kaggle image data into a [3, H, W] tensor or array.

        Args:
            image: Image data as numpy array, PyTorch tensor, dict, or file path to .npz file.

        Returns:
            torch.Tensor or np.ndarray: Formatted 3-channel tensor/array of shape [3, 224, 224].
        """
        img_arr: np.ndarray

        if isinstance(image, (str, Path)):
            file_path = Path(image)
            if not file_path.exists():
                raise FileNotFoundError(f"Kaggle file not found at: {file_path}")
            data = np.load(file_path)
            key = 'image' if 'image' in data else list(data.keys())[0]
            img_arr = data[key]
        elif isinstance(image, dict):
            key = 'image' if 'image' in image else list(image.keys())[0]
            img_arr = image[key]
        elif isinstance(image, torch.Tensor):
            img_arr = image.detach().cpu().numpy()
        else:
            img_arr = np.asarray(image, dtype=np.float32)

        # Convert to float32
        img_arr = img_arr.astype(np.float32)

        # Reshape to [3, H, W]
        if img_arr.ndim == 2:
            # Squeezed 2D image -> replicate to 3 channels [3, H, W]
            img_arr = np.stack([img_arr] * 3, axis=0)
        elif img_arr.ndim == 3:
            if img_arr.shape[2] == 3:
                # [H, W, 3] -> [3, H, W]
                img_arr = np.transpose(img_arr, (2, 0, 1))
            elif img_arr.shape[0] == 3:
                # Already [3, H, W]
                pass
            elif img_arr.shape[2] == 1:
                img_arr = np.squeeze(img_arr, axis=2)
                img_arr = np.stack([img_arr] * 3, axis=0)
            elif img_arr.shape[0] == 1:
                img_arr = np.squeeze(img_arr, axis=0)
                img_arr = np.stack([img_arr] * 3, axis=0)
            else:
                raise ValueError(f"Unexpected image shape for Kaggle pass-through: {img_arr.shape}")
        else:
            raise ValueError(f"Unexpected dimensions for Kaggle image: {img_arr.ndim}")

        # Verify spatial dimensions match target size
        if self.target_size is not None and (img_arr.shape[1], img_arr.shape[2]) != self.target_size:
            raise ValueError(
                f"Kaggle image spatial shape {img_arr.shape[1:]} does not match target size {self.target_size}"
            )

        # NaN / Inf validation
        if np.isnan(img_arr).any():
            raise ValueError("Kaggle image tensor contains NaN values!")
        if np.isinf(img_arr).any():
            raise ValueError("Kaggle image tensor contains Inf values!")

        if self.return_tensor:
            return torch.from_numpy(img_arr).to(dtype=self.dtype)

        return img_arr


def fuse_brats_modalities(
    t1: Union[np.ndarray, torch.Tensor],
    t1ce: Union[np.ndarray, torch.Tensor],
    t2: Union[np.ndarray, torch.Tensor],
    flair: Union[np.ndarray, torch.Tensor],
    target_size: Tuple[int, int] = (224, 224),
    return_tensor: bool = True,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Functional early fusion helper for 4 BraTS modalities.

    Channel Order:
        [0]: T1
        [1]: T1ce
        [2]: T2
        [3]: FLAIR

    Returns:
        Tensor or array of shape [4, H, W].
    """
    fusion_module = BraTSEarlyFusion(target_size=target_size, return_tensor=return_tensor)
    return fusion_module(t1=t1, t1ce=t1ce, t2=t2, flair=flair)


def pass_through_kaggle(
    image: Union[np.ndarray, torch.Tensor, str, Path],
    target_size: Tuple[int, int] = (224, 224),
    return_tensor: bool = True,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Functional pass-through helper for Kaggle 3-channel image.

    Returns:
        Tensor or array of shape [3, H, W].
    """
    module = KagglePassThrough(target_size=target_size, return_tensor=return_tensor)
    return module(image)


def get_fusion_module(
    dataset_name: str = "brats2020_fusion",
    target_size: Tuple[int, int] = (224, 224),
    return_tensor: bool = True,
) -> nn.Module:
    """
    Factory function returning the appropriate fusion module according to dataset name.

    Args:
        dataset_name (str): Dataset tag (e.g. 'brats2020_fusion', 'kaggle').
        target_size (Tuple[int, int]): Spatial resolution (H, W).
        return_tensor (bool): If True, returns PyTorch torch.Tensor.

    Returns:
        nn.Module: Instance of BraTSEarlyFusion or KagglePassThrough.
    """
    tag = dataset_name.lower()
    if "brats" in tag:
        return BraTSEarlyFusion(target_size=target_size, return_tensor=return_tensor)
    elif "kaggle" in tag:
        return KagglePassThrough(target_size=target_size, return_tensor=return_tensor)
    else:
        raise ValueError(f"Unsupported dataset name '{dataset_name}'. Expected 'brats' or 'kaggle' variant.")
