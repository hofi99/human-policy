"""
Dex5 to Dex3 Hand Keypoint Adapter

This module provides functions to convert 5-finger (Dex5) hand keypoints to 3-finger (Dex3) format
using weighted fingertip merging in task space.
"""

import numpy as np
from hdt.constants import RETARGETTING_INDICES


def adapt_dex5_to_dex3_keypoints(rel_hand_keypoints, w_index=0.7, w_middle=0.3, w_ring=0.5, w_pinky=0.5):
    """
    Convert Dex5 (5-finger) hand keypoints to Dex3 (3-finger) format by merging fingertips.
    
    Merging strategy:
    - Index 9 (index) + Index 14 (middle) → merged result at index 9
    - Index 19 (ring) + Index 24 (pinky) → merged result at index 14
    - Indices 19 and 24 are zeroed out
    
    Args:
        rel_hand_keypoints: np.array, shape (num_timesteps, 25, 3) or (25, 3)
            Hand keypoints in 25-keypoint skeleton format.
        w_index: float, default 0.7
            Weight for index finger when merging with middle finger
        w_middle: float, default 0.3
            Weight for middle finger when merging with index finger
        w_ring: float, default 0.5
            Weight for ring finger when merging with pinky finger
        w_pinky: float, default 0.5
            Weight for pinky finger when merging with ring finger
    
    Returns:
        np.array: Same shape as input, with merged keypoints.
            - Index 0: Palm center (unchanged)
            - Index 4: Thumb fingertip (unchanged)
            - Index 9: Merged index+middle fingertip (w_index*index9 + w_middle*index14)
            - Index 14: Merged ring+pinky fingertip (w_ring*index19 + w_pinky*index24)
            - Indices 19, 24: Zeroed out
    """
    # Normalize input to handle both (25, 3) and (num_timesteps, 25, 3) shapes
    original_shape = rel_hand_keypoints.shape
    is_single_timestep = len(original_shape) == 2
    
    if is_single_timestep:
        rel_hand_keypoints = rel_hand_keypoints[np.newaxis, :, :]
    
    num_timesteps = rel_hand_keypoints.shape[0]
    
    # Create a copy to avoid modifying input
    adapted_keypoints = rel_hand_keypoints.copy()
    
    # Keypoint indices
    INDEX_IDX = 9
    MIDDLE_IDX = 14
    RING_IDX = 19
    PINKY_IDX = 24
    
    # Merge index + middle → index 9
    # Original index 9 (index) and index 14 (middle) are merged
    adapted_keypoints[:, INDEX_IDX, :] = (
        w_index * rel_hand_keypoints[:, INDEX_IDX, :] +
        w_middle * rel_hand_keypoints[:, MIDDLE_IDX, :]
    )
    
    # Merge ring + pinky → index 14
    # Original index 19 (ring) and index 24 (pinky) are merged
    adapted_keypoints[:, MIDDLE_IDX, :] = (
        w_ring * rel_hand_keypoints[:, RING_IDX, :] +
        w_pinky * rel_hand_keypoints[:, PINKY_IDX, :]
    )
    
    # Zero out indices 19 and 24 (no longer needed for Dex3)
    adapted_keypoints[:, RING_IDX, :] = 0.0
    adapted_keypoints[:, PINKY_IDX, :] = 0.0
    
    # Restore original shape
    if is_single_timestep:
        adapted_keypoints = adapted_keypoints.squeeze(0)
    
    return adapted_keypoints
