"""dtw.py — High-performance 2D Dynamic Time Warping (DTW) with Sakoe-Chiba band."""
from __future__ import annotations
import numpy as np

def dtw_distance_2d(seq1: np.ndarray, seq2: np.ndarray, w: int = 10) -> float:
    """Compute the 2D Dynamic Time Warping distance between two arrays of shape (N, D)
    using the Sakoe-Chiba band constraint to restrict the warping path.

    Parameters
    ----------
    seq1: np.ndarray of shape (N, D)
        First feature sequence (e.g. query window).
    seq2: np.ndarray of shape (M, D)
        Second feature sequence (e.g. historical candidate).
    w: int, default=10
        Warping window constraint (max shift in time axis).
    """
    n, d = seq1.shape
    m = seq2.shape[0]
    
    # Warping window constraint must be at least the difference in sequence lengths
    w = max(w, abs(n - m))
    
    # Initialize DP table with infinity
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    
    for i in range(1, n + 1):
        # Limit search to the Sakoe-Chiba band [i - w, i + w]
        start_j = max(1, i - w)
        end_j = min(m + 1, i + w + 1)
        for j in range(start_j, end_j):
            # Compute Euclidean distance in feature space (D dimensions)
            cost = np.linalg.norm(seq1[i - 1] - seq2[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
            
    return float(dtw[n, m])
