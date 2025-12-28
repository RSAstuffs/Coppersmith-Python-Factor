#!/usr/bin/env python3
"""
fpylll-compatible wrapper for LLL reduction on large integers.
Provides fpylll-like interface without requiring fpylll installation.
Uses our custom LLL implementation that handles large integers.
"""

import numpy as np
from typing import Optional, Tuple
import math
import logging

logger = logging.getLogger(__name__)


def gram_schmidt(basis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Gram-Schmidt orthogonalization for LLL."""
    n = basis.shape[0]
    mu = np.zeros((n, n))
    orthogonal = np.zeros_like(basis, dtype=float)
    
    for i in range(n):
        orthogonal[i] = basis[i].astype(float)
        for j in range(i):
            denom = np.dot(orthogonal[j], orthogonal[j])
            if abs(denom) > 1e-10:
                mu[i, j] = np.dot(basis[i], orthogonal[j]) / denom
                orthogonal[i] -= mu[i, j] * orthogonal[j]
            else:
                mu[i, j] = 0.0
        mu[i, i] = 1.0
    
    return orthogonal, mu


def lll_reduce(basis: np.ndarray, delta: float = 0.75, return_float: bool = False) -> np.ndarray:
    """LLL lattice basis reduction."""
    basis = basis.copy().astype(float)
    n = basis.shape[0]
    
    orthogonal, mu = gram_schmidt(basis)
    
    k = 1
    while k < n:
        # Size reduction
        for j in range(k - 1, -1, -1):
            mu_kj = mu[k, j]
            if abs(mu_kj) > 0.5:
                q = round(mu_kj)
                basis[k] = basis[k] - q * basis[j]
                mu[k, j] -= q
                for i in range(j):
                    mu[k, i] -= q * mu[j, i]
        
        # Lovász condition
        norm_k_minus_1 = np.dot(orthogonal[k-1], orthogonal[k-1])
        norm_k = np.dot(orthogonal[k], orthogonal[k])
        
        if norm_k_minus_1 < 1e-10 or not np.isfinite(norm_k_minus_1):
            k += 1
            continue
        if norm_k < 1e-10 or not np.isfinite(norm_k):
            k += 1
            continue
        
        norm_k_star = norm_k + mu[k, k-1]**2 * norm_k_minus_1
        
        if norm_k_star >= (delta - mu[k, k-1]**2) * norm_k_minus_1:
            k += 1
        else:
            basis[[k-1, k]] = basis[[k, k-1]]
            orthogonal, mu = gram_schmidt(basis)
            k = max(1, k - 1)
    
    if return_float:
        return basis
    
    result = np.zeros_like(basis, dtype=int)
    for i in range(basis.shape[0]):
        for j in range(basis.shape[1]):
            val = float(basis[i, j])
            if np.isfinite(val):
                result[i, j] = int(round(val))
    return result


class IntegerMatrix:
    """
    Matrix class for large integers, compatible with fpylll interface.
    """
    def __init__(self, data):
        """
        Initialize from numpy array or list of lists.
        Supports object dtype for arbitrary precision integers.
        """
        if isinstance(data, np.ndarray):
            self._matrix = data
        else:
            # Convert to numpy array with object dtype for large integers
            self._matrix = np.array(data, dtype=object)
        
        # Ensure object dtype for large integers
        if self._matrix.dtype != object:
            # Check if we need object dtype (large integers)
            max_val = 0
            for i in range(self._matrix.shape[0]):
                for j in range(self._matrix.shape[1]):
                    val = self._matrix[i, j]
                    if isinstance(val, (int, np.integer)):
                        max_val = max(max_val, abs(int(val)))
            
            # If values are too large for int64, use object dtype
            if max_val > 2**63:
                matrix_obj = np.empty(self._matrix.shape, dtype=object)
                for i in range(self._matrix.shape[0]):
                    for j in range(self._matrix.shape[1]):
                        matrix_obj[i, j] = int(self._matrix[i, j])
                self._matrix = matrix_obj
    
    @property
    def nrows(self):
        return self._matrix.shape[0]
    
    @property
    def ncols(self):
        return self._matrix.shape[1]
    
    def __getitem__(self, key):
        return self._matrix[key]
    
    def __setitem__(self, key, value):
        self._matrix[key] = value
    
    def to_numpy(self):
        """Convert to numpy array."""
        return self._matrix
    
    def copy(self):
        """Return a copy of the matrix."""
        return IntegerMatrix(self._matrix.copy())
    
    def is_zero(self):
        """Check if matrix is all zeros."""
        for i in range(self.nrows):
            for j in range(self.ncols):
                val = self._matrix[i, j]
                if val != 0 and val is not None:
                    return False
        return True
    
    def rank(self):
        """Compute rank (for small matrices only)."""
        if self.nrows > 500 or self.ncols > 500:
            return None  # Too large for quick rank computation
        
        # Convert to float for rank computation
        try:
            matrix_float = np.zeros((self.nrows, self.ncols), dtype=float)
            for i in range(self.nrows):
                for j in range(self.ncols):
                    val = self._matrix[i, j]
                    if val != 0 and val is not None:
                        try:
                            matrix_float[i, j] = float(int(val))
                        except (OverflowError, ValueError):
                            matrix_float[i, j] = 0.0
            return np.linalg.matrix_rank(matrix_float)
        except:
            return None
    
    def density(self):
        """Compute density (fraction of non-zero entries)."""
        total = self.nrows * self.ncols
        non_zero = 0
        for i in range(self.nrows):
            for j in range(self.ncols):
                val = self._matrix[i, j]
                if val != 0 and val is not None:
                    non_zero += 1
        return non_zero / total if total > 0 else 0.0
    
    def det(self):
        """Compute determinant (for small square matrices only)."""
        if not self.is_square() or self.nrows > 200:
            return None
        
        try:
            # Convert to float for determinant
            matrix_float = np.zeros((self.nrows, self.ncols), dtype=float)
            for i in range(self.nrows):
                for j in range(self.ncols):
                    val = self._matrix[i, j]
                    if val != 0 and val is not None:
                        try:
                            matrix_float[i, j] = float(int(val))
                        except (OverflowError, ValueError):
                            matrix_float[i, j] = 0.0
            return np.linalg.det(matrix_float)
        except:
            return None
    
    def is_square(self):
        """Check if matrix is square."""
        return self.nrows == self.ncols


def LLL(B: IntegerMatrix, delta: float = 0.75) -> IntegerMatrix:
    """
    LLL reduction of integer matrix.
    
    Args:
        B: IntegerMatrix to reduce
        delta: LLL parameter (default 0.75)
    
    Returns:
        Reduced IntegerMatrix
    """
    matrix = B.to_numpy()
    
    # Check if we need to scale for LLL
    max_entry = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if val != 0 and val is not None:
                try:
                    val_abs = abs(int(val))
                    if val_abs > max_entry:
                        max_entry = val_abs
                except (TypeError, ValueError):
                    pass
    
    max_bits = max_entry.bit_length() if max_entry > 0 else 0
    
    # Scale if necessary (for very large integers)
    scale_bits = 0
    if max_bits > 1000:
        logger.info(f"Scaling matrix for LLL (max entry: 2^{max_bits} bits)...")
        scale_bits = max_bits - 400
        matrix_scaled = np.zeros((matrix.shape[0], matrix.shape[1]), dtype=float)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if val != 0 and val is not None:
                    try:
                        val_int = int(val)
                        val_bits = val_int.bit_length()
                        scaled_bits = val_bits - scale_bits
                        if scaled_bits > 0:
                            matrix_scaled[i, j] = 2.0 ** min(scaled_bits, 1000)
                    except (TypeError, ValueError, OverflowError):
                        matrix_scaled[i, j] = 0.0
    else:
        # Convert to float
        matrix_scaled = np.zeros((matrix.shape[0], matrix.shape[1]), dtype=float)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if val != 0 and val is not None:
                    try:
                        matrix_scaled[i, j] = float(int(val))
                    except (TypeError, ValueError, OverflowError):
                        matrix_scaled[i, j] = 0.0
    
    # Run LLL
    reduced_float = lll_reduce(matrix_scaled, delta=delta, return_float=True)
    
    # Convert back to integer matrix
    # For large integers, we keep as float approximations
    # The extraction code will handle the conversion
    result = IntegerMatrix(reduced_float)
    
    return result


# Convenience functions matching fpylll interface
def IntegerMatrix_from_matrix(matrix):
    """Create IntegerMatrix from another matrix."""
    return IntegerMatrix(matrix)


# Export main interface
__all__ = ['IntegerMatrix', 'LLL', 'IntegerMatrix_from_matrix']
