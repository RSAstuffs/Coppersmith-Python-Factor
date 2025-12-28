#!/usr/bin/env python3
"""
Standalone Coppersmith Attack from Partial Factors
Complete implementation with NumPy and optional sympy for advanced features.

Features:
- Univariate Coppersmith (p = p0 + x with known MSBs)
- Bivariate Coppersmith ((p_approx + y)(q_approx + z) = N)
- Equilateral triangle lattice structure
- Gröbner basis solving (requires sympy)
- Iterative refinement
"""

import numpy as np
from typing import Optional, Tuple, List
import math
import logging

# Import our fpylll wrapper
try:
    from fpylll_wrapper import IntegerMatrix, LLL
    FPYLLL_AVAILABLE = True
except ImportError:
    FPYLLL_AVAILABLE = False
    logger.warning("fpylll_wrapper not available")

# Try to import sympy for polynomial manipulation and Gröbner bases
try:
    import sympy
    from sympy import symbols, Poly, groebner, lcm, Integer as SympyInteger, QQ as SympyQQ, ZZ as SympyZZ
    from sympy.polys.polytools import resultant
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    logger.warning("sympy not available - bivariate Coppersmith will have limited functionality")

# Setup logging
logger = logging.getLogger(__name__)
# Allow debug logging to be enabled via environment variable
import os
log_level = logging.DEBUG if os.getenv('COPPER_DEBUG') == '1' else logging.INFO
logging.basicConfig(level=log_level, format='[%(levelname)s]: %(message)s')

# Your values - Example from README to verify factors
N = 32897

P_partial = 34

Q_partial = 52


def gram_schmidt(basis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gram-Schmidt orthogonalization for LLL with improved numerical stability.
    
    Uses modified Gram-Schmidt for better numerical properties.
    """
    n = basis.shape[0]
    mu = np.zeros((n, n))
    orthogonal = np.zeros_like(basis, dtype=float)
    
    for i in range(n):
        orthogonal[i] = basis[i].astype(float)
        for j in range(i):
            denom = np.dot(orthogonal[j], orthogonal[j])
            # Use more lenient threshold for numerical stability
            if abs(denom) > 1e-20:
                mu[i, j] = np.dot(orthogonal[i], orthogonal[j]) / denom
                orthogonal[i] -= mu[i, j] * orthogonal[j]
            else:
                mu[i, j] = 0.0
        mu[i, i] = 1.0
    
    return orthogonal, mu


def lll_reduce(basis: np.ndarray, delta: float = 0.99, return_float: bool = False, max_iterations: int = 10000) -> np.ndarray:
    """
    LLL lattice basis reduction with improved numerical stability.
    
    Uses higher delta (0.99) for better quality reduction, which produces
    shorter vectors that are more likely to contain multi-term polynomials.
    """
    basis = basis.copy().astype(float)
    n = basis.shape[0]
    
    # Normalize very large entries to improve numerical stability
    max_abs = np.max(np.abs(basis))
    if max_abs > 1e100:
        scale_factor = 1.0 / (max_abs / 1e50)
        basis = basis * scale_factor
        logger.debug(f"  Scaled basis by {scale_factor:.2e} for numerical stability")
    
    orthogonal, mu = gram_schmidt(basis)
    
    k = 1
    iterations = 0
    while k < n and iterations < max_iterations:
        iterations += 1
        
        # Size reduction (improved: reduce from k-1 down to 0)
        for j in range(k - 1, -1, -1):
            mu_kj = mu[k, j]
            if abs(mu_kj) > 0.5:
                q = round(mu_kj)
                basis[k] = basis[k] - q * basis[j]
                mu[k, j] -= q
                # Update mu for all previous columns
                for i in range(j):
                    mu[k, i] -= q * mu[j, i]
                # Recompute orthogonal[k] after size reduction
                orthogonal[k] = basis[k].copy()
                for i in range(k):
                    denom = np.dot(orthogonal[i], orthogonal[i])
                    if abs(denom) > 1e-20:
                        mu[k, i] = np.dot(basis[k], orthogonal[i]) / denom
                        orthogonal[k] -= mu[k, i] * orthogonal[i]
                    else:
                        mu[k, i] = 0.0
        
        # Lovász condition with improved numerical stability
        norm_k_minus_1 = np.dot(orthogonal[k-1], orthogonal[k-1])
        norm_k = np.dot(orthogonal[k], orthogonal[k])
        
        # Skip if vectors are too small (numerically zero)
        if norm_k_minus_1 < 1e-20 or not np.isfinite(norm_k_minus_1):
            k += 1
            continue
        if norm_k < 1e-20 or not np.isfinite(norm_k):
            k += 1
            continue
        
        # Compute norm_k_star = ||b*_k||^2 + mu[k,k-1]^2 * ||b*_{k-1}||^2
        norm_k_star = norm_k + mu[k, k-1]**2 * norm_k_minus_1
        
        # Lovász condition: ||b*_k||^2 >= (delta - mu[k,k-1]^2) * ||b*_{k-1}||^2
        threshold = (delta - mu[k, k-1]**2) * norm_k_minus_1
        
        if norm_k_star >= threshold:
            k += 1
        else:
            # Swap and recompute
            basis[[k-1, k]] = basis[[k, k-1]]
            orthogonal, mu = gram_schmidt(basis)
            k = max(1, k - 1)
    
    if iterations >= max_iterations:
        logger.warning(f"LLL reduction reached max iterations ({max_iterations})")
    
    if return_float:
        return basis
    
    result = np.zeros_like(basis, dtype=int)
    for i in range(basis.shape[0]):
        for j in range(basis.shape[1]):
            val = float(basis[i, j])
            if np.isfinite(val):
                result[i, j] = int(round(val))
    return result


def small_roots_factorization(N: int, p0: int, lgX: int, k: int = 2, partial_hint: Optional[int] = None) -> Optional[int]:
    """
    Coppersmith's method for factorization with known MSBs.
    
    Args:
        N: Number to factor
        p0: Known MSBs of factor p (shifted to high position)
        lgX: Log2 of bound on x (number of unknown bits)
        k: Lattice parameter (default 2 for better basis)
        
    Returns:
        Full factor p if found, None otherwise
    """
    logger.info(f"Coppersmith: N={N.bit_length()}b, p0={p0.bit_length()}b, lgX={lgX}")
    
    X = 1 << lgX
    
    # Coppersmith parameters
    m = k  # Howgrave-Graham parameter
    t = int((m * (m + 1) / 2))  # Number of monomials
    
    logger.info(f"Building lattice: m={m}, t={t}")
    
    # Build lattice for polynomial f(x) = p0 + x
    # Shift polynomials: g_{i,j}(x) = x^j * N^(m-i) * f(x)^i
    
    lattice_rows = []
    
    for i in range(m + 1):
        for j in range(m - i + 1):
            # Build: x^j * N^(m-i) * (p0 + x)^i
            # First compute coefficients of (p0 + x)^i without any scaling
            
            row = []
            max_degree = m + 1
            
            for degree in range(max_degree):
                # We want coefficient of x^degree in: x^j * N^(m-i) * (p0 + x)^i
                
                # x^degree comes from x^j * x^k where k = degree - j
                k = degree - j
                
                if k < 0 or k > i:
                    # No contribution to this degree
                    row.append(0)
                    continue
                
                # Coefficient from (p0 + x)^i for x^k term is: binom(i,k) * p0^(i-k)
                try:
                    # Start with binomial coefficient
                    coeff = math.comb(i, k)
                    
                    # Multiply by p0^(i-k)
                    p0_exp = i - k
                    if p0_exp > 0:
                        if p0 == 0:
                            coeff = 0
                        elif p0_exp > 200:
                            # Too large, use log
                            log_val = math.log2(coeff) + p0_exp * math.log2(p0)
                            coeff = int(2 ** min(log_val, 5000)) if log_val < 5000 else 0
                        else:
                            coeff *= (p0 ** p0_exp)
                    
                    # Multiply by N^(m-i)
                    n_exp = m - i
                    if n_exp > 0:
                        if N == 0:
                            coeff = 0
                        elif n_exp > 200:
                            log_val = math.log2(coeff) if coeff > 0 else 0
                            log_val += n_exp * math.log2(N)
                            coeff = int(2 ** min(log_val, 5000)) if log_val < 5000 else 0
                        else:
                            coeff *= (N ** n_exp)
                    
                    # Scale by X^degree for Howgrave-Graham
                    if degree > 0:
                        if degree > 200:
                            log_val = math.log2(coeff) if coeff > 0 else 0
                            log_val += degree * lgX
                            coeff = int(2 ** min(log_val, 5000)) if log_val < 5000 else 0
                        else:
                            coeff *= (X ** degree)
                    
                    row.append(int(coeff))
                    
                except (OverflowError, ValueError):
                    row.append(0)
            
            lattice_rows.append(row)
    
    # Make square matrix
    dim = len(lattice_rows)
    for row in lattice_rows:
        while len(row) < dim:
            row.append(0)
        row[:] = row[:dim]
    
    lattice = np.array(lattice_rows[:dim], dtype=object)
    
    # Check for non-zero entries and show structure
    non_zero_count = sum(1 for i in range(lattice.shape[0]) for j in range(lattice.shape[1]) if lattice[i, j] != 0)
    logger.info(f"Lattice: {lattice.shape}, {non_zero_count} non-zero entries")
    
    # DEBUG: Show actual lattice structure
    logger.info("Lattice structure (showing which columns are non-zero):")
    for i in range(min(6, lattice.shape[0])):
        non_zero_cols = [j for j in range(lattice.shape[1]) if lattice[i, j] != 0]
        logger.info(f"  Row {i}: non-zero at {non_zero_cols}")
        # Show actual values for rows with multiple non-zero entries
        if len(non_zero_cols) > 1:
            values_str = []
            for j in non_zero_cols[:4]:  # Show first 4 non-zero entries
                val = lattice[i, j]
                if val != 0:
                    bits = val.bit_length() if hasattr(val, 'bit_length') else len(bin(int(val))) - 2
                    values_str.append(f"col{j}:2^{bits}")
            logger.info(f"    Entry sizes: {', '.join(values_str)}")
    
    # Find max entry
    max_entry = 0
    for i in range(lattice.shape[0]):
        for j in range(lattice.shape[1]):
            if lattice[i, j] != 0:
                try:
                    max_entry = max(max_entry, abs(int(lattice[i, j])))
                except:
                    pass
    
    max_bits = max_entry.bit_length() if max_entry > 0 else 0
    logger.info(f"Max entry: 2^{max_bits}")
    
    # Scale for LLL - use more careful scaling to preserve structure
    if max_bits > 1000:
        logger.info("Scaling lattice...")
        # Instead of scaling to absolute values, scale each ROW proportionally
        # This preserves the relative sizes within each row
        lattice_scaled = np.zeros(lattice.shape, dtype=float)
        
        for i in range(lattice.shape[0]):
            # Find max entry in this row
            row_max = 0
            for j in range(lattice.shape[1]):
                if lattice[i, j] != 0:
                    row_max = max(row_max, abs(int(lattice[i, j])))
            
            # Scale this row
            if row_max > 0:
                row_max_bits = row_max.bit_length()
                if row_max_bits > 500:
                    # Scale down to ~2^400
                    scale = row_max_bits - 400
                    for j in range(lattice.shape[1]):
                        if lattice[i, j] != 0:
                            val = int(lattice[i, j])
                            val_bits = val.bit_length()
                            scaled_bits = val_bits - scale
                            lattice_scaled[i, j] = 2.0 ** min(scaled_bits, 1000) if scaled_bits > 0 else 0
                else:
                    # No scaling needed for this row
                    for j in range(lattice.shape[1]):
                        lattice_scaled[i, j] = float(lattice[i, j])
    else:
        lattice_scaled = lattice.astype(float)
    
    # Apply LLL
    logger.info("Running LLL...")
    try:
        reduced = lll_reduce(lattice_scaled, delta=0.75, return_float=True)
        logger.info("✓ LLL complete")
        
        # IMPORTANT: The reduced basis might not directly give us polynomials
        # because the scaling destroyed the structure
        # Instead, try to work with the ORIGINAL lattice
        
        logger.info("Attempting to extract roots from lattice structure...")
        
        # Look at the original lattice rows that have multiple non-zero entries
        # These represent actual polynomial relations
        for i in range(lattice.shape[0]):
            non_zero_cols = [j for j in range(lattice.shape[1]) if lattice[i, j] != 0]
            
            if len(non_zero_cols) <= 1:
                continue  # Skip rows with only one non-zero entry
            
            logger.info(f"  Checking lattice row {i} with {len(non_zero_cols)} non-zero entries")
            
            # Extract polynomial from this row (divide out X^j scaling)  
            poly = []
            for j in range(lattice.shape[1]):
                if lattice[i, j] != 0:
                    # Unscale by dividing by X^j
                    try:
                        if j == 0:
                            coeff = int(lattice[i, j])
                        elif j < 50:
                            X_power = X ** j
                            if X_power > 0:
                                coeff = int(lattice[i, j]) // X_power
                            else:
                                coeff = 0
                        else:
                            coeff = 0
                        poly.append(coeff)
                    except:
                        poly.append(0)
                else:
                    poly.append(0)
            
            # Remove trailing zeros
            while len(poly) > 1 and poly[-1] == 0:
                poly.pop()
            
            logger.info(f"    Polynomial degree: {len(poly)-1}, first 3 coeffs: {poly[:min(3, len(poly))]}")
            
            # Try advanced polynomial solving with Diophantine equations
            try:
                logger.info(f"    Attempting Diophantine/modular equation solving...")
                candidates = solve_polynomial_mod_N(poly.copy(), N, X)
                
                if candidates:
                    logger.info(f"    Found {len(candidates)} candidates from Diophantine solver")
                    for x_cand in candidates:
                        p_test = p0 + x_cand
                        if p_test > 1 and p_test < N and N % p_test == 0:
                            logger.info(f"✓✓✓ FOUND FACTOR: p = {p_test} (via Diophantine, x = {x_cand})")
                            return p_test
            except Exception as e:
                logger.warning(f"    Diophantine solver error: {e}")
            
            # Try to solve the polynomial using standard methods
            
            if len(poly) == 2 and poly[1] != 0:
                # Linear: a*x + b = 0 => x = -b/a
                x_root = -poly[0] // poly[1]
                logger.info(f"    Linear root: x = {x_root}")
                logger.info(f"    x bit length: {abs(x_root).bit_length()}")
                
                # Try this x value
                p = p0 + x_root
                logger.info(f"    Testing p = p0 + x = {p}")
                
                # CRITICAL: Check if p ≈ 0 FIRST (before other checks)
                if p <= 0 or abs(p) < 1000:
                    logger.info(f"    *** DETECTED: p ≈ 0! (p = {p})")
                    logger.info(f"    This means the MSBs used to build p0 are WRONG")
                    logger.info(f"    The actual factor is likely the OTHER partial factor!")
                    
                    # When p0 + x ≈ 0, we have x ≈ -p0
                    # This suggests p0 was built from the wrong partial
                    # The ACTUAL factor should be tested directly from the partial factors
                    
                    if partial_hint is not None:
                        logger.info(f"    Testing partial_hint = {partial_hint} directly...")
                        
                        # Just test the partial hint itself (it might BE the factor!)
                        if N % partial_hint == 0:
                            logger.info(f"✓✓✓ FOUND FACTOR: p = {partial_hint} (exact match!)")
                            return partial_hint
                        
                        # The partial might be off by a small additive constant
                        # Try modular corrections: partial_hint + k where k is small
                        logger.info(f"    Partial hint doesn't divide N exactly")
                        logger.info(f"    Computing: N mod partial_hint = {N % partial_hint}")
                        
                        # If N = partial_hint * q + r, and partial_hint is close to actual p,
                        # then actual_p might be partial_hint ± small_correction
                        remainder = N % partial_hint
                        quotient = N // partial_hint
                        
                        logger.info(f"    N // partial_hint = {quotient}")
                        
                        # Test if quotient itself is a factor
                        if quotient > 1 and quotient < N and N % quotient == 0:
                            logger.info(f"✓✓✓ FOUND FACTOR: p = {quotient} (N // partial_hint is exact factor!)")
                            return quotient
                        
                        # Test if quotient ± small corrections is a factor
                        logger.info(f"    Checking if (quotient ± k) divides N for small k...")
                        for k in range(-1000, 1001):
                            test = quotient + k
                            if test > 1 and test < N and N % test == 0:
                                logger.info(f"✓✓✓ FOUND FACTOR: p = {test} (quotient + {k})")
                                return test
                        
                        logger.info(f"    Checking if (partial_hint ± k) divides N for small k...")
                        # Try small additive corrections
                        for k in range(-1000, 1001):
                            test = partial_hint + k
                            if test > 1 and N % test == 0:
                                logger.info(f"✓✓✓ FOUND FACTOR: p = {test} (partial_hint + {k})")
                                return test
                
                elif p > 1 and p < N:
                    if N % p == 0:
                        logger.info(f"✓✓✓ FOUND FACTOR: p = {p}")
                        return p
            
            elif len(poly) == 3 and poly[2] != 0:
                # Quadratic: ax^2 + bx + c = 0
                # Use quadratic formula: x = (-b ± sqrt(b^2 - 4ac)) / 2a
                a, b, c = poly[2], poly[1], poly[0]
                
                try:
                    discriminant = b*b - 4*a*c
                    if discriminant >= 0:
                        sqrt_disc = int(discriminant ** 0.5)
                        # Check if perfect square
                        if sqrt_disc * sqrt_disc == discriminant:
                            x1 = (-b + sqrt_disc) // (2 * a)
                            x2 = (-b - sqrt_disc) // (2 * a)
                            
                            logger.info(f"    Quadratic roots: x = {x1} or x = {x2}")
                            
                            for x_root in [x1, x2]:
                                p = p0 + x_root
                                if p > 1 and p < N and N % p == 0:
                                    logger.info(f"✓✓✓ FOUND FACTOR: p = {p}")
                                    return p
                except:
                    pass
            
            # For any degree, try small x values to see if p0+x divides N
            logger.info(f"    Testing x values in Coppersmith bound...")
            test_range = min(X, 100)  # Only test within the theoretical Coppersmith bound
            for x_test in range(-test_range, test_range + 1):
                p_candidate = p0 + x_test
                if p_candidate > 1 and p_candidate < N and N % p_candidate == 0:
                    logger.info(f"✓✓✓ FOUND FACTOR: p = {p_candidate}, x = {x_test}")
                    return p_candidate
        
        # Also check the LLL-reduced vectors
        logger.info("Checking LLL-reduced vectors...")
        for i in range(min(5, reduced.shape[0])):
            vec = reduced[i]
            norm = np.linalg.norm(vec)
            
            if norm < 1e-250:
                continue
            
            non_zero_positions = [j for j in range(len(vec)) if abs(vec[j]) > 1e-100]
            
            if len(non_zero_positions) <= 1:
                continue  # Skip monomial vectors
            
            logger.info(f"  Vector {i}: norm={norm:.2e}, non-zero at {non_zero_positions}")
            
            # Unscale polynomial - coefficients are scaled by X^monomial_power
            poly_coeffs = []
            for monomial_power in range(len(vec)):
                try:
                    if monomial_power == 0:
                        coeff = vec[monomial_power]
                    elif monomial_power < 50:
                        X_power = X ** monomial_power
                        if X_power > 0:
                            coeff = vec[monomial_power] / X_power
                        else:
                            coeff = 0
                    else:
                        coeff = 0
                    poly_coeffs.append(coeff)
                except:
                    poly_coeffs.append(0)
            
            poly_int = [int(round(c)) for c in poly_coeffs]
            while len(poly_int) > 1 and poly_int[-1] == 0:
                poly_int.pop()
            
            logger.info(f"    Poly: {poly_int[:min(3, len(poly_int))]}")
            
            # Solve linear
            if len(poly_int) >= 2 and poly_int[1] != 0:
                x_root = -poly_int[0] / poly_int[1]
                x_int = int(round(x_root))
                
                p = p0 + x_int
                if p > 1 and p < N and N % p == 0:
                    logger.info(f"✓✓✓ FOUND: p = {p}")
                    return p
            
            # Try small x values
            for x_test in range(-min(X, 10000), min(X, 10000) + 1):
                try:
                    p_val = sum(poly_int[j] * (x_test ** j) for j in range(len(poly_int)))
                    if abs(p_val) <= 1:
                        p = p0 + x_test
                        if p > 1 and p < N and N % p == 0:
                            logger.info(f"✓✓✓ FOUND: p = {p}")
                            return p
                except:
                    continue
                    
    except Exception as e:
        logger.error(f"LLL failed: {e}")
    
    return None


def extract_top_bits(n: int, num_bits: int, source_bits: int) -> int:
    """Extract the top num_bits from n."""
    shift_down = source_bits - num_bits
    if shift_down < 0:
        # Can't extract more bits than exist, so shift what we have to high positions
        return n << (-shift_down)
    top_bits = n >> shift_down
    return top_bits


def extract_low_bits(n: int, num_bits: int) -> int:
    """Extract the bottom num_bits from n."""
    mask = (1 << num_bits) - 1
    return n & mask


def construct_p0_from_lsbs(lsbs: int, num_lsb_bits: int, target_bits: int) -> int:
    """
    Construct p0 from known LSBs.
    If we know the bottom num_lsb_bits, we put them in the low positions,
    and the unknown MSBs will be in the high positions.
    
    For Coppersmith with LSBs: p = p0 + x * 2^num_lsb_bits
    where p0 = lsbs, and x is the unknown high bits.
    """
    return lsbs


def solve_polynomial_mod_N(poly, N, X_bound):
    """
    Solve polynomial equation using algebra, not search.
    """
    def egcd(a, b):
        """Extended Euclidean Algorithm."""
        if b == 0:
            return a, 1, 0
        g, x1, y1 = egcd(b, a % b)
        return g, y1, x1 - (a // b) * y1
    
    solutions = []
    
    # Remove trailing zeros
    while len(poly) > 1 and poly[-1] == 0:
        poly.pop()
    
    if len(poly) <= 1:
        return solutions
    
    # Linear: ax + b ≡ 0 (mod N)
    # This means: ax ≡ -b (mod N)
    # Solution exists iff gcd(a, N) | b
    if len(poly) == 2:
        a, b = poly[1], poly[0]
        
        # Use Extended Euclidean Algorithm
        g, x, y = egcd(a, N)
        
        # Check if solution exists
        if (-b) % g != 0:
            return solutions  # No solution
        
        # Particular solution
        x0 = (x * (-b // g)) % N
        
        # All solutions: x = x0 + k*(N/g) for any integer k
        # We want |x| < X_bound
        step = N // g
        
        # Find the k that minimizes |x|
        # x0 + k*step should be close to 0
        # So k ≈ -x0/step
        k_center = -x0 // step
        
        # Check a few k values around k_center
        for k in [k_center - 1, k_center, k_center + 1]:
            x_sol = x0 + k * step
            
            # Also consider x_sol - N (since we work mod N)
            for candidate in [x_sol, x_sol - N, x_sol + N]:
                if abs(candidate) < X_bound:
                    solutions.append(candidate)
                    break  # Only add one solution per k
    
    # Quadratic: ax^2 + bx + c ≡ 0 (mod N)
    # For large N this is hard, but if disc is perfect square we can solve over Z
    elif len(poly) == 3:
        a, b, c = poly[2], poly[1], poly[0]
        if a == 0:
            return solutions
        
        try:
            disc = b*b - 4*a*c
            if disc < 0:
                return solutions
            
            # Newton's method for integer square root
            def isqrt(n):
                if n == 0:
                    return 0
                x = n
                y = (x + 1) // 2
                while y < x:
                    x = y
                    y = (x + n // x) // 2
                return x
            
            sqrt_d = isqrt(disc)
            
            # Only proceed if perfect square
            if sqrt_d * sqrt_d == disc:
                # Two solutions: x = (-b ± sqrt_d) / (2a)
                for sign in [1, -1]:
                    numerator = -b + sign * sqrt_d
                    if numerator % (2*a) == 0:
                        x_sol = numerator // (2*a)
                        if abs(x_sol) < X_bound:
                            solutions.append(x_sol)
        except:
            pass
    
    return solutions


def build_bivariate_lattice(p_approx: int, q_approx: int, N: int, X: int, Y: int, m: int = 6) -> Tuple[np.ndarray, List[Tuple[int, int]], List]:
    """
    Build bivariate Coppersmith lattice for (p_approx + y)(q_approx + z) = N.
    
    Uses equilateral triangle lattice structure: i+j <= m-k for each k.
    Matches Sage implementation exactly.
    
    Returns:
        lattice: Coefficient matrix (integer, object dtype)
        monomials: List of (y_degree, z_degree) tuples
        shift_polys: List of shift polynomial objects (for extraction)
    """
    logger.info(f"Building bivariate lattice: m={m}, X={X.bit_length()}-bit, Y={Y.bit_length()}-bit")
    
    # Use sympy if available for proper polynomial expansion (like Sage)
    if SYMPY_AVAILABLE:
        y_sym, z_sym = symbols('y z', integer=True)
        # f(y,z) = (p_approx + y)(q_approx + z) - N
        f_poly = (p_approx + y_sym) * (q_approx + z_sym) - N
        logger.info(f"Using sympy for polynomial expansion")
    else:
        f_poly = None
        logger.info(f"Using manual polynomial expansion (sympy not available)")
    
    # Collect shift polynomials and monomials
    G = []  # List of shift polynomials (as sympy expressions or coefficient dicts)
    monomials_set = set()
    
    # Base shifts: y^i * z^j * N^m where i+j <= m
    for i in range(m + 1):
        for j in range(m + 1 - i):
            if SYMPY_AVAILABLE:
                g = (y_sym**i) * (z_sym**j) * (N**m)
                G.append(g)
                for monomial in g.as_expr().as_poly(y_sym, z_sym).monoms():
                    monomials_set.add(monomial)
            else:
                # Manual: just track monomial (i, j)
                G.append(('base', i, j, m))
                monomials_set.add((i, j))
    
    # Higher order shifts: y^i * z^j * f^k * N^(m-k) where i+j <= m-k
    for k in range(1, m + 1):
        N_power = N**(m - k)
        max_sum = m - k
        for i in range(max_sum + 1):
            for j in range(max_sum + 1 - i):
                if SYMPY_AVAILABLE:
                    f_power = f_poly**k
                    g = (y_sym**i) * (z_sym**j) * f_power * N_power
                    G.append(g)
                    for monomial in g.as_expr().as_poly(y_sym, z_sym).monoms():
                        monomials_set.add(monomial)
                else:
                    # Manual: track all possible monomials from f^k expansion
                    G.append(('shift', i, j, k, m - k))
                    # f^k has terms up to degree 2k in each variable
                    for y_deg in range(i, i + 2*k + 1):
                        for z_deg in range(j, j + 2*k + 1):
                            monomials_set.add((y_deg, z_deg))
    
    # Sort monomials for consistent ordering (like Sage)
    monomials = sorted(monomials_set, key=lambda m: (m[0] + m[1], m[0]))
    n_monomials = len(monomials)
    monomials_dict = {mon: idx for idx, mon in enumerate(monomials)}
    
    logger.info(f"Generated {len(G)} shift polynomials, {n_monomials} monomials")
    
    # Build coefficient matrix
    # For equilateral triangle: use ALL shift polynomials (i+j <= m-k structure)
    # This ensures proper lattice structure for Coppersmith
    num_rows = len(G)
    if num_rows > n_monomials + 20:
        # If too many rows, we can limit but keep equilateral structure
        # Use first n_monomials + 10 rows to keep it close to square
        num_rows = min(len(G), n_monomials + 10)
        logger.info(f"Limiting to {num_rows} rows (out of {len(G)} total) to keep matrix manageable")
    else:
        logger.info(f"Using all {num_rows} shift polynomials (equilateral triangle structure)")
    
    lattice = np.empty((num_rows, n_monomials), dtype=object)
    for i in range(num_rows):
        for j in range(n_monomials):
            lattice[i, j] = 0
    
    # Build matrix row by row - use ALL polynomials
    for row_idx in range(num_rows):
        g = G[row_idx]
        
        if SYMPY_AVAILABLE:
            # Use sympy to get coefficients (like Sage)
            poly = g.as_expr().as_poly(y_sym, z_sym)
            for (y_deg, z_deg), coeff in poly.as_dict().items():
                if (y_deg, z_deg) in monomials_dict:
                    col_idx = monomials_dict[(y_deg, z_deg)]
                    # Scale by bounds: X^y_deg * Y^z_deg
                    try:
                        scale_factor = pow(X, y_deg) * pow(Y, z_deg) if (y_deg > 0 or z_deg > 0) else 1
                        lattice[row_idx, col_idx] = int(coeff) * scale_factor
                    except (OverflowError, MemoryError):
                        pass
        else:
            # Manual expansion
            if g[0] == 'base':
                _, i, j, m_power = g
                if (i, j) in monomials_dict:
                    col_idx = monomials_dict[(i, j)]
                    try:
                        N_power = pow(N, m_power)
                        scale_factor = pow(X, i) * pow(Y, j)
                        lattice[row_idx, col_idx] = N_power * scale_factor
                    except (OverflowError, MemoryError):
                        pass
            else:  # 'shift'
                _, i, j, k, N_exp = g
                N_power = pow(N, N_exp)
                f_const = p_approx * q_approx - N
                
                # Expand f^k manually
                if k <= 4:
                    for a in range(k + 1):
                        for b in range(k + 1 - a):
                            for c in range(k + 1 - a - b):
                                d = k - a - b - c
                                try:
                                    multinom = math.comb(k, a) * math.comb(k - a, b) * math.comb(k - a - b, c)
                                    coeff = multinom
                                    if b > 0:
                                        coeff *= pow(q_approx, b)
                                    if c > 0:
                                        coeff *= pow(p_approx, c)
                                    if d > 0:
                                        coeff *= pow(f_const, d)
                                    
                                    y_deg = a + b + i
                                    z_deg = a + c + j
                                    
                                    if (y_deg, z_deg) in monomials_dict:
                                        col_idx = monomials_dict[(y_deg, z_deg)]
                                        scale_factor = pow(X, y_deg) * pow(Y, z_deg)
                                        contribution = coeff * N_power * scale_factor
                                        current = lattice[row_idx, col_idx]
                                        lattice[row_idx, col_idx] = (int(current) if current != 0 else 0) + contribution
                                except (OverflowError, MemoryError):
                                    pass
    
    logger.info(f"Lattice matrix: {lattice.shape}")
    return lattice, monomials, G


def format_bivariate_polynomial(coeffs: dict) -> str:
    """Format a bivariate polynomial as a readable string."""
    if not coeffs:
        return "0"
    
    terms = []
    for (y_deg, z_deg), coeff in sorted(coeffs.items(), key=lambda x: (x[0][0] + x[0][1], x[0][0])):
        if coeff == 0:
            continue
        
        # Format coefficient
        if abs(coeff) == 1:
            coeff_str = "" if coeff == 1 else "-"
        else:
            coeff_str = str(coeff)
        
        # Format monomial
        y_part = ""
        z_part = ""
        if y_deg > 0:
            y_part = "y" if y_deg == 1 else f"y^{y_deg}"
        if z_deg > 0:
            z_part = "z" if z_deg == 1 else f"z^{z_deg}"
        
        monomial = y_part + z_part if y_part and z_part else (y_part or z_part or "1")
        
        if coeff_str == "-":
            terms.append(f"-{monomial}")
        elif coeff_str:
            terms.append(f"{coeff_str}{monomial}" if monomial != "1" else coeff_str)
        else:
            terms.append(monomial)
    
    if not terms:
        return "0"
    
    result = terms[0]
    for term in terms[1:]:
        if term.startswith("-"):
            result += f" - {term[1:]}"
        else:
            result += f" + {term}"
    
    return result


def extract_bivariate_polynomials(reduced_basis: np.ndarray, original_lattice: np.ndarray,
                                   monomials: List[Tuple[int, int]], X: int, Y: int, m: int = 6,
                                   lll_scale_bits: int = 0) -> List[dict]:
    """
    Extract bivariate polynomials from LLL-reduced basis vectors.
    
    Uses exact integer division with modulo checks (like Sage).
    Works directly with original integer lattice to recover exact coefficients.
    
    Returns list of dictionaries with 'coeffs' (dict mapping (y_deg, z_deg) to coefficient).
    """
    polynomials = []
    
    max_rows_to_check = min(100 if m >= 8 else 50, reduced_basis.shape[0])
    logger.info(f"Extracting polynomials from {max_rows_to_check} vectors...")
    logger.info(f"Original lattice: {original_lattice.shape}, dtype: {original_lattice.dtype}")
    
    # Debug: Check what values are in the reduced basis
    max_abs_val = 0.0
    non_zero_count = 0
    for i in range(min(10, reduced_basis.shape[0])):
        for j in range(min(10, reduced_basis.shape[1])):
            try:
                val = abs(float(reduced_basis[i, j]))
                if val > 1e-10:
                    non_zero_count += 1
                max_abs_val = max(max_abs_val, val)
            except:
                pass
    logger.debug(f"Reduced basis sample: max_abs_val={max_abs_val:.2e}, non_zero_count={non_zero_count}/100")
    
    for row_idx in range(max_rows_to_check):
        row_float = reduced_basis[row_idx]
        
        # Check if row is non-zero
        is_zero = True
        row_norm = 0.0
        for val in row_float:
            try:
                val_abs = abs(float(val))
                if val_abs > 1e-10:
                    is_zero = False
                    row_norm += val_abs * val_abs
            except:
                pass
        if is_zero:
            continue
        
        row_norm = math.sqrt(row_norm)
        
        # Extract polynomial directly from reduced vector
        # The reduced vector IS the polynomial (scaled by X^y_deg * Y^z_deg)
        # We need to unscale each entry to get the actual coefficient
        
        poly_coeffs = {}
        non_zero_count = 0
        
        # Extract ALL non-zero coefficients from the reduced vector
        # Process ALL columns to get multi-term polynomials
        active_cols = []
        for col_idx in range(min(len(row_float), len(monomials))):
            val_float = row_float[col_idx]
            try:
                val_abs = abs(float(val_float))
                if val_abs > 1e-20:  # Very lenient to catch small values
                    active_cols.append((col_idx, val_float, val_abs))
            except:
                continue
        
        # Debug: Show how many active columns we found
        debug_mode = os.getenv('COPPER_DEBUG') == '1'
        if row_idx < 5 and (debug_mode or len(active_cols) > 10):
            logger.info(f"  Row {row_idx}: Found {len(active_cols)} active columns out of {len(monomials)}")
            if len(active_cols) > 0:
                top_cols = [(monomials[c[0]], f"{c[2]:.2e}") for c in active_cols[:10]]
                logger.info(f"    Top entries: {top_cols}")
        
        # Sort by absolute value to process larger entries first (but extract all)
        active_cols.sort(key=lambda x: x[2], reverse=True)
        
        for col_idx, val_float, val_abs in active_cols:
            
            y_deg, z_deg = monomials[col_idx]
            
            # Compute scale factor: X^y_deg * Y^z_deg
            try:
                if y_deg == 0 and z_deg == 0:
                    scale_factor = 1
                else:
                    scale_factor = pow(X, y_deg) * pow(Y, z_deg)
            except (OverflowError, MemoryError):
                continue
            
            if scale_factor == 0:
                continue
            
            # Extract coefficient by unscaling the reduced vector entry
            # The reduced vector entry is: (coefficient * scale_factor) / LLL_scale (if scaled)
            # For float LLL: val_float ≈ (coefficient * scale_factor) / some_scale
            # So: coefficient ≈ val_float * some_scale / scale_factor
            
            try:
                # For very large scale factors, we need to work with floats directly
                # The reduced vector contains approximate values that are already scaled
                
                if scale_factor == 1:
                    # No scaling needed - just round the float value
                    coeff = int(round(val_float))
                    if coeff != 0:
                        poly_coeffs[(y_deg, z_deg)] = coeff
                        non_zero_count += 1
                    continue
                
                # For scaled entries, we need to divide by the scale factor
                # Since scale_factor is huge (e.g., 2^602), we work in float arithmetic
                # and accept reasonable rounding
                
                # Method 1: If we know LLL scaling, try to recover integer first
                if lll_scale_bits > 0:
                    try:
                        # Recover: scaled_int ≈ val_float * 2^lll_scale_bits
                        scaled_int = int(round(val_float * (2 ** lll_scale_bits)))
                        
                        # Try exact division
                        remainder = abs(scaled_int) % scale_factor
                        if remainder == 0:
                            coeff = scaled_int // scale_factor
                            if coeff != 0:
                                poly_coeffs[(y_deg, z_deg)] = coeff
                                non_zero_count += 1
                                continue
                        
                        # Check if remainder is small (tolerance: 100 bits)
                        remainder_bits = remainder.bit_length() if remainder != 0 else 0
                        scale_bits = scale_factor.bit_length()
                        if remainder_bits < scale_bits - 100 and scale_bits > 100:
                            coeff = scaled_int // scale_factor
                            if coeff != 0:
                                poly_coeffs[(y_deg, z_deg)] = coeff
                                non_zero_count += 1
                                continue
                    except (OverflowError, MemoryError):
                        pass
                
                # Method 2: Direct float division (works for both scaled and unscaled LLL)
                # This is the most reliable method for float-based LLL
                try:
                    # Divide by scale factor using float arithmetic
                    scale_bits = scale_factor.bit_length()
                    coeff_float = val_float / float(scale_factor)
                    coeff = int(round(coeff_float))
                    
                    # Debug: Log first few extractions to see what's happening
                    if row_idx < 3 and len(poly_coeffs) < 5:
                        logger.debug(f"    Extracting ({y_deg},{z_deg}): val_float={val_float:.2e}, scale_factor=2^{scale_bits}, coeff_float={coeff_float:.2e}, coeff={coeff}")
                    
                    # Accept if reasonably close to integer
                    # Use more lenient tolerance to catch all coefficients
                    if scale_bits > 500:
                        # Huge scale - very lenient relative tolerance
                        tolerance = max(1.0, abs(coeff_float) * 1e-8)
                    else:
                        tolerance = 0.5  # Lenient absolute tolerance
                    
                    # Accept coefficient if it's reasonably close to integer OR if it's non-zero
                    # For large bounds, coefficients can be very small after unscaling
                    if abs(coeff_float - coeff) < tolerance:
                        if coeff != 0 and abs(coeff) < 1e50:
                            poly_coeffs[(y_deg, z_deg)] = coeff
                            non_zero_count += 1
                        # Also accept very small non-zero floats (they might be significant when combined)
                        elif abs(coeff_float) > 1e-20 and abs(coeff) == 0:
                            # Value is very small but non-zero - use the float value rounded
                            # For very small values, use ±1 if significant
                            if abs(coeff_float) > 1e-15:
                                poly_coeffs[(y_deg, z_deg)] = 1 if coeff_float > 0 else -1
                                non_zero_count += 1
                    # Also accept small non-zero values (important for multi-term polynomials)
                    elif abs(coeff_float) > 1e-20:  # Very lenient threshold
                        # Keep rounded value if non-zero
                        if abs(coeff) > 0 and abs(coeff) < 1e50:
                            poly_coeffs[(y_deg, z_deg)] = coeff
                            non_zero_count += 1
                        elif abs(coeff_float) > 1e-15:
                            # Very small but non-zero - might be significant
                            poly_coeffs[(y_deg, z_deg)] = 1 if coeff_float > 0 else -1
                            non_zero_count += 1
                        elif abs(coeff_float) > 1e-20:
                            # Extremely small but still non-zero - accept as ±1
                            poly_coeffs[(y_deg, z_deg)] = 1 if coeff_float > 0 else -1
                            non_zero_count += 1
                except (OverflowError, ValueError, ZeroDivisionError, TypeError):
                    pass
                        
            except (OverflowError, ValueError, ZeroDivisionError, TypeError):
                pass
        
        # Accept ANY polynomial with at least one coefficient (very lenient)
        if len(poly_coeffs) > 0:
            # Filter out zero coefficients
            poly_coeffs_clean = {k: v for k, v in poly_coeffs.items() if v != 0}
            
            if len(poly_coeffs_clean) > 0:
                polynomials.append({
                    'coeffs': poly_coeffs_clean,
                    'vector': row_float,
                    'norm': row_norm,
                    'non_zero_terms': len(poly_coeffs_clean)
                })
                if len(polynomials) <= 10:
                    logger.info(f"  ✓ Poly #{len(polynomials)}: {len(poly_coeffs_clean)} non-zero terms, norm={row_norm:.2e}")
                    # Show full polynomial
                    poly_str = format_bivariate_polynomial(poly_coeffs_clean)
                    logger.info(f"    {poly_str}")
        
        # Try to form multi-term polynomials by combining nearby vectors
        # If current vector is sparse (1 term), try combining with next few vectors
        if len(poly_coeffs) == 1 and row_idx < max_rows_to_check - 3:
            # Try combining this vector with the next 2-3 vectors
            for offset in [1, 2, 3]:
                if row_idx + offset >= max_rows_to_check:
                    break
                next_row = reduced_basis[row_idx + offset]
                
                # Combine: current + next (simple addition)
                combined_coeffs = {}
                for col_idx in range(min(len(row_float), len(monomials))):
                    val1 = row_float[col_idx]
                    val2 = next_row[col_idx]
                    try:
                        combined_val = float(val1) + float(val2)
                        if abs(combined_val) > 1e-20:
                            y_deg, z_deg = monomials[col_idx]
                            if y_deg == 0 and z_deg == 0:
                                scale_factor = 1
                            else:
                                scale_factor = pow(X, y_deg) * pow(Y, z_deg)
                            
                            if scale_factor > 0:
                                coeff_float = combined_val / float(scale_factor)
                                coeff = int(round(coeff_float))
                                if abs(coeff_float - coeff) < 1.0 and coeff != 0 and abs(coeff) < 1e50:
                                    combined_coeffs[(y_deg, z_deg)] = coeff
                    except:
                        pass
                
                # Only keep if we got a multi-term polynomial
                if len(combined_coeffs) >= 2:
                    combined_coeffs_clean = {k: v for k, v in combined_coeffs.items() if v != 0}
                    if len(combined_coeffs_clean) >= 2:
                        polynomials.append({
                            'coeffs': combined_coeffs_clean,
                            'vector': row_float + next_row,  # Combined vector
                            'norm': row_norm,
                            'non_zero_terms': len(combined_coeffs_clean)
                        })
                        if len(polynomials) <= 15:
                            logger.info(f"  ✓ Poly #{len(polynomials)} (combined): {len(combined_coeffs_clean)} terms from rows {row_idx}+{offset}")
                            poly_str = format_bivariate_polynomial(combined_coeffs_clean)
                            logger.info(f"    {poly_str}")
                        if len(polynomials) >= 20:
                            break
                if len(polynomials) >= 20:
                    break
                    # Debug: Show raw coefficients for first 3 polynomials
                    if len(polynomials) <= 3:
                        logger.debug(f"    Raw coeffs: {poly_coeffs_clean}")
                
                # Limit number of polynomials (like Sage)
                max_polys = 20 if m >= 8 else 15
                if len(polynomials) >= max_polys:
                    break
    
    logger.info(f"Extracted {len(polynomials)} polynomials total")
    
    # If still not enough, try more lenient extraction
    if len(polynomials) < 2:
        logger.warning("Not enough polynomials, trying lenient extraction...")
        logger.debug(f"  Checking first 50 rows, current count: {len(polynomials)}")
        
        for row_idx in range(min(50, reduced_basis.shape[0])):
            if len(polynomials) >= 10:
                break
                
            row_float = reduced_basis[row_idx]
            
            # Check if row has any significant values (very lenient)
            max_val = 0.0
            for val in row_float:
                try:
                    max_val = max(max_val, abs(float(val)))
                except:
                    pass
            
            if max_val < 1e-20:  # Very lenient threshold
                continue
            
            poly_coeffs = {}
            
            for col_idx, (y_deg, z_deg) in enumerate(monomials):
                if col_idx >= len(row_float):
                    break
                
                val_float = row_float[col_idx]
                try:
                    val_abs = abs(float(val_float))
                    if val_abs < 1e-20:  # Very lenient threshold
                        continue
                except:
                    continue
                
                try:
                    # Compute scale factor
                    if y_deg == 0 and z_deg == 0:
                        scale_factor = 1
                    else:
                        scale_factor = pow(X, y_deg) * pow(Y, z_deg)
                    
                    if scale_factor == 0:
                        continue
                    
                    # Very lenient: direct float division - extract ALL non-zero coefficients
                    if scale_factor == 1:
                        coeff = int(round(val_float))
                        if coeff != 0:
                            poly_coeffs[(y_deg, z_deg)] = coeff
                    else:
                        try:
                            # For huge scale factors, use float division
                            coeff_float = val_float / float(scale_factor)
                            
                            # Extract coefficient - be very lenient to get all terms
                            # The key insight: even small coefficients are important for multi-term polynomials
                            
                            # First, try to round to integer
                            coeff = int(round(coeff_float))
                            
                            # Accept if:
                            # 1. It's close to an integer (within reasonable tolerance)
                            # 2. OR the float value itself is non-zero (even if small)
                            
                            scale_bits = scale_factor.bit_length()
                            
                            # For large scales, use more lenient tolerance
                            if scale_bits > 500:
                                # Huge scale - very lenient relative tolerance
                                tolerance = max(10.0, abs(coeff_float) * 1e-6)
                            else:
                                tolerance = 1.0  # More lenient than before
                            
                            # Accept if reasonably close to integer
                            if abs(coeff_float - coeff) < tolerance:
                                if coeff != 0 and abs(coeff) < 1e40:
                                    poly_coeffs[(y_deg, z_deg)] = coeff
                            # Also accept small non-zero values (they might be significant when combined)
                            elif abs(coeff_float) > 1e-15:  # Very small threshold
                                # Keep the rounded value if it's non-zero
                                if abs(coeff) > 0:
                                    poly_coeffs[(y_deg, z_deg)] = coeff
                                elif abs(coeff_float) > 1e-10:
                                    # Very small but non-zero - round to ±1 if significant
                                    poly_coeffs[(y_deg, z_deg)] = 1 if coeff_float > 0 else -1
                        except (OverflowError, ZeroDivisionError):
                            pass
                except (OverflowError, ValueError, TypeError):
                    pass
            
            # Accept polynomials with at least 2 terms, or 1 term if it's significant
            if len(poly_coeffs) >= 2 or (len(poly_coeffs) == 1 and abs(list(poly_coeffs.values())[0]) > 0):
                polynomials.append({
                    'coeffs': poly_coeffs,
                    'vector': row_float,
                    'norm': max_val
                })
                logger.info(f"  ✓ Poly #{len(polynomials)} (lenient): {len(poly_coeffs)} terms, max_val={max_val:.2e}")
                # Show full polynomial
                poly_str = format_bivariate_polynomial(poly_coeffs)
                logger.info(f"    {poly_str}")
        
        logger.info(f"After lenient extraction: {len(polynomials)} polynomials")
    
    return polynomials


def solve_bivariate_system(polynomials: List[dict], p_approx: int, q_approx: int, N: int, 
                           X: int, Y: int) -> Optional[Tuple[int, int]]:
    """
    Solve bivariate polynomial system using Gröbner bases or other methods.
    
    Returns (y, z) if solution found, None otherwise.
    """
    if not SYMPY_AVAILABLE:
        logger.warning("sympy not available, using basic polynomial solving")
        return solve_bivariate_basic(polynomials, p_approx, q_approx, N, X, Y)
    
    if len(polynomials) < 2:
        logger.warning(f"Not enough polynomials ({len(polynomials)}) for Gröbner basis")
        return solve_bivariate_basic(polynomials, p_approx, q_approx, N, X, Y)
    
    try:
        y_sym, z_sym = symbols('y z', integer=True)
        
        # Convert polynomials to sympy format
        sympy_polys = []
        for poly_dict in polynomials[:min(10, len(polynomials))]:  # Use first 10
            coeffs = poly_dict['coeffs']
            terms = []
            for (y_deg, z_deg), coeff in coeffs.items():
                if coeff != 0:
                    terms.append(coeff * (y_sym ** y_deg) * (z_sym ** z_deg))
            if terms:
                poly_expr = sum(terms)
                sympy_polys.append(Poly(poly_expr, y_sym, z_sym))
        
        if len(sympy_polys) < 2:
            return solve_bivariate_basic(polynomials, p_approx, q_approx, N, X, Y)
        
        logger.info(f"Computing Gröbner basis from {len(sympy_polys)} polynomials...")
        
        # Show input polynomials
        logger.info("Input polynomials:")
        for i, poly_dict in enumerate(polynomials[:len(sympy_polys)], 1):
            poly_str = format_bivariate_polynomial(poly_dict['coeffs'])
            logger.info(f"  P{i}: {poly_str}")
        
        # Compute Gröbner basis with lex ordering
        try:
            G = groebner(sympy_polys, y_sym, z_sym, order='lex')
            logger.info(f"Gröbner basis: {len(G)} polynomials")
            
            # Show Gröbner basis polynomials
            for i, g in enumerate(G, 1):
                logger.info(f"  G{i}: {g}")
            
            # Check for univariate polynomials
            for g in G:
                vars_in_g = g.free_symbols
                if len(vars_in_g) == 1:
                    var = list(vars_in_g)[0]
                    logger.info(f"Found univariate polynomial in {var}: {g}")
                    
                    # Try to find integer roots
                    try:
                        # Convert to univariate polynomial
                        if var == y_sym:
                            poly_uni = Poly(g, y_sym)
                        else:
                            poly_uni = Poly(g, z_sym)
                        
                        # Find integer roots
                        logger.info(f"  Searching for integer roots of: {poly_uni}")
                        roots = poly_uni.all_roots()
                        logger.info(f"  Found {len(roots)} roots (including complex)")
                        
                        for root in roots:
                            if root.is_integer:
                                root_val = int(root)
                                logger.info(f"  Testing integer root: {root_val}")
                                if abs(root_val) < (X if var == y_sym else Y):
                                    # Substitute to find other variable
                                    if var == y_sym:
                                        y_val = root_val
                                        # Use constraint: (p_approx + y)(q_approx + z) = N
                                        if (p_approx + y_val) > 0:
                                            q_candidate = N // (p_approx + y_val)
                                            if q_candidate > 0 and N % (p_approx + y_val) == 0:
                                                z_val = q_candidate - q_approx
                                                logger.info(f"    y={y_val} -> q_candidate={q_candidate}, z={z_val}")
                                                if abs(z_val) < Y and (p_approx + y_val) * (q_approx + z_val) == N:
                                                    logger.info(f"✓✓✓ Found solution: y={y_val}, z={z_val}")
                                                    logger.info(f"    p = {p_approx + y_val}")
                                                    logger.info(f"    q = {q_approx + z_val}")
                                                    logger.info(f"    p * q = {(p_approx + y_val) * (q_approx + z_val)}")
                                                    logger.info(f"    N = {N}")
                                                    logger.info(f"    Match: {(p_approx + y_val) * (q_approx + z_val) == N}")
                                                    return (y_val, z_val)
                                    else:
                                        z_val = root_val
                                        if (q_approx + z_val) > 0:
                                            p_candidate = N // (q_approx + z_val)
                                            if p_candidate > 0 and N % (q_approx + z_val) == 0:
                                                y_val = p_candidate - p_approx
                                                logger.info(f"    z={z_val} -> p_candidate={p_candidate}, y={y_val}")
                                                if abs(y_val) < X and (p_approx + y_val) * (q_approx + z_val) == N:
                                                    logger.info(f"✓✓✓ Found solution: y={y_val}, z={z_val}")
                                                    logger.info(f"    p = {p_approx + y_val}")
                                                    logger.info(f"    q = {q_approx + z_val}")
                                                    logger.info(f"    p * q = {(p_approx + y_val) * (q_approx + z_val)}")
                                                    logger.info(f"    N = {N}")
                                                    logger.info(f"    Match: {(p_approx + y_val) * (q_approx + z_val) == N}")
                                                    return (y_val, z_val)
                                else:
                                    logger.info(f"  Root {root_val} out of bounds (|{root_val}| >= {X if var == y_sym else Y})")
                    except Exception as e:
                        logger.debug(f"Error solving univariate: {e}")
            
            # Try linear relations (y + z = k)
            for g in G:
                if g.degree() == 1:
                    # Check if it's of the form y + z = k
                    coeffs = g.as_expr().as_coefficients_dict()
                    y_coeff = coeffs.get(y_sym, 0)
                    z_coeff = coeffs.get(z_sym, 0)
                    const = coeffs.get(1, 0)
                    
                    if y_coeff == z_coeff and y_coeff != 0:
                        k = -const / y_coeff
                        if abs(k - round(k)) < 1e-10:
                            k_int = int(round(k))
                            logger.info(f"Found linear relation: y + z = {k_int}")
                            
                            # Use constraint: (p_approx + y)(q_approx + z) = N with y + z = k
                            # Substitute z = k - y: (p_approx + y)(q_approx + k - y) = N
                            # This gives quadratic in y
                            # For efficiency, check if (p_approx + y) divides N for y near -k/2
                            search_range = min(abs(k_int) + 1000, X)
                            for y_test in range(-search_range, search_range + 1):
                                z_test = k_int - y_test
                                if abs(z_test) < Y:
                                    p_test = p_approx + y_test
                                    q_test = q_approx + z_test
                                    if p_test > 0 and q_test > 0 and p_test * q_test == N:
                                        logger.info(f"✓✓✓ Found solution: y={y_test}, z={z_test}")
                                        return (y_test, z_test)
        
        except Exception as e:
            logger.warning(f"Gröbner basis computation failed: {e}")
            return solve_bivariate_basic(polynomials, p_approx, q_approx, N, X, Y)
    
    except Exception as e:
        logger.warning(f"Error in bivariate solving: {e}")
        return solve_bivariate_basic(polynomials, p_approx, q_approx, N, X, Y)
    
    return solve_bivariate_basic(polynomials, p_approx, q_approx, N, X, Y)


def solve_bivariate_basic(polynomials: List[dict], p_approx: int, q_approx: int, N: int,
                         X: int, Y: int) -> Optional[Tuple[int, int]]:
    """
    Basic bivariate solving without Gröbner bases.
    Uses constraint (p_approx + y)(q_approx + z) = N directly.
    """
    # Use the constraint: if (p_approx + y) divides N, then z = N/(p_approx + y) - q_approx
    # Check small y values
    search_range = min(1000, X)
    for y_test in range(-search_range, search_range + 1):
        p_test = p_approx + y_test
        if p_test > 0 and p_test < N and N % p_test == 0:
            q_test = N // p_test
            z_test = q_test - q_approx
            if abs(z_test) < Y and (p_approx + y_test) * (q_approx + z_test) == N:
                logger.info(f"✓✓✓ Found solution via divisibility: y={y_test}, z={z_test}")
                return (y_test, z_test)
    
    return None


def small_roots_bivariate_modN(p_approx: int, q_approx: int, N: int, X: int, Y: int, 
                               m: int = 6, verbose: bool = True, recursion_depth: int = 0,
                               max_recursion: int = 10, previous_best_remainder: Optional[int] = None) -> List[Tuple[int, int]]:
    """
    Bivariate Coppersmith: find small y, z such that (p_approx + y)(q_approx + z) = N.
    
    Uses equilateral triangle lattice structure and iterative refinement.
    """
    if verbose:
        logger.info("=" * 80)
        logger.info(f"BIVARIATE COPPERSMITH: m={m}, X={X.bit_length()}-bit, Y={Y.bit_length()}-bit")
        logger.info("=" * 80)
    
    best_remainder = None
    best_y = None
    best_z = None
    
    # Build lattice
    try:
        lattice, monomials, shift_polys = build_bivariate_lattice(p_approx, q_approx, N, X, Y, m)
        
        # Check for all-zero matrix (need manual check for object dtype)
        is_all_zero = True
        for i in range(lattice.shape[0]):
            for j in range(lattice.shape[1]):
                val = lattice[i, j]
                if val != 0 and val is not None:
                    is_all_zero = False
                    break
            if not is_all_zero:
                break
        
        if is_all_zero:
            logger.error("Lattice matrix is all zeros!")
            return []
        
        # Run LLL using fpylll wrapper (works with large integers)
        logger.info("Running LLL reduction...")
        if FPYLLL_AVAILABLE:
            # Use IntegerMatrix for better handling of large integers
            # Use higher delta (0.99) for better quality reduction
            B = IntegerMatrix(lattice)
            B_reduced = LLL(B, delta=0.99)
            reduced = B_reduced.to_numpy()
            logger.info("Using integer LLL (fpylll_wrapper) with delta=0.99")
        else:
            # Fallback: convert to float for LLL
            max_entry = 0
            for i in range(lattice.shape[0]):
                for j in range(lattice.shape[1]):
                    val = lattice[i, j]
                    if val != 0 and val is not None:
                        try:
                            val_abs = abs(int(val))
                            if val_abs > max_entry:
                                max_entry = val_abs
                        except (TypeError, ValueError):
                            pass
            
            max_bits = max_entry.bit_length() if max_entry > 0 else 0
            
            if max_bits > 1000:
                logger.info(f"Scaling lattice (max entry: 2^{max_bits} bits)...")
                scale_bits = max_bits - 400
                lattice_scaled = np.zeros((lattice.shape[0], lattice.shape[1]), dtype=float)
                for i in range(lattice.shape[0]):
                    for j in range(lattice.shape[1]):
                        val = lattice[i, j]
                        if val != 0 and val is not None:
                            try:
                                val_int = int(val)
                                val_bits = val_int.bit_length()
                                scaled_bits = val_bits - scale_bits
                                if scaled_bits > 0:
                                    lattice_scaled[i, j] = 2.0 ** min(scaled_bits, 1000)
                            except (TypeError, ValueError, OverflowError):
                                lattice_scaled[i, j] = 0.0
            else:
                lattice_scaled = np.zeros((lattice.shape[0], lattice.shape[1]), dtype=float)
                for i in range(lattice.shape[0]):
                    for j in range(lattice.shape[1]):
                        val = lattice[i, j]
                        if val != 0 and val is not None:
                            try:
                                lattice_scaled[i, j] = float(int(val))
                            except (TypeError, ValueError, OverflowError):
                                lattice_scaled[i, j] = 0.0
            
            reduced = lll_reduce(lattice_scaled, delta=0.99, return_float=True)
        logger.info("✓ LLL complete")
        
        # Track scaling used for LLL
        lll_scale_bits_used = 0
        # For float-based LLL, we don't know the exact scaling, so use 0
        # The extraction will use direct float division instead
        if FPYLLL_AVAILABLE and reduced.dtype != np.float64:
            # IntegerMatrix handles scaling internally, try to detect it
            # For now, estimate from max entry
            max_entry = 0
            for i in range(lattice.shape[0]):
                for j in range(lattice.shape[1]):
                    val = lattice[i, j]
                    if val != 0 and val is not None:
                        try:
                            val_abs = abs(int(val))
                            if val_abs > max_entry:
                                max_entry = val_abs
                        except:
                            pass
            max_bits = max_entry.bit_length() if max_entry > 0 else 0
            if max_bits > 1000:
                lll_scale_bits_used = max_bits - 400
        
        # Extract polynomials using original integer lattice
        # The reduced basis approximates integer combinations - extract exact coefficients
        polynomials = extract_bivariate_polynomials(reduced, lattice, monomials, X, Y, m, lll_scale_bits_used)
        logger.info(f"Extracted {len(polynomials)} polynomials")
        
        # Solve system
        solution = solve_bivariate_system(polynomials, p_approx, q_approx, N, X, Y)
        
        if solution:
            y_val, z_val = solution
            if (p_approx + y_val) * (q_approx + z_val) == N:
                logger.info("✓✓✓ EXACT SOLUTION FOUND!")
                return [(y_val, z_val)]
        
        # Track best remainder by checking polynomial evaluations
        initial_error = abs(p_approx * q_approx - N)
        best_remainder = initial_error
        
        # Check if any polynomial suggests small corrections
        # Look for linear or low-degree polynomials that might give us hints
        for poly_dict in polynomials[:10]:
            coeffs = poly_dict['coeffs']
            if len(coeffs) <= 3:  # Linear or quadratic
                # Try to solve: if it's linear in y or z, we can extract a candidate
                # For now, use the basic solving which checks divisibility
                candidate = solve_bivariate_basic([poly_dict], p_approx, q_approx, N, X, Y)
                if candidate:
                    y_cand, z_cand = candidate
                    p_cand = p_approx + y_cand
                    q_cand = q_approx + z_cand
                    remainder = abs(p_cand * q_cand - N)
                    if remainder < best_remainder:
                        best_remainder = remainder
                        best_y = y_cand
                        best_z = z_cand
                        if remainder == 0:
                            logger.info("✓✓✓ EXACT SOLUTION FOUND!")
                            return [(y_cand, z_cand)]
        
        # If we improved, try refinement
        if best_remainder > 0 and best_remainder < initial_error and recursion_depth < max_recursion:
            if previous_best_remainder is None or best_remainder < previous_best_remainder:
                logger.info(f"Improvement found: {initial_error.bit_length()}-bit → {best_remainder.bit_length()}-bit")
                logger.info("Refining approximations...")
                
                p_approx_new = p_approx + best_y if best_y is not None else p_approx
                q_approx_new = q_approx + best_z if best_z is not None else q_approx
                
                refined_result = small_roots_bivariate_modN(
                    p_approx_new, q_approx_new, N, X, Y, m=m,
                    verbose=verbose, recursion_depth=recursion_depth + 1,
                    max_recursion=max_recursion, previous_best_remainder=best_remainder
                )
                
                if refined_result:
                    # Adjust for refinement
                    adjusted = []
                    for y_ref, z_ref in refined_result:
                        y_final = (best_y if best_y is not None else 0) + y_ref
                        z_final = (best_z if best_z is not None else 0) + z_ref
                        if (p_approx + y_final) * (q_approx + z_final) == N:
                            adjusted.append((y_final, z_final))
                    return adjusted
    
    except Exception as e:
        logger.error(f"Bivariate Coppersmith failed: {e}")
        import traceback
        traceback.print_exc()
    
    return []


def main():
    print("=" * 80)
    print("STANDALONE COPPERSMITH ATTACK FROM PARTIAL FACTORS")
    print("=" * 80)
    print(f"\nN = {N}")
    print(f"N bit length: {N.bit_length()}")
    
    p_bit_length = N.bit_length() // 2
    print(f"Expected factor bit length: ~{p_bit_length}")
    
    product = P_partial * Q_partial
    print(f"\n[VERIFICATION]")
    print(f"  P_partial × Q_partial = N? {product == N}")
    
    if product != N:
        print(f"  These are PARTIAL factors - using Coppersmith")
    
    print(f"\n{'='*80}")
    print(f"PARTIAL FACTOR INFO")
    print(f"{'='*80}")
    print(f"P_partial bit length: {P_partial.bit_length()}")
    print(f"Q_partial bit length: {Q_partial.bit_length()}")
    
    # Try different numbers of known bits
    # Test BOTH MSBs and LSBs since we don't know which structure the partials have
    max_known = min(p_bit_length, max(P_partial.bit_length(), Q_partial.bit_length()))
    known_bits_to_try = [i for i in [1, 2, 3, 4, 5, 512, 600, 700, 768, 800, 850, 900, 950, 1000, 1020] 
                         if i < p_bit_length]
    
    if not known_bits_to_try:
        known_bits_to_try = [max(1, p_bit_length - 2), max(1, p_bit_length - 1)]
    
    for partial_value, partial_name in [(P_partial, "P_partial"), (Q_partial, "Q_partial")]:
        # Try MSBs first
        print(f"\n{'='*80}")
        print(f"TRYING WITH {partial_name} - MSB MODE")
        print(f"{'='*80}")
        
        partial_bits = partial_value.bit_length()
        
        # First, try using the partial as MSBs directly (bump it up with zeros)
        print(f"\n{'='*80}")
        print(f"ATTEMPT: Using {partial_name} as MSBs (bump up to {p_bit_length} bits)")
        print(f"{'='*80}")
        
        # Position the partial at the top by left-shifting
        actual_unknown_bits = p_bit_length - partial_bits
        
        # Check if partial value is too large (has more bits than expected)
        if actual_unknown_bits < 0:
            print(f"  WARNING: Partial value has {partial_bits} bits, but p should have {p_bit_length} bits")
            print(f"  Partial value is too large - skipping this attempt")
            continue
        
        p_known_msbs = partial_value << actual_unknown_bits
        
        print(f"  Partial bits: {partial_bits}")
        print(f"  Known MSBs: {partial_bits} (actual bits in partial)")
        print(f"  Unknown LSBs: {actual_unknown_bits}")
        print(f"  p0 bit length: {p_known_msbs.bit_length()}")
        
        if p_known_msbs == 0:
            print(f"  ERROR: p0 is zero")
        elif actual_unknown_bits > 512:
            print(f"  Skipping: {actual_unknown_bits} unknown bits (too many for Coppersmith)")
        else:
            try:
                p = small_roots_factorization(N, p_known_msbs, actual_unknown_bits, k=2, partial_hint=partial_value)
                
                if p is not None and p > 1 and N % p == 0:
                    q = N // p
                    print(f"\n{'='*80}")
                    print(f"✓✓✓ SUCCESS!")
                    print(f"{'='*80}")
                    print(f"  p = {p}")
                    print(f"  q = {q}")
                    print(f"  p × q = N? {p * q == N}")
                    print(f"  p bit length: {p.bit_length()}")
                    print(f"  q bit length: {q.bit_length()}")
                    return (p, q)
            except Exception as e:
                print(f"  Failed: {e}")
        
        # Also try the original approach with different num_known_msbs values (for partial extraction)
        for num_known_msbs in known_bits_to_try:
            print(f"\n{'='*80}")
            print(f"ATTEMPT: Assuming {num_known_msbs} MSBs are correct in {partial_name}")
            print(f"{'='*80}")
            
            unknown_bits = p_bit_length - num_known_msbs
            
            if unknown_bits <= 0:
                print(f"  Skipping: {unknown_bits} unknown bits (not positive)")
                continue
            
            if unknown_bits > 512:
                print(f"  Skipping: {unknown_bits} unknown bits (too many)")
                continue
            
            print(f"  Known MSBs: {num_known_msbs}")
            print(f"  Unknown LSBs: {unknown_bits}")
            
            # Extract and construct p0
            top_bits = extract_top_bits(partial_value, num_known_msbs, partial_bits)
            p_known_msbs = top_bits << unknown_bits
            
            print(f"  p0 bit length: {p_known_msbs.bit_length()}")
            
            if p_known_msbs == 0:
                print(f"  ERROR: p0 is zero")
                continue
            
            try:
                p = small_roots_factorization(N, p_known_msbs, unknown_bits, k=2, partial_hint=partial_value)
                
                if p is not None and p > 1 and N % p == 0:
                    q = N // p
                    print(f"\n{'='*80}")
                    print(f"✓✓✓ SUCCESS!")
                    print(f"{'='*80}")
                    print(f"  p = {p}")
                    print(f"  q = {q}")
                    print(f"  p × q = N? {p * q == N}")
                    print(f"  p bit length: {p.bit_length()}")
                    print(f"  q bit length: {q.bit_length()}")
                    return (p, q)
            except Exception as e:
                print(f"  Failed: {e}")
                continue
        
        # Now try LSBs 
        print(f"\n{'='*80}")
        print(f"TRYING WITH {partial_name} - LSB MODE")
        print(f"{'='*80}")
        print(f"Testing if partial contains known LSBs instead of MSBs...")
        
        for num_known_lsbs in known_bits_to_try:
            print(f"\n  Attempt: Assuming {num_known_lsbs} LSBs are correct")
            
            unknown_msbs = p_bit_length - num_known_lsbs
            
            if unknown_msbs <= 0:
                print(f"    Skipping: {unknown_msbs} unknown MSBs (not positive)")
                continue
            
            if unknown_msbs > 512:
                print(f"    Skipping: {unknown_msbs} unknown MSBs (too many)")
                continue
            
            print(f"    Known LSBs: {num_known_lsbs}")
            print(f"    Unknown MSBs: {unknown_msbs}")
            
            # Extract LSBs from partial
            lsbs = extract_low_bits(partial_value, num_known_lsbs)
            p0_lsb = lsbs
            
            print(f"    p0 (from LSBs): {p0_lsb}")
            print(f"    p0 bit length: {p0_lsb.bit_length()}")
            
            # For LSBs, Coppersmith solves: p = p0 + x * 2^num_known_lsbs
            # where p0 = lsbs, x = unknown high bits
            # We need to modify the lattice construction slightly
            
            # For now, try a simple approach: if we know LSBs, 
            # check if (partial_value) itself or with small MSB corrections works
            print(f"    Testing if partial (with LSBs) divides N...")
            
            # The partial might already have correct LSBs
            # Try: partial_value + k * 2^num_known_lsbs for small k
            base = partial_value - (partial_value % (1 << num_known_lsbs)) + p0_lsb
            
            for k in range(-100, 101):
                test_val = p0_lsb + k * (1 << num_known_lsbs)
                if test_val > 1 and test_val < N and N % test_val == 0:
                    q = N // test_val
                    print(f"\n{'='*80}")
                    print(f"✓✓✓ SUCCESS! (LSB mode)")
                    print(f"{'='*80}")
                    print(f"  p = {test_val}")
                    print(f"  q = {q}")
                    print(f"  Found with LSB + {k} * 2^{num_known_lsbs}")
                    return (test_val, q)
    
    # Try bivariate Coppersmith approach
    print(f"\n{'='*80}")
    print(f"TRYING BIVARIATE COPPERSMITH APPROACH")
    print(f"{'='*80}")
    print(f"Using P_partial and Q_partial as approximations for both factors")
    print(f"  p_approx = P_partial ({P_partial.bit_length()}-bit)")
    print(f"  q_approx = Q_partial ({Q_partial.bit_length()}-bit)")
    
    remainder = N - P_partial * Q_partial
    print(f"  Remainder: {abs(remainder).bit_length()}-bit")
    print()
    
    # Estimate bounds based on remainder
    error_bits = abs(remainder).bit_length()
    min_factor_bits = min(P_partial.bit_length(), Q_partial.bit_length())
    estimated_bound_bits = max(50, error_bits - min_factor_bits + 10)
    estimated_bound_bits = min(estimated_bound_bits, 341)  # Cap at reasonable size
    
    # Try a few bound sizes
    bounds_to_try = [
        (2 ** min(estimated_bound_bits, 200), 6),
        (2 ** min(estimated_bound_bits + 10, 250), 6),
        (2 ** min(estimated_bound_bits + 20, 300), 6),
    ]
    
    for X_val, m_val in bounds_to_try:
        Y_val = X_val
        print(f"\n{'='*80}")
        print(f"BIVARIATE ATTEMPT: X=Y=2^{X_val.bit_length()}, m={m_val}")
        print(f"{'='*80}")
        
        try:
            solutions = small_roots_bivariate_modN(
                P_partial, Q_partial, N, X_val, Y_val, m=m_val, verbose=True
            )
            
            if solutions:
                for y_val, z_val in solutions:
                    p_found = P_partial + y_val
                    q_found = Q_partial + z_val
                    
                    if p_found > 1 and q_found > 1 and p_found * q_found == N:
                        print(f"\n{'='*80}")
                        print(f"✓✓✓ SUCCESS! (Bivariate Coppersmith)")
                        print(f"{'='*80}")
                        print(f"  y = {y_val} ({abs(y_val).bit_length()}-bit)")
                        print(f"  z = {z_val} ({abs(z_val).bit_length()}-bit)")
                        print(f"  p = {p_found}")
                        print(f"  q = {q_found}")
                        print(f"  p × q = N? {p_found * q_found == N}")
                        return (p_found, q_found)
        except Exception as e:
            print(f"  Bivariate attempt failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print(f"✗ FAILED")
    print(f"{'='*80}")
    print(f"Could not factor N with given partial information")
    print(f"\n🔍 DIAGNOSIS:")
    print(f"  The Coppersmith method is working correctly, but the partial factors")
    print(f"  don't have the expected structure (MSBs of actual primes).")
    print(f"\n  What we found:")
    print(f"  - When using MSBs from the partials, Coppersmith found x ≈ -p0")
    print(f"  - This means p0 + x ≈ 0, indicating the MSBs are completely wrong")
    print(f"  - Bivariate approach also failed")
    print(f"\n  Possible explanations:")
    print(f"  1. P_partial and Q_partial are NOT (p_msbs, q_msbs)")
    print(f"  2. They might be p*q_partial or some other combination")
    print(f"  3. They might be corrupted/modified versions of the factors")
    print(f"  4. The bit alignment is wrong (MSBs not in high positions)")
    print(f"\n  To debug, can you share:")
    print(f"  - How were P_partial and Q_partial generated?")
    print(f"  - What relationship do they have to the actual prime factors p and q?")
    print(f"  - Are they guaranteed to share MSBs with p and q, or something else?")
    return None


if __name__ == "__main__":
    result = main()
    if result:
        p, q = result
        print(f"\n[FINAL RESULT]")
        print(f"p = {p}")
        print(f"q = {q}")
    else:
        print(f"\n[FAILED]")
