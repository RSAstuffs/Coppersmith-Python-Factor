# Coppersmith Attack for RSA Factorization

A standalone implementation of Coppersmith's method for factoring RSA moduli when partial information about the prime factors is known (e.g., known MSBs or LSBs).

## Overview

This project implements Coppersmith's attack algorithm for integer factorization, which is particularly useful when you have partial knowledge of one or both prime factors of an RSA modulus. The attack uses lattice basis reduction (LLL algorithm) to efficiently recover the complete factors from partial information.

## Features

- **Standalone Implementation**: No external dependencies except NumPy
- **Multiple Attack Modes**: Supports both MSB (Most Significant Bits) and LSB (Least Significant Bits) partial factor scenarios
- **Built-in LLL Reduction**: Custom implementation of the Lenstra-Lenstra-Lovász lattice basis reduction algorithm
- **Flexible Parameter Tuning**: Configurable lattice parameters for different attack scenarios
- **Comprehensive Logging**: Detailed progress output for debugging and analysis
- **Automatic Mode Detection**: Attempts to detect whether partials contain MSBs or LSBs

## Requirements

- Python 3.6+
- NumPy

## Installation

```bash
pip install numpy
```

## Usage

### Basic Usage

1. Edit the values at the top of `copper.py`:
   - `N`: The RSA modulus to factor
   - `P_partial`: Partial information about one prime factor
   - `Q_partial`: Partial information about the other prime factor

2. Run the script:
```bash
python3 copper.py
```

### Programmatic Usage

```python
from copper import small_roots_factorization

# N: RSA modulus
# p0: Known MSBs of factor p (shifted to high position)
# lgX: Log2 of bound on x (number of unknown bits)
# k: Lattice parameter (default 2)

N = X

# Example: 1000 known MSBs, 512 unknown bits
p0 = X << 512

p = small_roots_factorization(N, p0, lgX=512, k=2)
if p:
    q = N // p
    print(f"Successfully factored: p = {p}, q = {q}")
```

## How It Works

### Coppersmith's Method

Coppersmith's method leverages lattice basis reduction to find small roots of polynomial equations modulo N. When partial information about a factor is known:

1. **Polynomial Construction**: The known partial information (e.g., MSBs) is used to construct a polynomial `f(x) = p0 + x`, where `p0` contains the known bits and `x` represents the unknown bits.

2. **Lattice Building**: A lattice is constructed from shifted polynomial multiples: `x^j * N^(m-i) * f(x)^i` for various values of `i` and `j`.

3. **LLL Reduction**: The Lenstra-Lenstra-Lovász (LLL) algorithm reduces the lattice basis to find short vectors that correspond to small roots.

4. **Root Extraction**: The reduced lattice vectors are analyzed to extract polynomial relationships, which are then solved to recover the unknown bits.

### LLL Algorithm

The implementation includes a custom LLL reduction algorithm that:
- Performs Gram-Schmidt orthogonalization
- Applies size reduction conditions
- Enforces the Lovász condition
- Handles numerical stability for large integers

## Technical Details

### Complexity

The success of the attack depends on:
- The ratio of known bits to unknown bits
- The size of the modulus N
- The lattice parameters (m, k)

Typically, if you know approximately `n/2` bits of an `n`-bit factor, the attack becomes feasible.

### Limitations

- **Bit Requirement**: Requires a significant portion of the factor to be known (typically >50% of the bits)
- **Computational Cost**: Lattice reduction can be computationally expensive for very large moduli
- **Numerical Stability**: Large integer arithmetic requires careful scaling to avoid overflow/underflow

### Supported Scenarios

1. **Known MSBs**: When the most significant bits of a factor are known
2. **Known LSBs**: When the least significant bits of a factor are known
3. **Partial Factor Hints**: When you have approximate values of the factors

## Algorithm Components

### Core Functions

- `small_roots_factorization()`: Main Coppersmith attack implementation
- `lll_reduce()`: LLL lattice basis reduction
- `gram_schmidt()`: Gram-Schmidt orthogonalization
- `solve_polynomial_mod_N()`: Solves polynomial equations using modular arithmetic
- `extract_top_bits()` / `extract_low_bits()`: Bit manipulation utilities

### Main Workflow

The `main()` function orchestrates the attack:
1. Tests if partial factors are close to actual factors (brute force check)
2. Tries MSB mode for both P_partial and Q_partial
3. Tries LSB mode for both partial factors
4. Attempts various numbers of known bits
5. Returns the complete factors if found

## Example Output

```
================================================================================
STANDALONE COPPERSMITH ATTACK FROM PARTIAL FACTORS
================================================================================

N = X...
N bit length: 2048
Expected factor bit length: ~1024

[VERIFICATION]
  P_partial × Q_partial = N? False
  These are PARTIAL factors - using Coppersmith

================================================================================
TRYING WITH P_partial - MSB MODE
================================================================================

[INFO]: Coppersmith: N=2048b, p0=1024b, lgX=512
[INFO]: Building lattice: m=2, t=3
[INFO]: Lattice: (3, 3), 6 non-zero entries
[INFO]: Running LLL...
[INFO]: ✓ LLL complete
[INFO]: Attempting to extract roots from lattice structure...
...

✓✓✓ SUCCESS!
  p = X
  q = X
  p × q = N? True
```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## References

- Coppersmith, D. (1996). "Finding a Small Root of a Univariate Modular Equation"
- Howgrave-Graham, N. (1997). "Finding Small Roots of Univariate Modular Equations Revisited"
- Lenstra, A. K., Lenstra, H. W., & Lovász, L. (1982). "Factoring polynomials with rational coefficients"

## License

This project is provided as-is for educational and research purposes.

## Disclaimer

This tool is intended for educational purposes, authorized security testing, and research. Only use this tool on systems you own or have explicit permission to test.

