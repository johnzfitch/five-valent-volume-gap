"""Closed-form tail bounds (THEOREM.md section 3) for the j1 = 3/2 box quantities F0, F1, W, as plain float functions.
These are the formulas proved in THEOREM.md; tail_certify.py evaluates them rigorously at the box edge and
tail_validate.py checks them numerically against the box quantities on the overlap.
L = 1 + ln n (>= H_n for the lattices used), S(x) = sum_{T>=1} 1/(T(T+x))."""
import math
Z2 = math.pi ** 2 / 6

def L(n): return 1 + math.log(n)
def S(x):
    """upper bound of sum_{T>=1} 1/(T(T+x)), x >= 0:  pi^2/6 at x = 0, else 1/(1+x) + ln(1+x)/x."""
    return Z2 if x == 0 else min(Z2, 1 / (1 + x) + math.log(1 + x) / x)

# ---- R2- (a >= 1)
def R2m(n, a=None):
    F0 = 32 / (3 * n)
    F1 = 12 / n + (32 / 3) * L(n) / (n * (n + 2))
    W = 8 * (2.5 + math.log(n)) * L(n) + 400 * (2.5 + math.log(n)) / (n - 10)
    return F0, F1, W

# ---- R2+ (a >= 1/2)
def R2p(n, a=None):
    F0 = 16 / n
    F1 = (32 / 3) / n + (32 / 3) * L(n) / (n * (n + 2))
    W = 16 * L(n) ** 2 + (2 / 3) * L(n)
    return F0, F1, W

# ---- R1 (a >= 3/2)
def R1(n, a):
    F0 = 4.22 / n
    F1 = (16 / 3) * S(2 * a - 3) / n
    W = a * (2.67 * L(n) ** 2 + 12.6 * L(n) + 9.3)
    return F0, F1, W

# ---- R4 (W = 0)
def R4(n, a=None):
    F0 = (16 / 3) * Z2 / n + (64 / 3) * L(n) / (n * (n + 3))
    F1 = 3.52 / n + (32 / 3) * L(n) / (n * (n + 2))
    return F0, F1, 0.0

# ---- R4c (W = 0)
def R4c(n, a=None):
    F0 = (16 / 3) * Z2 / n + (32 / 3) * L(n) / (n * (n + 2))
    F1 = 16 * (Z2 - 1) / n + (32 / 3) * L(n) / (n * (n + 3))
    return F0, F1, 0.0

# ---- R1c
def R1c(n, a=None):
    F0 = (16 / 3) * (1 + math.log(2 * n)) * (3 + 2 * math.log(n)) / n ** 2
    F1 = 16 * L(n) * (2 + 2 * math.log(n)) / (n * (n + 1))
    W = (n * n / 75) * F0
    return F0, F1, W

# ---- Tm, Tp
def Tm(n, a=None):
    F0 = (32 / 3) * L(n) ** 2 / n ** 2
    F1 = 16 * (Z2 - 1) / n + (32 / 3) * L(n) / (n * (n + 1))
    W = 6 * L(n) ** 2
    return F0, F1, W

def Tp(n, a=None):
    F0 = (64 / 3) * L(n) ** 2 / n ** 2
    F1 = 16 * (Z2 - 1) / n + (32 / 3) * L(n) / (n * (n + 2))
    W = 6 * L(n) ** 2
    return F0, F1, W

BOUNDS = {'R2-': R2m, 'R2+': R2p, 'R1': R1, 'R4': R4, 'R4c': R4c, 'R1c': R1c, 'Tm': Tm, 'Tp': Tp}
