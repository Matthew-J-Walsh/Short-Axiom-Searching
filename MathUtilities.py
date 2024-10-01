from Globals import *

def binomial(n: int, k: int) -> int:
    """Exact binomial coefficient.

    Parameters
    ----------
    n : int
    k : int

    Returns
    -------
    int
        Integer binomial coefficient
    """    
    return int(scipy.special.comb(n, k, exact=True))
    
@functools.cache
def catalan(n: int) -> int:
    """The nth Catalan number. https://oeis.org/A000108

    Parameters
    ----------
    n : int
        Index

    Returns
    -------
    int
        The nth Catalan number
    """
    return binomial(2*n, n) - binomial(2*n, n+1)

@functools.cache
def bells(n: int) -> int:
    """The nth Bell number. https://oeis.org/A000110

    Parameters
    ----------
    n : int
        Index

    Returns
    -------
    int
        The nth Bell number
    """    
    if n == 0:
        return 1
    v = 0
    for k in range(0, n):
        v += binomial(n-1, k) * bells(k)
    return v

def degenerate_constant_combinations(n: int) -> int:
    """Number of degenerate constant and variable combinations for n slots

    Parameters
    ----------
    n : int
        Number of slots

    Returns
    -------
    int
        Number of degenerate constant and variable combinations
    """    
    return sum(binomial(n, i) * bells(i) for i in range(0, n+1))
    
def nondegenerate_constant_combinations(n: int) -> int:
    """Number of non-degenerate constant and variable combinations for n slots

    Parameters
    ----------
    n : int
        Number of slots

    Returns
    -------
    int
        Number of non-degenerate constant and variable combinations
    """    
    return sum(binomial(n, i) * bells(i) for i in range(1, n))
    
    
    
    
    
    
    
    
    
    
    