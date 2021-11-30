import scipy
import scipy.special

def binomial(n, k):
    """
    Binomial coefficient.
    """
    return int(scipy.special.comb(n, k, exact=True))
    
def catalan(n):
    """
    Catalan numbers. A000108
    """
    return binomial(2*n, n) - binomial(2*n, n+1)

bells_memo = [1]
def bells(n):
    """
    Bells numbers. A000110
    """
    global bells_memo
    if len(bells_memo)>n:
        return bells_memo[n]
    v = 0
    for k in range(0, n):
        v += binomial(n-1, k) * bells(k)
    bells_memo += [v]
    return v
    
    
    
    
    
    
    
    
    
    
    
    
    