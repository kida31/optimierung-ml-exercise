# %%
import numpy as np


def diagonalize_qr(mat, it=10, verbose=False):
    a = mat
    u = np.eye(mat.shape[0])
    for n in range(it):
        if verbose:
            print(f"==========\n#{n=}")
            print(f'a_{n}=\n{a}')
        
        q, r = np.linalg.qr(a)
        if verbose:
            print(f'q_{n}=\n{q}\nr_{n}=\n{r}')
        
        a = r @ q # a_n+1
        if verbose:
            print(f'a_{n+1}=\n{a}\n')
        
        u = u @ q
    return a, u


def solveUpper(R, b):
    """ Solve Rx = b via back-substitution
    R is upper-triangular.
    """
    n = R.shape[0]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        """
        By hand we usually solve (for some row i):
        
            a1*x1 + a2*x2 + a3*x3 + ... = b     | move a2*x2 + a3*x3 + ... to right
        =>  a1*x1 = b - (a2*x2 + a3*x3 + ...)   | divide by a1    
        =>  x1 = (b - (a2*x2 + a3*x3 + ...))
        
        a1 is R[i, 1] (i-th row, 1st column) (for diagonal 1 is generally i as well)
        """
        var1 = np.dot(R[i], x)
        var2 = np.dot(R[i, i:], x[i:])
        var3 = sum(a*b for a, b in zip(R[i, i:], x[i:]))
        assert var1 == var2
        assert var2 == var3
        
        x[i] = (b[i] - var1) / R[i, i]
    return x


def myQRSolver(A, b):
    q, r = np.linalg.qr(A)
    rhs = q.T @ b
    return solveUpper(r, rhs)

