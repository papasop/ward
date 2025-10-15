# Adjust certificates to the algebraically correct form with standard incidence B:
# - Global: Cert_global = sum(B j + s)  (since 1^T B = 0 for standard incidence)
# - Subdomain [0..m]: Cert_mid = chi^T (B j + s) where chi is 1 on 0..m, i.e. j[m] + sum(s[:m+1])
# This should drive both certificates to ~machine precision.

import numpy as np
from scipy.sparse import csr_matrix, diags, csc_matrix, bmat
from scipy.sparse.linalg import spsolve

def build_chain(N:int):
    E = N - 1
    row = np.concatenate([np.arange(E), np.arange(1, N)])
    col = np.concatenate([np.arange(E), np.arange(E)])
    data = np.concatenate([np.ones(E), -np.ones(E)])
    B = csr_matrix((data, (row, col)), shape=(N, E))
    return B, B.T, E

def solve_neumann_bordered(L, s):
    N = L.shape[0]
    c = np.ones((N,1)) / N
    K = bmat([[csc_matrix(L), csc_matrix(c)],
              [csc_matrix(c.T), csc_matrix((1,1))]], format='csc')
    rhs = np.concatenate([s, np.array([0.0])])
    sol = spsolve(K, rhs)
    return sol[:N]

def verify_one_final(N, mode="node", alpha=0.1):
    B, d, E = build_chain(N)
    h = 1.0/np.sqrt(N)
    W_E = diags(np.ones(E)*h, 0, shape=(E,E))
    L = B @ W_E @ B.T

    U = np.linspace(0.0, 1.0, N)
    if mode=="scalar":
        WU = np.ones(N)
    elif mode=="node":
        WU = 1.0 + alpha*U**2
    elif mode=="edge":
        Ue = 0.5*(U[:-1]+U[1:])
        WUe = 1.0 + alpha*Ue**2
        WU = np.ones(N)
        WU[:-1] += 0.5*(WUe-1.0)
        WU[1:]  += 0.5*(WUe-1.0)
    else:
        raise ValueError

    s = WU - WU.mean()
    phi = solve_neumann_bordered(L, s)
    j = - (W_E @ (d @ phi))
    ward = (B @ j) + s
    rms = float(np.sqrt(np.mean(ward**2)))

    m = N//2
    cert_mid = float((j[m]) + s[:m+1].sum())     # chi^T (B j + s)
    cert_global = float(ward.sum())              # 1^T (B j + s)

    return dict(h=1.0/np.sqrt(N), rms=rms, cert_global=cert_global, cert_mid=cert_mid)

def run_all_final(Ns=(32,48,64,96), modes=("scalar","node","edge")):
    for mode in modes:
        for N in Ns:
            out = verify_one_final(N, mode=mode)
            print(f"[{mode:6s}] N={N:3d} h={out['h']:.4f} | "
                  f"RMS(ward)={out['rms']:.3e} | "
                  f"Cert_global(sum)={out['cert_global']:.3e} | "
                  f"Cert_mid={out['cert_mid']:.3e}")

run_all_final()
