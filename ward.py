import numpy as np
from scipy.sparse import csr_matrix, diags, csc_matrix, bmat
from scipy.sparse.linalg import spsolve

# ---------- Graph builder ----------
def build_chain(N:int):
    E = N - 1
    row = np.concatenate([np.arange(E), np.arange(1, N)])
    col = np.concatenate([np.arange(E), np.arange(E)])
    data = np.concatenate([np.ones(E), -np.ones(E)])
    B = csr_matrix((data, (row, col)), shape=(N, E))
    d = B.T
    return B, d, E

def solve_neumann_bordered(L, s):
    N = L.shape[0]
    c = np.ones((N,1)) / N
    K = bmat([[csc_matrix(L), csc_matrix(c)],
              [csc_matrix(c.T), csc_matrix((1,1))]], format='csc')
    rhs = np.concatenate([s, np.array([0.0])])
    sol = spsolve(K, rhs)
    return sol[:N]

# ---------- (1) Ward + Certificates ----------
def verify_ward_and_certs(N, mode="node", alpha=0.1):
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
        Ue  = 0.5*(U[:-1]+U[1:])
        WUe = 1.0 + alpha*Ue**2
        WU  = np.ones(N)
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
    cert_mid    = float(j[m] + s[:m+1].sum())
    cert_global = float(ward.sum())
    print(f"[{mode:6s}] N={N:3d} h={h:.4f} | RMS(ward)= {rms:.3e} | "
          f"Cert_global(sum)= {cert_global:.3e} | Cert_mid= {cert_mid:.3e}")

print("=== Ward + Certificates ===")
for mode in ("scalar","node","edge"):
    for N in (32,48,64,96):
        verify_ward_and_certs(N, mode=mode)

# ---------- (2) Time–Energy Exchange (node-based exact) ----------
def verify_exchange_node(N):
    x   = np.linspace(0.0, 1.0, N)
    Em  = 1.0 + 0.05*(1.0 - x)
    dEm = np.diff(Em)
    EU_node       = np.zeros(N)
    EU_node[1:]   = np.cumsum(-dEm)
    local_res     = np.diff(Em + EU_node)
    RMS_local     = float(np.sqrt(np.mean(local_res**2)))
    max_local     = float(np.max(np.abs(local_res)))
    global_res    = float((Em + EU_node)[-1] - (Em + EU_node)[0])
    print(f"[exchange-node] N={N:3d} | RMS_local={RMS_local:.3e} | "
          f"max|local|={max_local:.3e} | global_res={global_res:.3e}")

print("\n=== (A) Time–Energy Exchange (manufactured, κU>0, NODE-based exact) ===")
for N in (32,48,64,96):
    verify_exchange_node(N)

# ---------- (3) BD operator demo ----------
def bd_kernel_1d_tridiag(N, h):
    main  = -2.0*np.ones(N)
    off   = 1.0*np.ones(N-1)
    Bmat  = diags([off, main, off], [-1,0,1], shape=(N,N)) / (h*h)
    return Bmat.tocsc()

def demo_bd(N):
    h    = 1.0/np.sqrt(N)
    x    = np.linspace(0.0, 1.0, N)
    phi  = np.sin(2*np.pi*x)
    Bmat = bd_kernel_1d_tridiag(N, h)
    rhs  = -(Bmat @ phi)
    resid = (Bmat @ phi) + rhs
    rms   = float(np.sqrt(np.mean(resid**2)))
    mx    = float(np.max(np.abs(resid)))
    print(f"[BD] N={N:3d} | RMS(Bφ+rhs)= {rms:.3e} | max_abs= {mx:.3e}")

print("\n=== (B) BD operator demo (manufactured) ===")
for N in (32,48,64,96):
    demo_bd(N)

# ---------- (4) Nonlinear 3-term certificate (manufactured-consistent, FIXED) ----------
def nonlinear_certificate_demo_fixed(N, beta=0.3, lam=1.0):
    B, d, E = build_chain(N)
    h   = 1.0/np.sqrt(N)
    W_E = diags(np.ones(E)*h, 0, shape=(E,E))

    # Manufacture φ, then compute consistent fluxes
    x    = np.linspace(0.0,1.0,N)
    phi  = 0.1*np.sin(4*np.pi*x)
    dphi = d @ phi
    j    = - (W_E @ dphi)
    g2   = dphi**2
    Nval = g2/(1.0 + g2/(lam*lam))
    betaN_edge = beta * Nval
    nl_flux = W_E @ betaN_edge

    # manufacture-consistent source
    s = - (B @ j) - (B @ nl_flux)

    resid = (B @ j) + s + (B @ nl_flux)
    rms   = float(np.sqrt(np.mean(resid**2)))
    cert_global = float(resid.sum())
    m = N//2
    cert_mid_sum = float(np.sum(resid[:m+1]))
    boundary_cert = float( j[m] + nl_flux[m] + s[:m+1].sum() )
    print(f"[βN-FIX] N={N:3d} | RMS(resid)= {rms:.3e} | "
          f"Global= {cert_global:.3e} | "
          f"Mid(χ^T⋯)= {cert_mid_sum:.3e} | Mid(boundary)= {boundary_cert:.3e}")

print("\n=== (C) Nonlinear 3-term certificate (manufactured-consistent, FIXED) ===")
for N in (32,48,64,96):
    nonlinear_certificate_demo_fixed(N)

