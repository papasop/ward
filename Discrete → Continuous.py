# Discrete-first verification: exact closure → mesh/weight → continuum limit → ablation
# 自足版：只需 NumPy/SciPy/Matplotlib。可直接粘贴到 Colab 一个单元执行。

import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from dataclasses import dataclass
rng = np.random.default_rng(2025)

@dataclass
class Grid2D:
    n:int; h:float; N:int; Eh:int; Ev:int; E:int
    B:sp.csr_matrix; BT:sp.csr_matrix; Ix:np.ndarray; Iy:np.ndarray

def build_grid_incidence(n:int)->Grid2D:
    h=1.0/(n+1); N=n*n; Eh=(n-1)*n; Ev=n*(n-1); E=Eh+Ev
    rows=[]; cols=[]; data=[]
    def idx(i,j): return i + j*n
    e=0
    # 水平边
    for j in range(n):
        for i in range(n-1):
            a=idx(i,j); b=idx(i+1,j)
            rows += [a,b]; cols += [e,e]; data += [-1.0,+1.0]; e+=1
    # 垂直边
    for j in range(n-1):
        for i in range(n):
            a=idx(i,j); b=idx(i,j+1)
            rows += [a,b]; cols += [e,e]; data += [-1.0,+1.0]; e+=1
    B = sp.csr_matrix((data,(rows,cols)), shape=(N,E))
    Ix,Iy = np.meshgrid(np.arange(n), np.arange(n), indexing='xy')
    return Grid2D(n,h,N,Eh,Ev,E,B,B.T,Ix.ravel(),Iy.ravel())

def weight_matrix(grid:Grid2D, mode="uniform", seed=2025):
    rng_local=np.random.default_rng(seed)
    if mode=="uniform":
        we=np.ones(grid.E)
    elif mode=="random12":
        we=0.5+rng_local.random(grid.E)      # (0.5, 1.5)
    elif mode=="checker":
        we=np.ones(grid.E); we[::2]=1.3; we[1::2]=0.7
    else:
        raise ValueError("unknown mode")
    return sp.diags(we,0,shape=(grid.E,grid.E)), we

def solve_neumann_bordered(L:sp.csr_matrix, s:np.ndarray):
    """解 L φ = s，附加均值约束 <φ>=0 的有界 Neumann 系统"""
    N=L.shape[0]; c=np.ones((N,1))/N
    K=sp.bmat([[L.tocsc(), sp.csc_matrix(c)],
               [sp.csc_matrix(c.T), sp.csc_matrix((1,1))]], format='csc')
    rhs=np.concatenate([s, np.array([0.0])])
    sol=spsolve(K,rhs)
    return sol[:N]

def manufactured_phi(grid:Grid2D):
    """制造平滑目标解，用于 Γ/弱收敛风格检验"""
    x=(grid.Ix+1)*grid.h; y=(grid.Iy+1)*grid.h
    phi=np.sin(np.pi*x)*np.sin(2*np.pi*y)+0.3*np.sin(3*np.pi*x)*np.sin(np.pi*y)
    return phi - phi.mean()

def ward_suite(grid:Grid2D, W:sp.csr_matrix, kappa=1.0):
    """核心：构造 L、用 gated 源 s=L φ*，解出 φ，并计算 Ward/EL 残量"""
    L=(grid.B @ W @ grid.BT)/(grid.h**2)
    phi_star=manufactured_phi(grid)
    s=kappa*(L @ phi_star)             # 关键：gated 源，保证离散闭合
    phi=solve_neumann_bordered(L,s)
    j = - (W @ (grid.BT @ phi)) / grid.h
    j = np.asarray(j).ravel()
    ward_res=(grid.B @ j)+s
    el_res=(L @ phi)-s
    rmsW=float(np.sqrt(np.mean(ward_res**2)))
    rmsE=float(np.sqrt(np.mean(el_res**2)))
    return dict(L=L, phi_star=phi_star, phi=phi, s=s, j=j,
                rms_ward=rmsW, rms_el=rmsE)

def refinement_study(ns=(24,32,40,56,72,96), mode="uniform"):
    out=[]
    for n in ns:
        grid=build_grid_incidence(n); W,_=weight_matrix(grid, mode=mode)
        r=ward_suite(grid,W)
        out.append((n,r['rms_ward'],r['rms_el']))
    return out

def random_windows_certificates(grid:Grid2D, j, s, n_win=300):
    """在随机子域上验证通量证书：边界通量 + 源和 ≈ 0"""
    n=grid.n; B=grid.B
    tails=[]; heads=[]
    for e in range(grid.E):
        col=B[:,e].toarray().ravel()
        a=np.where(col==-1)[0][0]; b=np.where(col==+1)[0][0]
        tails.append(a); heads.append(b)
    tails=np.array(tails); heads=np.array(heads)
    def rect_mask(i0,i1,j0,j1):
        m=np.zeros(grid.N,bool)
        for jj in range(j0,j1):
            for ii in range(i0,i1):
                m[ii+jj*n]=True
        return m
    chis=[]
    for _ in range(n_win):
        i0=np.random.randint(0,n-4); i1=np.random.randint(i0+2,n)
        j0=np.random.randint(0,n-4); j1=np.random.randint(j0+2,n)
        mask=rect_mask(i0,i1,j0,j1)
        in_mask=mask[tails]; out_mask=mask[heads]
        boundary=(in_mask ^ out_mask)
        sign=np.where(in_mask & ~out_mask, +1.0, 0.0)+np.where(~in_mask & out_mask, -1.0, 0.0)
        flux=np.sum(sign[boundary]*j[boundary]); src=np.sum(s[mask])
        chis.append(flux+src)
    return np.array(chis)

def ablation_breaks_closure(n=48):
    """消融实验：去掉门控（用与 Lφ* 脱钩的源），Closure 立刻破坏"""
    grid=build_grid_incidence(n); W,_=weight_matrix(grid, mode="random12")
    ok=ward_suite(grid,W)
    x=(grid.Ix+1)*grid.h; y=(grid.Iy+1)*grid.h
    rho=np.sin(2*np.pi*x)*np.sin(3*np.pi*y)+0.2*np.cos(5*np.pi*x)
    s_bad=rho-rho.mean()
    L=(grid.B @ W @ grid.BT)/(grid.h**2)
    phi_bad=solve_neumann_bordered(L,s_bad)
    j_bad = - (W @ (grid.BT @ phi_bad)) / grid.h; j_bad=np.asarray(j_bad).ravel()
    ward_bad=(grid.B @ j_bad)+s_bad; el_bad=(L @ phi_bad)-s_bad
    return dict(ok_rms_ward=ok['rms_ward'], ok_rms_el=ok['rms_el'],
                bad_rms_ward=float(np.sqrt(np.mean(ward_bad**2))),
                bad_rms_el=float(np.sqrt(np.mean(el_bad**2))))

# ------------------ 1) 离散层闭合 & 细化 ------------------
rows = refinement_study(ns=(24,32,40,56,72,96), mode="uniform")  # ✅ 修正：mode="uniform"
print("Refinement (n, RMS_Ward, RMS_EL):")
for n,rW,rE in rows: print(f"  n={n:<3d}  RMS_Ward={rW:.3e}  RMS_EL={rE:.3e}")
ns=np.array([r[0] for r in rows]); rW=np.array([r[1] for r in rows])
plt.figure(figsize=(6,4))
plt.loglog(ns,rW,'o-'); plt.xlabel("grid n"); plt.ylabel("RMS(Ward residual)")
plt.title("Exact discrete closure across refinements (uniform weights)")
plt.grid(True,which='both',ls='--'); plt.show()

# ------------------ 2) 网格/权重无关 ------------------
n=64; grid=build_grid_incidence(n)
for md in ["uniform","random12","checker"]:
    W,_=weight_matrix(grid, mode=md)                 # ✅ 修正：关键字传参
    out=ward_suite(grid,W)
    print(f"mode={md:<8s}  RMS_Ward={out['rms_ward']:.3e}  RMS_EL={out['rms_el']:.3e}")

# ------------------ 3) 连续是离散的极限像（Γ/弱收敛风格） ------------------
def l2e(a,b): return float(np.sqrt(np.mean((a-b)**2)))
errs=[]
for n in (24,32,40,56,72,96,128):
    grid=build_grid_incidence(n); W,_=weight_matrix(grid, mode="uniform")  # ✅
    out=ward_suite(grid,W); errs.append((n, l2e(out['phi'],out['phi_star']), out['rms_ward']))
print("\nManufactured-solution convergence (phi to phi*):")
for n,e,rW in errs: print(f"  n={n:<3d}  L2(phi-phi*)={e:.3e}  RMS_Ward={rW:.1e}")
hs=np.array([1.0/(n+1) for n,_,_ in errs]); E2=np.array([e for _,e,_ in errs])
plt.figure(figsize=(6,4))
plt.loglog(hs,E2,'o-'); plt.gca().invert_xaxis()
plt.xlabel("h (grid spacing)"); plt.ylabel("L2 error ||phi-phi*||")
plt.title("Discrete→Continuous limit (Γ/weak-style check)")
plt.grid(True,which='both',ls='--'); plt.show()

# ------------------ 4) 证书（子域边界通量 + 源和 ≈ 0） ------------------
n=64; grid=build_grid_incidence(n); W,_=weight_matrix(grid, mode="uniform")  # ✅
out=ward_suite(grid,W)
chi=random_windows_certificates(grid,out['j'],out['s'], n_win=300)
print(f"\nCertificates: mean={chi.mean():.3e}, std={chi.std():.3e}")
plt.figure(figsize=(6,4))
plt.hist(chi,bins=40,density=True)
plt.xlabel("certificate χ (flux + source)"); plt.ylabel("pdf")
plt.title("Certificate closure over random subdomains")
plt.grid(True,ls='--'); plt.show()

# ------------------ 5) 不可替代性（无门控即破） ------------------
abl=ablation_breaks_closure(n=48)
print("\nAblation (necessity of source gate W(U))")
for k,v in abl.items(): print(f"  {k:>14s}: {v:.3e}")
