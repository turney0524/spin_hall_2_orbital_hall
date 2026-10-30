#!/usr/bin/env python3
"""Qualitative Fig. 4 reconstruction using a three-d-orbital triangular TB model.

Basis: (d_z2, d_xy, d_x2-y2).  All quantities are dimensionless apart from
energies (eV) and momentum (1/a).  This is an explicitly spin-polarized block.
This first illustrative build predates receipt of the PRL Supplemental Material.
The parameters below are illustrative and are NOT the authors' fitted values.
"""
from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

_cache = Path(tempfile.gettempdir()) / "valley-ohe-mpl-cache"
_cache.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


@dataclass(frozen=True)
class Parameters:
    # eV.  Hopping ratios are the MoS2 GGA nearest-neighbor values from
    # Liu et al., PRB 88, 085433 (2013), Table II, scaled by 0.3.
    # Onsite energies and SOC are illustrative values adjusted here.
    eps_z2: float = 0.0
    eps_e: float = 0.685
    t0: float = -0.184 * 0.3
    t1: float = 0.401 * 0.3
    t2: float = 0.507 * 0.3
    t11: float = 0.218 * 0.3
    t12: float = 0.338 * 0.3
    t22: float = 0.057 * 0.3
    soc_z: float = 0.025
    spin_z: float = 0.5  # units of hbar


# L_z / hbar in the real-harmonic basis above.
LZ = np.array([[0, 0, 0], [0, 0, 2j], [0, -2j, 0]], complex)
I3 = np.eye(3, dtype=complex)


def hamiltonian(kx: float, ky: float, p: Parameters) -> np.ndarray:
    """Nearest-neighbor C3-symmetric 3-orbital Hamiltonian, plus projected SOC."""
    a, b = kx / 2, np.sqrt(3) * ky / 2
    c, s = np.cos, np.sin
    h0 = p.eps_z2 + 2 * p.t0 * (c(2*a) + 2*c(a)*c(b))
    h1 = 2j*p.t1*(s(2*a) + s(a)*c(b)) - 2*np.sqrt(3)*p.t2*s(a)*s(b)
    h2 = 2*p.t2*(c(2*a) - c(a)*c(b)) + 2j*np.sqrt(3)*p.t1*c(a)*s(b)
    h11 = p.eps_e + (p.t11+3*p.t22)*c(a)*c(b) + 2*p.t11*c(2*a)
    h22 = p.eps_e + (3*p.t11+p.t22)*c(a)*c(b) + 2*p.t22*c(2*a)
    h12 = (np.sqrt(3)*(p.t22-p.t11)*s(a)*s(b)
           + 4j*p.t12*s(a)*(c(a)-c(b)))
    h = np.array([[h0, h1, h2],
                  [h1.conjugate(), h11, h12],
                  [h2.conjugate(), h12.conjugate(), h22]], complex)
    return h + p.soc_z * LZ


def operators(kx: float, ky: float, p: Parameters, dk: float = 1e-5):
    """Velocity matrices dH/dk in units eV*a/hbar (common factor omitted)."""
    vx = (hamiltonian(kx+dk, ky, p)-hamiltonian(kx-dk, ky, p))/(2*dk)
    vy = (hamiltonian(kx, ky+dk, p)-hamiltonian(kx, ky-dk, p))/(2*dk)
    return vx, vy


def kubo_at(kx: float, ky: float, p: Parameters, dk: float = 1e-5):
    """Occupied-lowest-band BC, SBC, OBC, and SOBC from interband Kubo sums.

    Convention: Omega^X_xy = 2 Im sum_{m>0} [(J^X_x)_0m (v_y)_m0]/gap^2,
    J^X_x={X,v_x}.  BC instead uses v_x.  This fixes the signs shown here;
    reversing x/y reverses every curvature together.

    SOBC is explicitly DEFINED HERE using X=S_z L_z. It is not a separate
    curve in the paper's Fig. 4; for a single fixed spin it equals S_z*OBC.
    """
    h = hamiltonian(kx, ky, p)
    e, u = np.linalg.eigh(h)
    vx, vy = operators(kx, ky, p, dk)
    sz = p.spin_z * I3
    currents = (vx, sz@vx+vx@sz, LZ@vx+vx@LZ,
                (sz@LZ)@vx+vx@(sz@LZ))
    y = u.conj().T @ vy @ u
    ans = []
    for j in currents:
        x = u.conj().T @ j @ u
        ans.append(sum(2*np.imag(x[0, m]*y[m, 0])/(e[m]-e[0])**2
                       for m in (1, 2)))
    l = np.real(np.diag(u.conj().T @ LZ @ u))
    s = np.full(3, p.spin_z)
    return e, l, s, np.asarray(ans)


def high_symmetry_path(points_per_segment: int):
    # Lattice constant a=1.  M--K--Gamma--K'--M' as in the PRL figure.
    m = np.array([np.pi, np.pi/np.sqrt(3)])
    k = np.array([4*np.pi/3, 0])
    g = np.zeros(2)
    kp = -k
    mp = -m
    vertices = [m, k, g, kp, mp]
    chunks = [np.linspace(vertices[i], vertices[i+1], points_per_segment,
                          endpoint=False) for i in range(4)]
    path = np.vstack(chunks + [mp[None, :]])
    distances = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    ticks = distances[np.array([0, 1, 2, 3, 4])*points_per_segment]
    return path, distances, ticks


def evaluate_path(p: Parameters, nseg: int, dk: float):
    path, x, ticks = high_symmetry_path(nseg)
    rows = [kubo_at(*k, p, dk) for k in path]
    return path, x, ticks, np.array([r[0] for r in rows]), \
        np.array([r[1] for r in rows]), np.array([r[2] for r in rows]), \
        np.array([r[3] for r in rows])


def setup_axis(ax, ticks, labels=True):
    ax.set_xticks(ticks, [r"$M$", r"$K$", r"$\Gamma$", r"$K'$", r"$M'$"] if labels else [])
    for t in ticks[1:-1]:
        ax.axvline(t, color="0.65", ls="--", lw=0.8, zorder=0)
    ax.axhline(0, color="0.6", ls="--", lw=0.7, zorder=0)


def colored_bands(ax, x, energies, projection, limit, cmap="coolwarm"):
    for band in range(2):  # the two bands bounding the occupied gap
        pts = np.column_stack([x, energies[:, band]])
        seg = np.stack([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(seg, cmap=cmap, norm=Normalize(-limit, limit), lw=1.8)
        lc.set_array((projection[:-1, band]+projection[1:, band])/2)
        ax.add_collection(lc)
    ax.autoscale_view()
    return lc


def make_figures(outdir: Path, base, inv, pbase, pinv):
    _, x, ticks, eb, lb, sb, cb = base
    _, _, _, ei, li, si, ci = inv
    # Fig. 4-like comparison.  Curvatures are scaled only for legibility.
    plt.rcParams.update({"font.size": 9, "axes.spines.top": True,
                         "axes.spines.right": True, "savefig.dpi": 220})
    fig, ax = plt.subplots(2, 4, figsize=(15, 6), sharex="col", constrained_layout=True)
    ki = np.argmin(np.abs(x-ticks[1]))
    ef = (eb[ki,0]+eb[ki,1])/2
    colored_bands(ax[0,0], x, eb-ef, lb, 2)
    colored_bands(ax[1,0], x, eb-ef, sb, .5)
    ax[0,0].set_title("(a) Orbital $L_z/\\hbar$")
    ax[1,0].set_title("Spin $S_z/\\hbar$")
    for r in range(2):
        setup_axis(ax[r,0], ticks, r==1)
        ax[r,0].set_ylabel("Energy (eV)")
        ax[r,0].set_xlim(x[0], x[-1])
        ax[r,0].set_ylim(-.65, 1.15)
    # Boundary-bands only.  E_F halfway between valley gap edges.
    for col,(cur,title) in enumerate([(cb,"(b) Before inversion"),
                                      (ci,"(c) After one-valley inversion")],1):
        scale = max(np.max(np.abs(cur[:,0])), 1)
        ax[0,col].plot(x, cur[:,0]/scale, c="black", label="BC")
        ax[1,col].plot(x, cur[:,1]/scale, c="#d62728", label="SBC")
        ax[1,col].plot(x, cur[:,2]/scale, c="#2455d6", label="OBC")
        ax[0,col].set_title(title)
        for r in range(2):
            setup_axis(ax[r,col],ticks,r==1)
            ax[r,col].set_xlim(x[0],x[-1])
            ax[r,col].set_ylabel("Curvature / max|BC|")
            ax[r,col].legend(loc="best",frameon=False,fontsize=8)
    # Eq. (1) comparison: normalize the candidate prefactor at K for a fair
    # sign/shape comparison.  This does not turn it into an exact identity.
    lsum = lb[:,0]+lb[:,1]
    ssum = sb[:,0]+sb[:,1]
    for row,(idx,fac,name) in enumerate([(1,ssum,"SBC"),(2,lsum,"OBC")]):
        direct = cb[:,idx]
        approximate = fac*cb[:,0]
        ki = np.argmin(np.abs(x-ticks[1]))
        if abs(approximate[ki]) > 1e-10:
            approximate *= direct[ki]/approximate[ki]
        norm = max(np.max(np.abs(direct)), 1)
        ax[row,3].plot(x,direct/norm,c="#d62728",label="Kubo")
        ax[row,3].plot(x,approximate/norm,c="black",ls="--",label=r"$\langle X\rangle\,\Omega^{BC}$")
        setup_axis(ax[row,3],ticks,row==1)
        ax[row,3].set_xlim(x[0],x[-1]);ax[row,3].set_ylabel(name+" / max")
        ax[row,3].legend(loc="best",frameon=False,fontsize=8)
    ax[0,3].set_title("(d) Two-band factorization")
    fig.suptitle("Three-orbital qualitative reconstruction of PRL Fig. 4 (illustrative parameters)",fontsize=12)
    fig.savefig(outdir/"fig4_reconstruction.png")
    plt.close(fig)

    # Additional explicit SOBC output.  It is mathematically redundant for
    # this fixed-spin model but answers the user's requested quantity.
    fig, ax = plt.subplots(2,2,figsize=(10,6),constrained_layout=True)
    labels = ["BC", "SBC", "OBC", "SOBC = $S_z$ OBC"]
    colors = ["black", "#d62728", "#2455d6", "#7a3bb0"]
    for j,a in enumerate(ax.flat):
        scale=max(np.max(np.abs(np.r_[cb[:,j],ci[:,j]])),1)
        a.plot(x,cb[:,j]/scale,label="before",c=colors[j],lw=1.3)
        a.plot(x,ci[:,j]/scale,label="after",c=colors[j],lw=1.3,ls="--")
        setup_axis(a,ticks,True);a.set_xlim(x[0],x[-1]);a.set_title(labels[j]);a.legend(frameon=False)
        a.set_ylabel("curvature / common max")
    fig.suptitle("Berry-curvature and angular-momentum responses")
    fig.savefig(outdir/"bc_sbc_obc_sobc.png")
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,default=Path("tb_results"))
    ap.add_argument("--points-per-segment",type=int,default=140)
    ap.add_argument("--dk",type=float,default=1e-5)
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    p0=Parameters()
    p1=replace(p0,eps_e=0.816)
    base=evaluate_path(p0,args.points_per_segment,args.dk)
    inv=evaluate_path(p1,args.points_per_segment,args.dk)
    make_figures(args.output,base,inv,p0,p1)
    for label,p,data in [("before",p0,base),("after",p1,inv)]:
        path,x,ticks,e,l,s,c=data
        np.savez_compressed(args.output/f"{label}_path.npz",k=path,distance=x,
                            ticks=ticks,energy=e,Lz=l,Sz=s,
                            BC=c[:,0],SBC=c[:,1],OBC=c[:,2],SOBC=c[:,3])
        k=4*np.pi/3
        vals=[kubo_at(v,0,p)[3] for v in (k,-k)]
        print(f"{label}: K and K' [BC,SBC,OBC,SOBC] =")
        for v in vals:print(" ",np.round(v,4))
        assert np.allclose(c[:,1],2*p.spin_z*c[:,0],rtol=1e-8,atol=1e-8)
        assert np.allclose(c[:,3],p.spin_z*c[:,2],rtol=1e-8,atol=1e-8)
    print("Wrote",args.output.resolve())


if __name__=="__main__":
    main()
