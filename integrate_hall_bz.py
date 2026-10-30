#!/usr/bin/env python3
"""Brillouin-zone Kubo integration for the FIRST illustrative three-band model.

Outputs dimensionless geometric Hall weights versus chemical potential.  See
spin_orbital_hall_tutorial.md for the current-operator and unit conventions.
"""
from __future__ import annotations

import argparse
import csv
import os
import tempfile
from dataclasses import replace
from pathlib import Path

cache=Path(tempfile.gettempdir())/'valley-ohe-mpl-cache'
cache.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(cache))
os.environ.setdefault('XDG_CACHE_HOME',str(cache))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from reproduce_fig4 import LZ,Parameters,hamiltonian,operators


def band_curvatures(kx,ky,p,dk):
    """Energy and BC/SBC/OBC of EACH of the three bands at one k point."""
    energy,u=np.linalg.eigh(hamiltonian(kx,ky,p))
    vx,vy=operators(kx,ky,p,dk)
    currents=[vx,2*p.spin_z*vx,LZ@vx+vx@LZ]
    y=u.conj().T@vy@u
    omega=np.zeros((3,3),float)  # row: BC/SBC/OBC; column: band
    for kind,j in enumerate(currents):
        x=u.conj().T@j@u
        for n in range(3):
            for m in range(3):
                if n!=m:
                    omega[kind,n]+=2*np.imag(x[n,m]*y[m,n])/(energy[m]-energy[n])**2
    return energy,omega


def sample_bz(p,mesh,dk):
    # a1=(1,0), a2=(1/2,sqrt(3)/2); b_i·a_j=2πδ_ij.
    b1=np.array([2*np.pi,-2*np.pi/np.sqrt(3)])
    b2=np.array([0.,4*np.pi/np.sqrt(3)])
    area=abs(np.linalg.det(np.column_stack((b1,b2))))
    en=np.zeros((mesh*mesh,3));om=np.zeros((mesh*mesh,3,3))
    row=0
    for i in range(mesh):
        u=(i+.5)/mesh-.5
        for j in range(mesh):
            v=(j+.5)/mesh-.5
            k=u*b1+v*b2
            en[row],om[row]=band_curvatures(*k,p,dk)
            row+=1
    return en,om,area


def occupation(energy,mu,temperature):
    if temperature<=0:return (energy<mu).astype(float)
    kb_ev=8.617333262145e-5
    z=np.clip((energy-mu)/(kb_ev*temperature),-100,100)
    return 1/(1+np.exp(z))


def hall_weights(en,om,area,mus,temperature):
    """C_X(mu)=(1/2π)∫BZ d²k Σ_n f_n Omega^X_n.

    C_BC is the Chern number when mu lies in a true global gap at T=0.
    C_SBC/C_OBC use the code's dimensionless angular-momentum operators.
    """
    factor=area/(2*np.pi*len(en))
    answer=np.zeros((len(mus),3))
    for i,mu in enumerate(mus):
        f=occupation(en,mu,temperature)
        answer[i]=factor*np.einsum('kn,kxn->x',f,om)
    return answer


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mesh',type=int,default=121,help='midpoint grid points per reciprocal direction')
    ap.add_argument('--dk',type=float,default=1e-5)
    ap.add_argument('--temperature',type=float,default=0.,help='Kelvin; 0 uses a step function')
    ap.add_argument('--mu-min',type=float,default=-.1)
    ap.add_argument('--mu-max',type=float,default=.35)
    ap.add_argument('--mu-points',type=int,default=181)
    ap.add_argument('--output',type=Path,default=Path('tb_results/hall_bz'))
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    mus=np.linspace(args.mu_min,args.mu_max,args.mu_points)
    states=[('before',Parameters()),('after',replace(Parameters(),eps_e=.816))]
    fig,axes=plt.subplots(2,1,figsize=(8,7),sharex=True,constrained_layout=True)
    summary=[]
    for ax,(name,p) in zip(axes,states):
        en,om,area=sample_bz(p,args.mesh,args.dk)
        weights=hall_weights(en,om,area,mus,args.temperature)
        # The midpoint mesh can miss the exact valley extrema, especially for
        # the after state where E_0(K)=E_1(K').  Include both valleys explicitly.
        valley_en=np.array([np.linalg.eigvalsh(hamiltonian(kx,0.,p))
                            for kx in (4*np.pi/3,-4*np.pi/3)])
        valence_top=float(max(en[:,0].max(),valley_en[:,0].max()))
        conduction_bottom=float(min(en[:,1].min(),valley_en[:,1].min()))
        if abs(conduction_bottom-valence_top)<1e-10:
            conduction_bottom=valence_top
        global_gap=conduction_bottom-valence_top
        for col,(label,color) in enumerate([('BC / charge','#222222'),
                                             ('SBC / spin','#d62728'),
                                             ('OBC / orbital','#2455d6')]):
            ax.plot(mus,weights[:,col],label=label,color=color,lw=1.6)
        if global_gap>0:
            ax.axvspan(valence_top,conduction_bottom,color='green',alpha=.10,
                       label='global band gap')
        ax.set_title(f'{name}: indirect gap ≈ {global_gap:.4f} eV')
        ax.set_ylabel(r'$C_X=(1/2\pi)\int\Omega^X d^2k$')
        ax.axhline(0,color='.6',lw=.6)
        ax.legend(frameon=False,ncol=2,fontsize=8)
        np.savez_compressed(args.output/f'{name}_bz.npz',energy=en,curvature=om,
                            mu=mus,hall_weight=weights,bz_area=area,
                            valence_top=valence_top,
                            conduction_bottom=conduction_bottom)
        summary.append((name,valence_top,conduction_bottom,global_gap))
        with (args.output/f'{name}_hall_vs_mu.csv').open('w',newline='') as f:
            writer=csv.writer(f)
            writer.writerow(['mu_eV','C_BC','C_SBC','C_OBC'])
            writer.writerows(np.column_stack((mus,weights)))
    axes[-1].set_xlabel('chemical potential $\\mu$ (eV)')
    fig.suptitle(f'BZ-integrated geometric Hall weights; {args.mesh}×{args.mesh} mesh, '
                 f'T={args.temperature:g} K')
    fig.savefig(args.output/'hall_vs_mu.png',dpi=220)
    plt.close(fig)
    for row in summary:print(*row)
    print('Wrote',args.output.resolve())


if __name__=='__main__':main()
