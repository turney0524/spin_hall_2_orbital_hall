#!/usr/bin/env python3
"""Teaching plot: distinguish orbital weights from Lz and Sz band projections.

Imports the first illustrative model without changing reproduce_fig4.py or
its original Fig. 4 outputs.
"""
from __future__ import annotations

import argparse
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
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from reproduce_fig4 import LZ,Parameters,hamiltonian,high_symmetry_path


def projections(p,points_per_segment):
    k,x,ticks=high_symmetry_path(points_per_segment)
    energy=[];weight=[];lz=[];sz=[]
    for kx,ky in k:
        e,u=np.linalg.eigh(hamiltonian(kx,ky,p))
        energy.append(e)
        weight.append(abs(u[0,:])**2)
        lz.append(np.real(np.diag(u.conj().T@LZ@u)))
        sz.append(np.full(3,p.spin_z))
        assert np.allclose(np.sum(abs(u)**2,axis=0),1,atol=1e-12)
    return dict(k=k,distance=x,ticks=ticks,energy=np.asarray(energy),
                z2_weight=np.asarray(weight),Lz=np.asarray(lz),Sz=np.asarray(sz))


def colored_lines(ax,x,y,color,norm,cmap):
    for band in range(2):
        xy=np.column_stack((x,y[:,band]))
        segments=np.stack((xy[:-1],xy[1:]),axis=1)
        coll=LineCollection(segments,cmap=cmap,norm=norm,linewidth=2)
        coll.set_array((color[:-1,band]+color[1:,band])/2)
        ax.add_collection(coll)
    ax.autoscale_view()
    return coll


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,default=Path('tb_results'))
    ap.add_argument('--points-per-segment',type=int,default=140)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    states=[('Before inversion',Parameters()),
            ('After one-valley inversion',replace(Parameters(),eps_e=.816))]
    data=[projections(p,args.points_per_segment) for _,p in states]
    fig,axes=plt.subplots(2,3,figsize=(13,6),constrained_layout=True,sharex='row')
    specs=[('z2_weight',Normalize(0,1),'viridis','$d_{z^2}$ weight'),
           ('Lz',Normalize(-2,2),'coolwarm','$L_z/\\hbar$'),
           ('Sz',Normalize(-.5,.5),'coolwarm','$S_z/\\hbar$')]
    for row,((label,_),d) in enumerate(zip(states,data)):
        x=d['distance'];ticks=d['ticks']
        ik=np.argmin(abs(x-ticks[1]))
        ef=(d['energy'][ik,0]+d['energy'][ik,1])/2
        for col,(key,norm,cmap,title) in enumerate(specs):
            ax=axes[row,col]
            coll=colored_lines(ax,x,d['energy']-ef,d[key],norm,cmap)
            ax.set_xlim(x[0],x[-1]);ax.set_ylim(-.65,1.15)
            ax.axhline(0,color='.6',ls='--',lw=.7)
            for t in ticks[1:-1]:ax.axvline(t,color='.7',ls='--',lw=.6)
            ax.set_xticks(ticks,[r'$M$',r'$K$',r'$\Gamma$',r"$K'$",r"$M'$"])
            ax.set_title(label+' · '+title,fontsize=10)
            if col==0:ax.set_ylabel('Energy - $E_F$ (eV)')
            fig.colorbar(coll,ax=ax,shrink=.78,aspect=18)
    fig.suptitle('Orbital weight and angular-momentum projections are different quantities',fontsize=12)
    fig.savefig(args.output/'orbital_spin_projected_bands.png',dpi=220)
    plt.close(fig)
    np.savez_compressed(args.output/'projected_bands.npz',
                        **{state+'_'+key:val for state,d in zip(('before','after'),data)
                           for key,val in d.items()})
    print('Wrote',(args.output/'orbital_spin_projected_bands.png').resolve())


if __name__=='__main__':main()
