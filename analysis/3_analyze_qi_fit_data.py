# -*- coding: utf-8 -*-
"""


@author: Patrick Hildebrandt, 20.07.2020
"""

#import sys, configparser
import sys
import numpy as np
import pylab as plt

material='MoS2'

PATH='//nap33.rz-berlin.mpg.de/hildebrandt/Masterarbeit/'+material+'/'
qxfile='//nap33.rz-berlin.mpg.de/hildebrandt/Masterarbeit/'+material+'/qx_fit_data.txt'
qyfile='//nap33.rz-berlin.mpg.de/hildebrandt/Masterarbeit/'+material+'/qy_fit_data.txt'
CRfile='//nap33.rz-berlin.mpg.de/hildebrandt/Masterarbeit/'+material+'/cen_res_data.txt'

def give_change(qipar):
    for k in range(0,qipar.shape[1]):
        qipar[:,k]=(qipar[:,k]/qipar[0,k]-1)*100
    return qipar
        
qxpar=np.loadtxt(qxfile,dtype=float,skiprows=1)
qypar=np.loadtxt(qyfile,dtype=float,skiprows=1)
cen_res=np.loadtxt(CRfile,dtype=float,skiprows=1)

tot_res=np.sqrt(cen_res[:,2]**2+cen_res[:,3]**2)

dqxpar=give_change(qxpar)
dqypar=give_change(qypar)

with open(PATH+'center.npy', 'wb') as f:
     np.save(f, np.array([cen_res[0,0], cen_res[0,1]]))

################## Show parameter evolution #####################
plt.rcParams['font.size'] = 12
#Plot qx
ax=plt.figure(num=None, figsize=(6, 5), dpi=300, facecolor='w', edgecolor='k')
labstr=['a','b','c','d',r'$\theta$']
for k in range(0,qxpar.shape[1]):
    plt.plot(np.arange(1,len(qxpar)+1,1),dqxpar[:,k],label=labstr[k])
plt.xlabel('Image number [#]',fontsize=12)
plt.ylabel('Change of parameter [%]',fontsize=12)
plt.dist = 12
ax.legend(bbox_to_anchor=(0.9, 0.5))
plt.savefig("//nap33.rz-berlin.mpg.de/hildebrandt/Masterarbeit/MoS2/parameters_evo_qx.png", format="png")
plt.show()

#Plot qy parameters
ax=plt.figure(num=None, figsize=(6, 5), dpi=300, facecolor='w', edgecolor='k')
labstr=['a','b','c','d',r'$\theta$']
for k in range(0,qypar.shape[1]):
    plt.plot(np.arange(1,len(qypar)+1,1),dqypar[:,k],label=labstr[k])
plt.xlabel('Image number [#]',fontsize=12)
plt.ylabel('Change of parameter [%]',fontsize=12)
plt.dist = 14
ax.legend(bbox_to_anchor=(0.9, 0.5))
plt.savefig("//nap33.rz-berlin.mpg.de/hildebrandt/Masterarbeit/MoS2/parameters_evo-qy.png", format="png")
plt.show()

#Plot residual evolution
fs=4
fig,ax=plt.subplots(constrained_layout=True,num=None, figsize=(2, 1), dpi=300, facecolor='w', edgecolor='k')
labstr=['mean res $q_x(x,y)$','mean res $q_y(x,y)$','total residual']
for k in range(2,4):
    plt.plot(np.arange(1,len(cen_res)+1,1),cen_res[:,k],label=labstr[k-2],linewidth=0.5)
plt.plot(np.arange(1,len(cen_res)+1,1),tot_res,label=labstr[2],linewidth=0.5)
plt.xlabel('Image number [#]',fontsize=fs)
plt.ylabel('Absolute residual [$\AA^{-1}$]',fontsize=fs)
plt.dist = 14
def invA2px(x):
    return x*90
def px2inA(x):
    return x / 90
ax.tick_params(labelsize=fs)
t=ax.secondary_yaxis('right', functions=(invA2px, px2inA))
t.yaxis.set_tick_params(labelsize=fs)
t.set_ylabel('px',fontsize=fs)
ax.legend(bbox_to_anchor=(0.7, 0.7),fontsize=fs-1)
plt.savefig("//nap33.rz-berlin.mpg.de/hildebrandt/Masterarbeit/MoS2/residuals.png", format="png")
plt.show()