"""
5 - Inelastic intensities

This script uses parameters of q-space maps to bin intensities at a previously 
defined k-path coming from 'Quantum Espresso' DFPT calculations.

@author: Patrick Hildebrandt
"""
# %%
import sys, os
from dotenv import load_dotenv

load_dotenv()
ROOT_DIR = os.getenv("ROOT_DIR")
DATA_DIR = os.getenv("DATA_DIR")
sys.path.append(ROOT_DIR)

#%%
import tqdm
import math as m
import glob
import numpy as np
import skued
import scipy.io as spio
import helpers.fedutils as utils
import pylab as plt
from crystals import Crystal

PATH_CFG = os.path.join(DATA_DIR, "config_2020_19_07.cfg")

#%%
#Loading data from config file (parameters and pathes)
dict_path, dict_numerics = utils.read_cfg(PATH_CFG)
PATH_DATA = dict_path['path_data']
CIF_FILE = dict_path['cif']
PATH = dict_path['path']
partial = dict_numerics['partial']
if partial == 1:
    fn = glob.glob(PATH + '*.tif')[:dict_numerics['to_file']]
else:
    fn = glob.glob(PATH + '*.tif')
NO_FILES = len(fn)
path_mask = PATH + dict_path['path_mask']
PATH_FF =  dict_path['path_ff']
PATH_BKG = dict_path['path_bkg']
PATH_CEN = dict_path['path_center']
window_size_intensity = dict_numerics['window_size_intensity']

#%%
#Loads laser background (2048x2048)
LASER_BKG = np.array(skued.diffread(PATH + PATH_BKG), dtype = np.float64)

#Load mask (2048x2048)
mask_total = spio.loadmat(path_mask)['mask_total']

#Loads flatfield (2048x2048)
FFmat = spio.loadmat(PATH_FF)
FF = FFmat['FF']

#Loads center positions of the whole dataset
cen_res = np.loadtxt(PATH_DATA+'cen_res_data.txt',dtype = float, skiprows = 1)

#Load fit parameters for each pattern (legth of dataset x 5 (number of parameters)) 
qx_pars = np.loadtxt(PATH_DATA+'qx_fit_data.txt',dtype = float, skiprows = 1)
qy_pars = np.loadtxt(PATH_DATA+'qy_fit_data.txt',dtype = float, skiprows = 1)

#Load peak selection 
data_ps = np.loadtxt(PATH_DATA+'peak_selection.txt',dtype = float, skiprows = 1)
nop = data_ps.shape[0] #number of peaks

#Setup crystal by using 'crystals' crystal-class.
MoS2 = Crystal.from_cif(PATH_DATA+CIF_FILE)
#Get symmetry for 1BZ calculations
symmetry = MoS2.symmetry(symprec=0.01, angle_tolerance=-1.0)
rot_sym = int(symmetry['hm_symbol'][1])
ang_d = 360/rot_sym

#Get reciprocal lattice vectors from cif class in Angstrom
kVec=np.array(MoS2.reciprocal_vectors,dtype=float)

#Load k-path. !!! Materialscloud loads k-path coordinates in a different order
#than crystal class. Copy/ paste path vectors in txt and swap coordinates if needed.
kpath = np.loadtxt(PATH_DATA+'kpath_test.txt',dtype=str ,skiprows = 0)
if len(kpath.shape) == 1:
    kpoints = np.array(kpath[1:4],dtype=float).reshape(1,3)
else:
    kpoints = np.array(kpath[:,1:4],dtype=float)

#Get Bragg peak vectors
hkl = data_ps[:,2:5]
q, qvec_all = utils.get_qVectors(kVec,hkl)

##Loads initial peak positions
#!!! Modified by Patrick to load data from peak_selection tool!!!!!! 
PATH_PEAKS = dict_path['path_peaks']
peakpos_all = np.loadtxt(PATH_PEAKS,dtype = float, skiprows = 1)

#Give nearest neighbours (number of NoN = rot_sym). Calculates number and indices 
#of nearest neighbours of every Bragg peak. Additional gives peak positions, indices
#and scattering vectors of peaks with complete Brillouin zone. A BZ is complete 
#if the number of surrounding peaks is the same as the rotational symmetry.
nn_tol = 5 #tolerance 
NNs = np.empty([0,rot_sym],dtype=int)
qvect = qvec_all.transpose()
ind_BZs = []
for j in range(0,nop):
    diff = np.round(np.linalg.norm(qvect-qvect[j,:],axis=1),decimals=nn_tol)
    diff[j] = m.nan
    diff_min = np.nanmin(diff)
    self_ind = np.where(diff==diff_min)
    if len(self_ind[0])==rot_sym:
        NNs = np.vstack((NNs,self_ind[0]))
        ind_BZs.append(j)
ind_BZs = np.array(ind_BZs,dtype=int) #Array with indices of peaks with full BZ
peakpos = peakpos_all[ind_BZs,:] #Update peakpos
qvec = qvect[ind_BZs,:] #Update scattering vectors
print('Peak selection contains ',str(NNs.shape[0]),' complete Brillouin zones.')

sym_BZs = utils.get_rotsym_BZs(qvec,rot_sym,3)
    
def get_closest_pos(qx_map,qy_map,qvec):
    #Get distance of q-space to q-vec. q-vec ...3x1 or 1x3 array
    diff = np.sqrt(np.square(qx_map-qvec[0])+np.square(qy_map-qvec[1]))
    diff_min = np.nanmin(diff)
    temp = np.where(diff==diff_min) #where flips coordinates
    pos = [temp[1][0],temp[0][0]]
    pos = np.array(pos,dtype=float)
    return pos

def control_peakpos(qx_map,qy_map,qvec_all,image,data_ps):
    plt.figure()
    plt.imshow(image,vmin=0,vmax=6000)
    for j in range(0,nop):
        pos = get_closest_pos(qx_map,qy_map,qvec_all[:,j])
        plt.scatter(pos[1],pos[0]) #scatter needs fliped positions
    plt.show()
    
def vector_rotation_3D(ang_d,vec):
    ang = np.radians(ang_d) #Conversion to radians
    r = np.array([[np.cos(ang), -np.sin(ang),0],[np.sin(ang),  np.cos(ang),0],\
                  [0,0,1]],dtype=float)
    rot_vec= np.matmul(r,vec)
    return rot_vec
    

#Loop over all files
path_pos = np.zeros([NO_FILES,peakpos.shape[0],kpoints.shape[0],rot_sym,2],dtype=float) 
int_mean_path = np.zeros([NO_FILES,peakpos.shape[0],kpoints.shape[0],rot_sym],dtype=float)


for idx, fp in tqdm.tqdm(enumerate(fn[0:NO_FILES])):
    image = np.array(skued.diffread(fp), dtype = np.int64)
    
    #Apply mask
    image = image*mask_total
    #Substract background and flatfield
    image = utils.remove_bgk(image, LASER_BKG, FF)
    
    #Rotational averaging. DO NOT use nfold averaging for single BZs
    #avg_area = skued.nfold(area, mod = rot_sym, center = peakpos[j,:], mask=area_mask)

    #Get q-space map
    img_dim = np.arange(0,image.shape[0],1)
    x = img_dim-cen_res[idx,0]
    y = img_dim-cen_res[idx,1]
    
    xgrid, ygrid = np.meshgrid(x,y)
    qx_map=utils.qi_func((xgrid,ygrid), *qx_pars[idx,:])
    qy_map=utils.qi_func((xgrid,ygrid), *qy_pars[idx,:])
    
    new_peakpos = utils.refine_peakpos_arb_dim(peakpos, image, 0, 8)
    #peakpos_evolution.append(new_peakpos)
    peakpos = new_peakpos
    d = 30
    
    #NNs is defined for peakpos_all. The list peakpos is reduced to all peaks 
    #with full 1.BZ.
    for j in range(0,len(peakpos)):
        #Get NN position, qvec and hkl
        peakpos_NN = peakpos_all[NNs[j,:],:]
        qvec_NN = qvec_all[:,NNs[j,:]]
        hkl_NN = hkl[NNs[j,:],:]
        
        #Calculate positions of k-path. Instead of doing rotaional averaging 
        #by rotating the whole pattern, all symmetry equivalent intensities are
        #calculated and averaged later.
        
        for k in range(kpoints.shape[0]):
            for l in range(rot_sym):
                krot=vector_rotation_3D(ang_d*l,kpoints[k,:])
                qrot = qvec[j,:]+krot
                path_pos[idx,j,k,l,:]= get_closest_pos(qx_map,qy_map,qrot)
                int_mean_path[idx,j,k,l] = utils.sum_peak_pixels(image, path_pos[idx,j,k,l,:], window_size_intensity)

        #Display Bzs and path positions of the first image
        
        #Create area for checking/displayig results
        max_pos = np.array(np.nanmax(peakpos_NN,axis=0),dtype=int)
        min_pos = np.array(np.nanmin(peakpos_NN,axis=0),dtype=int)
        max_pos = max_pos+d
        min_pos = min_pos-d
        area = image[min_pos[0]:max_pos[0],min_pos[1]:max_pos[1]]
        area_mask = mask_total[min_pos[0]:max_pos[0],min_pos[1]:max_pos[1]]

        plt.figure()
        plt.imshow(area,vmin=0,vmax=6000)
        for k in range(0,kpoints.shape[0]):
        #Scatter needs flipped x,y positions
            for l in range(0,rot_sym):
                plt.scatter(path_pos[idx,j,k,l,1]-min_pos[1],path_pos[idx,j,k,l,0]-min_pos[0],s=15)
                plt.text(path_pos[idx,j,k,l,1]-min_pos[1],path_pos[idx,j,k,l,0]-min_pos[0],\
                         kpath[k,0]+'$_'+str(l)+'$',color='white')
        plt.show()
        
    if idx==0:
        #Show maps for control
        control_peakpos(qx_map,qy_map,qvec_all,image,data_ps)
        
        plt.figure()
        plt.imshow(qx_map)
        plt.title('qx(x,y)-map with (0,0) at center')
        plt.colorbar()
        plt.show()
        plt.figure()
        plt.imshow(qy_map)
        plt.title('qy(x,y)-map with (0,0) at center')
        plt.colorbar()
        plt.show()

#Saving data in an npz file
np.savez(PATH_DATA+'Inelastic_data/output_data_05.npz',int_in=int_mean_path,path_pos=path_pos)
np.savez(PATH_DATA+'Inelastic_data/peaks_symmetry_BZ_05.npz',qvec=qvec,qvec_all=qvect,sym_Bzs=sym_BZs)