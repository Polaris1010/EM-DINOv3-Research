"""Original-grid scientific comparison; slices selected solely by GT foreground count."""
from pathlib import Path
import json,numpy as np,nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
BASE=Path(__file__).resolve().parent
cases=['CMCC_crop113','CMCC_crop110','CMCT_jrc_mus-liver_crop145']
fig,axes=plt.subplots(3,6,figsize=(17,9),constrained_layout=True)
columns=['Original image','Ground truth','EM finetuned','MitoNet native','MitoNet pregrid','Gram 200k + old head']
for i,case in enumerate(cases):
 raw=nib.load(BASE/'cache'/(case+'_raw.nii'),mmap=True);gt=nib.load(BASE/'cache'/(case+'_gt.nii'),mmap=True)
 counts=[np.count_nonzero(np.asarray(gt.dataobj[:,:,z])) for z in range(gt.shape[2])];z=int(np.argmax(counts))
 image=np.asarray(raw.dataobj[:,:,z]).T/255.;true=np.asarray(gt.dataobj[:,:,z]).T>0
 paths=[None,None,BASE/'jobs/em_native_recheck'/(case+'.npy'),BASE/'jobs/mitonet_full_v2'/(case+'.npy'),BASE/'mitonet_pregrid_cpu'/(case+'.npy'),BASE/'jobs/gram200k_swap_full'/(case+'.npy')]
 for j,ax in enumerate(axes[i]):
  ax.imshow(image,cmap='gray',vmin=0,vmax=1)
  if j==1:
   ax.contour(true,levels=[.5],colors=['#45e879'],linewidths=.7)
  elif j>=2:
   if paths[j].exists():
    p=np.load(paths[j],mmap_mode='r')[z]>0
    if p.any():ax.contour(p,levels=[.5],colors=['#ffae42'],linewidths=.7)
    ax.contour(true,levels=[.5],colors=['#45e879'],linewidths=.5)
   else:ax.text(.5,.5,'same grid as native',transform=ax.transAxes,ha='center',color='white')
  ax.set_xticks([]);ax.set_yticks([])
  if i==0:ax.set_title(columns[j],fontsize=11)
  if j==0:ax.set_ylabel(case+'\nz='+str(z),fontsize=9)
fig.suptitle('Green: ground truth; orange: prediction. Representative slices only; use full-volume metrics for ranking.\nGram with an old head is a compatibility diagnostic, not a matched trained-head comparison.',fontsize=12)
fig.savefig(BASE/'baseline_overlays.png',dpi=150);plt.close(fig)
print(BASE/'baseline_overlays.png')
