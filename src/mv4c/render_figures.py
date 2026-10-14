"""Four-panel provenance illustrations from measured arrays, no model calls."""
import json
from pathlib import Path
import numpy as np,zarr,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit import R,BASE,M3,M4B,read,sha,load_original
def render():
 c=read(R/'MV4C_protocol.json');summary=read(R/'MV4C_summary.json');byid={r['region_id']:r for r in summary['regions']};out=R/'comparison_figures';out.mkdir(exist_ok=True);records=[]
 for row in c['regions']:
  q=byid[row['region_id']];cache=read(M3/'cache'/row['region_id']/'CACHE.json');group=zarr.open_group(q['zarr_path'],mode='r');gt=np.load(cache['paths']['label'],mmap_mode='r');prob=np.load(q['probability_path'],mmap_mode='r')
  raw=load_original(Path(q['original']['path']))[0] if q['original']['exists'] else None
  for z in row['figure_planes']:
   fig=plt.figure(figsize=(12,11));axes=[fig.add_axes(pos) for pos in [[.04,.54,.42,.36],[.54,.54,.42,.36],[.04,.06,.42,.36],[.54,.06,.36,.36]]]
   if raw is None:
    axes[0].set_facecolor('#eeeeee');axes[0].text(.5,.5,'Original TIFF unavailable\nPre-ingest intensity / orientation unverified\nNot replaced by Zarr-export TIFF',ha='center',va='center',transform=axes[0].transAxes);axes[0].set_xticks([]);axes[0].set_yticks([])
   else:
    axes[0].imshow(raw[z],cmap='gray_r' if q['original'].get('photometric')=='MINISWHITE' else 'gray',vmin=float(raw.min()),vmax=float(raw.max()),interpolation='nearest');axes[0].axis('off')
   axes[0].set_title('Pre-ingest author original' if raw is not None else 'Pre-ingest original: UNVERIFIED',fontsize=11)
   zi=np.asarray(group['img'][z]);axes[1].imshow(zi,cmap='gray',vmin=0,vmax=255,interpolation='nearest');axes[1].set_title('Current Zarr (fixed 0..255)',fontsize=11);axes[1].axis('off')
   axes[2].imshow(gt[z]>0,cmap='gray',vmin=0,vmax=1,interpolation='nearest');axes[2].set_title('Existing GT at same native Z / XY',fontsize=11);axes[2].axis('off')
   im=axes[3].imshow(prob[z],cmap='magma',vmin=0,vmax=1,interpolation='nearest');axes[3].set_title('Reused raw semantic probability',fontsize=11);axes[3].axis('off')
   fig.colorbar(im,cax=fig.add_axes([.925,.08,.016,.32]))
   fig.suptitle(row['region_id']+f' — native Z={z}\n'+q['normalization_recorded']+'; no inference / intensity or geometry adjustment',fontsize=12,y=.975)
   path=out/(row['region_id']+f'_z{z:04d}.png');fig.savefig(path,dpi=120,bbox_inches='tight',pad_inches=.2);plt.close(fig)
   records.append({'region_id':row['region_id'],'native_z':z,'path':str(path),'sha256':sha(path),'original_verified':raw is not None,'probability_reused':True,'display_ranges':{'raw':'linear native min/max only, TIFF photometric respected','zarr':[0,255],'probability':[0,1]}})
 (R/'MV4C_comparison_figures.json').write_text(json.dumps(records,indent=2)+'\n')
 print('FOUR_PANEL_FIGURES_DONE',len(records),flush=True)
if __name__=='__main__':render()
