import numpy as np
from scipy import ndimage as nd
CONN=nd.generate_binary_structure(2,1)
def disk(r):
    x=np.arange(-r,r+1)
    return x[:,None]**2+x[None,:]**2<=r*r

def gt_maps(gt):
    positive=gt>0
    _,inv=np.unique(gt[positive],return_inverse=True)
    ids=np.zeros(gt.shape,np.int32)
    ids[positive]=inv+1
    area=np.bincount(ids.ravel())
    area[0]=0
    maps={}
    for r in (3,5):
        core=np.zeros_like(ids)
        se=disk(r)
        for i,box in enumerate(nd.find_objects(ids),1):
            good=nd.binary_erosion(ids[box]==i,structure=se,border_value=0)
            core[box][good]=i
        v=np.bincount(core.ravel(),minlength=len(area))
        v[0]=0
        maps[r]=(core,v)
    return ids,area,maps

def metrics(ground,pred):
    ids,area,maps=ground
    got=np.bincount(ids[pred],minlength=len(area))
    detected=(area>0)&(10*got>=area)
    detected[0]=False
    missed=(ids>0)&~detected[ids]
    closed=nd.binary_fill_holes(pred,structure=CONN)&~pred
    result=[]
    overlay=None
    for r,(core,core_area) in maps.items():
        eligible=detected&(core_area>0)
        inside=eligible[core]
        cavity=inside&~pred
        holes=int(np.count_nonzero(cavity&closed))
        cv=np.bincount(core[cavity],minlength=len(area))
        affected=eligible&((cv>=32)|(cv*10>=core_area))
        lab,count=nd.label(cavity,structure=CONN)
        histogram=np.bincount(lab.ravel())
        grid=edges=components=0
        for i,box in enumerate(nd.find_objects(lab),1):
            if histogram[i]<64:continue
            coords=(box[0].start,box[1].start,box[0].stop,box[1].stop)
            grid+=sum(min(x%16,(-x)%16)<=1 for x in coords)
            edges+=4
            components+=1
        result.append(dict(radius=r,instances=len(area)-1,missed_instances=int((area>0).sum()-detected.sum()),
            detected_instances=int(detected.sum()),eligible_instances=int(eligible.sum()),skipped_instances=int((detected&~eligible).sum()),
            affected_instances=int(affected.sum()),interior_px=int(core_area[eligible].sum()),interior_fn_px=int(cavity.sum()),
            closed_hole_px=holes,open_cavity_px=int(cavity.sum())-holes,large_cavities=components,aligned_edges=grid,total_edges=edges))
        if r==3:
            # Yellow is eroded-away boundary of a detected instance; gray is separate.
            overlay=np.zeros(ids.shape,np.uint8)
            overlay[(ids>0)&pred]=1
            overlay[(ids>0)&~pred&~missed&~inside]=2
            overlay[cavity]=3
            overlay[(ids==0)&pred]=4
            overlay[missed&~pred]=5
    return result,overlay
