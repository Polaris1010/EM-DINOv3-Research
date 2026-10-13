"""Evidence for selected high-risk image pairs; never infer acquisition identity from similarity."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image


def decoded_hash(a):
    prefix=json.dumps(dict(shape=a.shape,dtype=str(a.dtype)),sort_keys=True).encode()
    return hashlib.sha256(prefix+a.tobytes()).hexdigest()


def phash(a):
    a=np.asarray(Image.fromarray(a).resize((32,32)),dtype=float)
    k=np.arange(8)[:,None];x=np.arange(32)[None,:]
    c=np.cos(np.pi*(2*x+1)*k/64)
    d=c@a@c.T
    bits=d.ravel()[1:] > np.median(d.ravel()[1:])
    return bits


def reduce(a):
    image=Image.fromarray(a)
    image.thumbnail((256,256))
    return np.asarray(image,dtype=float)


def registration(a,b):
    best={'ncc':None}
    base=reduce(a)
    template=reduce(b)
    for flip in [False,True]:
        for rotation in range(4):
            t=np.rot90(np.fliplr(template) if flip else template,rotation)
            for scale in [0.5,0.75,1.,1.5,2.]:
                width,height=round(t.shape[1]*scale),round(t.shape[0]*scale)
                if min(width,height)<16 or height>base.shape[0] or width>base.shape[1]:continue
                patch=np.asarray(Image.fromarray(t.astype('uint8')).resize((width,height)),dtype=float)
                centered=patch-patch.mean(); energy=(centered**2).sum()
                if energy<1e-8:continue
                fftshape=(base.shape[0]+height-1,base.shape[1]+width-1)
                corr=np.fft.irfft2(np.fft.rfft2(base,s=fftshape)*np.fft.rfft2(centered[::-1,::-1],s=fftshape),s=fftshape)
                corr=corr[height-1:base.shape[0],width-1:base.shape[1]]
                def sums(v):
                    c=np.pad(v.cumsum(0).cumsum(1),((1,0),(1,0)))
                    return c[height:,width:]-c[:-height,width:]-c[height:,:-width]+c[:-height,:-width]
                variance=np.maximum(sums(base**2)-sums(base)**2/(height*width),0)
                score=np.divide(corr,np.sqrt(variance*energy),out=np.full(corr.shape,-np.inf),where=variance>1e-8)
                score=np.where(np.isfinite(score),score,0)
                pos=np.unravel_index(np.argmax(np.abs(score)),score.shape);value=float(score[pos])
                if np.isfinite(value) and (best['ncc'] is None or abs(value)>abs(best['ncc'])):
                    best=dict(ncc=value,flip_lr=flip,rotation90=rotation,scale=scale,
                              contrast_inverted=value<0,
                              reduced_grid_yx=list(map(int,pos)),template_shape_yx=[height,width],
                              coordinate_frame='independently_downsampled_image_grids_not_parent_ROI')
    return best


def inspect(a_path,b_path):
    with Image.open(a_path) as im:a=np.asarray(im).copy();ag=np.asarray(im.convert('L')).copy()
    with Image.open(b_path) as im:b=np.asarray(im).copy();bg=np.asarray(im.convert('L')).copy()
    exact=decoded_hash(a)==decoded_hash(b)
    return dict(image_a=str(a_path),image_b=str(b_path),file_hash_equal=Path(a_path).read_bytes()==Path(b_path).read_bytes(),
                decoded_pixel_hash_equal=exact,decoded_hash_a=decoded_hash(a),decoded_hash_b=decoded_hash(b),
                phash_hamming=int(np.count_nonzero(phash(ag)!=phash(bg))),
                registration_a_contains_b=registration(ag,bg),registration_b_contains_a=registration(bg,ag),
                relation='L0_decoded_exact' if exact else 'requires_manual_evidence_review',
                acquisition_identity='not_determined',physical_ROI_identity='not_determined',
                note='Similarity only retrieves candidates; score thresholds do not certify L1/L2 or independence.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pairs',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    with args.pairs.open(encoding='utf-8-sig',newline='') as f:pairs=list(csv.DictReader(f))
    evidence=[inspect(Path(r['image_a']),Path(r['image_b'])) for r in pairs]
    args.out.write_text(json.dumps(evidence,indent=2),encoding='utf-8')


if __name__=='__main__':main()
