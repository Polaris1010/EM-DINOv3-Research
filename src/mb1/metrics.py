"""Explicit semantic empty cases and equal-dataset/equal-seed reductions."""
import math,statistics
METRICS=['dice','iou','precision','recall','hd95_nm','nsd_16nm','predicted_foreground_fraction']
def average(values):
 values=[v for v in values if v is not None];return statistics.mean(values) if values else None
def overlap(tp,fp,fn,tn):
 assert min(tp,fp,fn,tn)>=0 and tp+fp+fn+tn>0
 return {'dice':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,'iou':tp/(tp+fp+fn) if tp+fp+fn else 1.,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,'predicted_foreground_fraction':(tp+fp)/(tp+fp+fn+tn)}
def distribution(values):
 present=sorted(v for v in values if v is not None);finite=[v for v in present if math.isfinite(v)];n=len(present)
 def q(p):return present[max(0,min(n-1,math.ceil(p*n)-1))] if n else None
 return {'n':n,'n_missing':len(values)-n,'n_infinite':n-len(finite),'mean':average(present),'sample_sd':statistics.stdev(finite) if n==len(finite) and n>1 else None,'min':present[0] if n else None,'p10':q(.1),'median':q(.5),'p90':q(.9),'max':present[-1] if n else None,'quantiles':'empirical inverse CDF, no interpolation'}
def equal_model_dataset(rows):
 datasets=sorted({r['dataset_id'] for r in rows});assert len(datasets)==7
 models=sorted({r['model'] for r in rows});out={}
 for model in models:
  out[model]={k:average([average([r[k] for r in rows if r['model']==model and r['dataset_id']==d]) for d in datasets]) for k in METRICS}
 return out
