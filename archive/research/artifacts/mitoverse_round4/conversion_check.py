from common import *
import time
seed(0);sp=json.loads((BASE/'split_mitoverse_probe_v1.json').read_text());vid=sp['train'][0];row=sp['volumes'][vid];raw=zarr.open_array(row['image_path'],mode='r');d,h,w=raw.shape;im=np.asarray(raw[d//2,(h-256)//2:(h-256)//2+256,(w-256)//2:(w-256)//2+256]);x=torch.from_numpy(normalize(im,'N2'))[None]
a=encoder(encoder_sd(PATHS['P0']),True).eval();b=encoder(encoder_sd(BASE/'natural_converted.pth')).eval();t=time.time()
with torch.inference_mode():
 fa=a.get_intermediate_layers(x,n=[5,11,17,23],reshape=True);fb=b.get_intermediate_layers(x,n=[5,11,17,23],reshape=True)
 r=[{'relative_l2':float((u-v).norm()/u.norm()),'cosine':float(torch.nn.functional.cosine_similarity(u.flatten(),v.flatten(),dim=0))} for u,v in zip(fa,fb)]
save(BASE/'natural_conversion_training_patch_check.json',{'volume':vid,'split':'train','shape':list(x.shape),'normalization':'N2','changes':r,'seconds':time.time()-t,'note':'Additional diagnostic on actual 256-pixel training crop, no test data. Conversion is not output-equivalent.'});print(r)
