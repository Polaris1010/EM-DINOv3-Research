import hashlib
import json
import struct
import zlib
from pathlib import Path
import numpy as np
from PIL import Image

out=Path(__file__).resolve().parent
jobs=json.loads((out/'sources/duplicate_probe_jobs.json').read_text())
result=[];arrays=[]
for job in jobs:
    raw=(out/'sources'/f"{job['output_name']}.bin").read_bytes()
    h=struct.unpack_from('<4s5H3L2H',raw)
    if h[0]!=b'PK\x03\x04':raise ValueError('Bad local ZIP header')
    start=30+h[9]+h[10]
    name=raw[30:30+h[9]].decode()
    if name!=job['name']:raise ValueError('ZIP local/central filename mismatch')
    compressed=raw[start:start+job['compressed']]
    if len(compressed)!=job['compressed']:raise ValueError('Truncated compressed payload')
    payload=zlib.decompress(compressed,-15) if job['method']==8 else compressed if job['method']==0 else None
    if payload is None:raise ValueError('Unsupported ZIP compression')
    if len(payload)!=job['size'] or f'{zlib.crc32(payload):08x}'!=job['crc32']:
        raise ValueError('Payload length/CRC mismatch')
    image_path=out/'sources'/f"{job['output_name']}.tiff"
    image_path.write_bytes(payload)
    with Image.open(image_path) as im:a=np.asarray(im).copy()
    arrays.append(a)
    prefix=json.dumps(dict(shape=a.shape,dtype=str(a.dtype)),sort_keys=True).encode()
    result.append(dict(archive_path=name,sha256=hashlib.sha256(payload).hexdigest(),
                       decoded_sha256=hashlib.sha256(prefix+a.tobytes()).hexdigest(),shape=list(a.shape),dtype=str(a.dtype)))
summary=dict(payloads=result,raw_file_equal=result[0]['sha256']==result[2]['sha256'],
             raw_decoded_pixels_equal=result[0]['decoded_sha256']==result[2]['decoded_sha256'],
             label_file_equal=result[1]['sha256']==result[3]['sha256'],
             label_semantic_foreground_equal=bool(np.array_equal(arrays[1]>0,arrays[3]>0)),
             label_semantic_disagreement_pixels=int(np.count_nonzero((arrays[1]>0)!=(arrays[3]>0))),
             total_image_pixels=int(arrays[0].size),
             label_foreground_pixels=[int(np.count_nonzero(arrays[1])),int(np.count_nonzero(arrays[3]))],
             acquisition_identity='unresolved',relation='L0' if result[0]['sha256']==result[2]['sha256'] else 'candidate',
             no_automatic_training_admission=True)
(out/'mitolab_duplicate_probe_evidence.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='payloads'},indent=2))
