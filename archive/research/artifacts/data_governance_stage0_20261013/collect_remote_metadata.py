"""Read-only metadata collection on the data host. No labels/pixels, training or services."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    root=args.root.resolve(strict=True)
    catalog=json.loads((root/'catalog.json').read_text(encoding='utf-8'))
    result=[]
    for vid,row in sorted(catalog.items()):
        base=(root/row['zarr']).resolve(strict=True)
        if not base.is_relative_to(root):
            raise ValueError('Catalog path escaped declared data root')
        evidence={}
        for name in ['.zattrs','.zmetadata','img/.zarray','mito/.zarray','mask/.zarray']:
            p=base/name
            if p.exists():
                b=p.read_bytes()
                evidence[name]=dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),data=json.loads(b))
        result.append(dict(volume_id=vid,evidence=evidence))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(dict(root=str(root),pixel_data_read=False,
                        volumes=result),indent=2),encoding='utf-8')
    print('Metadata volumes collected:',len(result))


if __name__=='__main__':
    main()
