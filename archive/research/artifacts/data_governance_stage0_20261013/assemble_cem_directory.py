import json
from pathlib import Path

src = Path(__file__).resolve().parent / 'sources'
offset, size, chunk = 61542376666, 291420256, 10000000
receipts = []
paths = []
for i, start in enumerate(range(0, size, chunk)):
    p = src / f'cem_cd_part_{i:02d}.bin'
    r = json.loads(Path(str(p)+'.receipt.json').read_text(encoding='utf-8-sig'))
    length = min(chunk, size-start)
    expected = f'bytes {offset+start}-{offset+start+length-1}/61833797020'
    # Total archive length is fixed by the byte-range tail receipt, not guessed from EOCD.
    tail = json.loads((src/'cem_zip_tail.bin.receipt.json').read_text(encoding='utf-8-sig'))
    total = tail['content_range'].split('/')[-1]
    expected = expected.rsplit('/',1)[0]+'/'+total
    if r['content_range'] != expected or p.stat().st_size != length or r['bytes'] != length:
        raise ValueError(f'Range receipt mismatch: part {i}')
    if r['etag'] != tail['etag']:
        raise ValueError('Archive changed during ranged downloads')
    paths.append(p)
    receipts.append(r)
dest = src/'cem_central_directory_complete.bin'
with dest.open('wb') as out:
    for p in paths:
        with p.open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):
                out.write(b)
(src/'cem_central_directory_complete.receipt.json').write_text(json.dumps(receipts,indent=2),encoding='utf-8')
print('Validated and assembled central directory bytes:',dest.stat().st_size)
