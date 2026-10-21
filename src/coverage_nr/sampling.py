"""Deterministic image-only sampling and lossless indexed ZIP access."""
import bisect,io,random,sqlite3,struct,zlib

def choose(pool,cem,fraction,seed,index):
 rng=random.Random(seed*1_000_000_007+index);coin=rng.random()
 if coin<fraction:
  groups=cem['groups'];cumulative=[];total=0.
  for g in groups:total+=g['probability'];cumulative.append(total)
  group=groups[min(bisect.bisect_right(cumulative,rng.random()*total),len(groups)-1)]
  return {'domain':'CEM1.5M','source':group['name'],'volume_id':group['name'],'image_id':rng.randint(group['first_id'],group['last_id']),'view_seed':rng.randrange(2**31-1),'index':index}
 sources=sorted(pool['sampling_groups']);source=rng.choice(sources);acqs=pool['sampling_groups'][source];acq=rng.choice(sorted(acqs));vid=rng.choice(acqs[acq]);row=next(r for r in pool['volumes'] if r['volume_id']==vid)
 offset=rng.randrange(row['legal_slices']);z=None
 for a,b in row['allowed_z']:
  if offset<b-a:z=a+offset;break
  offset-=b-a
 assert z is not None
 return {'domain':'MitoVerse','source':source,'acquisition_sampling_key':acq,'volume_id':vid,'z':z,'y':rng.randrange(max(1,row['shape'][1]-512+1)),'x':rng.randrange(max(1,row['shape'][2]-512+1)),'view_seed':rng.randrange(2**31-1),'index':index}

def read_member(handle,offset,compressed,size,crc,method):
 handle.seek(offset);header=handle.read(30);fields=struct.unpack('<4s5H3I2H',header)
 assert fields[0]==b'PK\x03\x04' and fields[3]==method and not fields[2]&1
 handle.seek(fields[-2]+fields[-1],1);data=handle.read(compressed);assert len(data)==compressed
 value=zlib.decompress(data,-15) if method==8 else data
 assert len(value)==size and zlib.crc32(value)&0xffffffff==crc,'Corrupted selected CEM archive member'
 return value
