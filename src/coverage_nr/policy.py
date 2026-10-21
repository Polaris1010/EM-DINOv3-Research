"""Explicit shared100k continuation schedule and strict30k lineage validation."""
import math
def learning_rate(step,peak,minimum):
 if not 0<=step<100000:raise ValueError('Step outside100k budget')
 if step<5000:return peak*(step+1)/5000
 return minimum+(peak-minimum)*0.5*(1+math.cos(math.pi*(step-5000)/95000))
def validate_parent(saved,current,receipt,expected_path,parent_config_sha,parent_trainer_sha):
 expected={**current,'config_sha256':parent_config_sha,'trainer_sha256':parent_trainer_sha,'total_steps':30000,'run_path':expected_path}
 if saved!=expected:raise ValueError('Parent identity mismatch: '+str(sorted(k for k in set(saved)|set(expected) if saved.get(k)!=expected.get(k))))
 if receipt.get('step')!=30000 or any(receipt.get(k)!=v for k,v in saved.items()):raise ValueError('Parent receipt identity mismatch')
