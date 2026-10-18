"""Physical GPU allowlist; UUID remapping never exposes physicalGPU0."""
import csv,os,subprocess,urllib.request
def gpu_mapping():
 return {r[1].strip():int(r[0]) for r in csv.reader(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines())}
def training_gpu_guard(world_size):
 mapping=gpu_mapping();visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',');assert len(visible)==world_size
 assert len(set(visible))==len(visible) and all(v in mapping and mapping[v] in [1,2,3] for v in visible)
def website():
 with urllib.request.urlopen('http://127.0.0.1:7860/',timeout=10) as response:assert response.status==200
