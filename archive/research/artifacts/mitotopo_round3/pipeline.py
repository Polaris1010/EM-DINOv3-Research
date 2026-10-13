"""Bounded per-GPU continuation of the authorized experiment matrix."""
import os,sys,json,time,subprocess,argparse
from pathlib import Path
BASE=Path(__file__).resolve().parent
PY='/opt/data/3_117/tanghr/envs/meddinov3/bin/python'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True,choices=[1,2,3]);a=ap.parse_args();gpu=a.gpu
    upstream=BASE/'jobs'/f'ssl_b2_w{gpu-1}_s2000'
    while not (upstream/'guard_result.json').exists():time.sleep(10)
    guard=json.loads((upstream/'guard_result.json').read_text());result=json.loads((upstream/'result.json').read_text())
    assert guard['exit_code']==0 and guard['stop_reason'] is None and result['status']=='complete'
    jobs={
      1:[('mitonet_full_v2','evaluate.py',['--mode','mitonet']),('probe_em_final','probe.py',['--encoder','em','--eval-from',str(BASE/'jobs/probe_em_v2')]),('probe_w0','probe.py',['--encoder',str(upstream/'teacher_latest.pth')])],
      2:[('gram200k_swap_full','evaluate.py',['--mode','swap']),('probe_gram200k_v3','probe.py',['--encoder','gram200k']),('probe_w1','probe.py',['--encoder',str(upstream/'teacher_latest.pth')])],
      3:[('em_native_recheck','evaluate.py',['--mode','native_em']),('probe_w2','probe.py',['--encoder',str(upstream/'teacher_latest.pth')])]
    }[gpu]
    for name,script,args in jobs:
        if (BASE/'jobs'/name).exists():
            old=json.loads((BASE/'jobs'/name/'guard_result.json').read_text())
            assert old['exit_code']==0 and old['stop_reason'] is None,(name,old)
            print('ALREADY_COMPLETE',name,flush=True);continue
        cmd=[PY,'-u',str(BASE/'guard.py'),'--gpu',str(gpu),'--run',name,'--python',PY,str(BASE/script),*args]
        print('START',name,flush=True);code=subprocess.call(cmd,cwd=BASE)
        g=json.loads((BASE/'jobs'/name/'guard_result.json').read_text())
        assert code==0 and g['exit_code']==0 and g['stop_reason'] is None,(name,g)
        print('COMPLETE',name,flush=True)
    (BASE/f'pipeline_gpu{gpu}_complete.json').write_text(json.dumps({'status':'complete','jobs':[j[0] for j in jobs]}))
if __name__=='__main__':main()
