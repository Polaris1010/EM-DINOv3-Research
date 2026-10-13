import subprocess,os,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
GUARD='/opt/data/3_117/tanghr/envs/meddinov3/bin/python'
SSL='/opt/data/3_117/tanghr/dinov3_shared/bin/python'
def main():
    launches=[]
    for gpu,weight in [(1,0),(2,1),(3,2)]:
        name=f'ssl_b2_w{weight}_s2000'
        assert not (BASE/'jobs'/name).exists(),name
        f=open(BASE/(name+'_guard.log'),'w')
        cmd=[GUARD,'-u',str(BASE/'guard.py'),'--gpu',str(gpu),'--run',name,'--python',SSL,str(BASE/'ssl_sweep.py'),'--steps','2000','--gram-weight',str(weight),'--batch','2','--workers','0']
        p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,cwd=BASE,start_new_session=True)
        launches.append({'gpu':gpu,'weight':weight,'guard_pid':p.pid,'command':cmd})
    (BASE/'sweep_launches.json').write_text(json.dumps(launches,indent=2));print(json.dumps(launches),flush=True)
if __name__=='__main__':main()
