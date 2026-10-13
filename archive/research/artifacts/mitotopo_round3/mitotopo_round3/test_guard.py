"""Regression checks for process lifetime and PID reuse handling; no GPU calls."""
from pathlib import Path
import json
import guard
BASE=Path(__file__).resolve().parent/'guard_testdata';(BASE/'jobs/fake').mkdir(parents=True,exist_ok=True)
guard.BASE=BASE
g={i:{'uuid':f'GPU-{i}'} for i in [0,1,2,3]}
profile={'cwd':'/unknown','cmd':['python','unknown.py'],'birth':10.,'alive':True,'status':'running'}
class P:
 def __init__(self,pid):self.pid=pid
 def cwd(self):return profile['cwd']
 def cmdline(self):return profile['cmd']
 def username(self):return 'root'
 def create_time(self):return profile['birth']
 def is_running(self):return profile['alive']
 def status(self):return profile['status']
guard.psutil.Process=P;guard.run=lambda args:'GPU-3, 123\n'
def rejected():
 try:guard.check_processes(g)
 except RuntimeError:return True
 return False
assert rejected(),'unknown living process must stop'
profile['status']=guard.psutil.STATUS_ZOMBIE
assert guard.check_processes(g)==[(3,123)],'exited zombie must not be misclassified as active foreign process'
profile['status']='running';profile['alive']=False
assert guard.check_processes(g)==[(3,123)]
profile['alive']=True;profile['cwd']='/own';profile['cmd']=['python','ours.py']
record={'pid':123,'create_time':10.,'cwd':'/own','command':['python','ours.py']};f=BASE/'jobs/fake/pid.json';f.write_text(json.dumps(record))
assert guard.check_processes(g)==[(3,123)]
profile['birth']=11.;assert rejected(),'reused PID must not inherit old authorization'
profile.update(cwd=guard.SERVICE,cmd=[guard.SERVICE+'/venv/bin/python','-m','uvicorn','server.app:app'])
assert guard.check_processes(g)==[(3,123)]
guard.run=lambda args:'GPU-0, 999\n';assert guard.check_processes(g)==[]
print('PASS: unknown process, zombie exit, disappeared process, exact own registry, PID reuse, existing website, GPU0 exclusion')
