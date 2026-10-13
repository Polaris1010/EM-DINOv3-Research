"""Fail-closed heartbeat wrapper; only signals its own newly created process group."""
import os,sys,time,threading,signal,runpy,json
from pathlib import Path
def watch():
 p=Path(os.environ['FULL_HEARTBEAT'])
 while True:
  try: age=time.monotonic()-json.loads(p.read_text())['monotonic']
  except (OSError,ValueError,KeyError): age=999
  if age>30:
   print('Coordinator heartbeat stale; releasing this registered process group',flush=True)
   os.killpg(os.getpgrp(),signal.SIGKILL)
  time.sleep(2)
if __name__=='__main__':
 assert os.getpid()==os.getpgrp(), 'Worker must have a dedicated process group'
 threading.Thread(target=watch,daemon=True).start()
 sys.argv=sys.argv[1:]
 runpy.run_path(sys.argv[0],run_name='__main__')
