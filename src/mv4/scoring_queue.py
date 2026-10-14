"""Bounded scheduling of existing score jobs; no scoring/math/model changes."""
import time

def score_queue(tasks,launch,health,available_bytes,sleep=time.sleep,reserve_bytes=16<<30):
 pending=sorted(tasks,key=lambda row:row['memory_bytes']);active={}
 while pending or active:
  for gpu,(process,need) in list(active.items()):
   code=process.poll()
   if code is not None:
    if code!=0:raise RuntimeError(f'Own score process {process.pid} failed with {code}')
    del active[gpu]
  for gpu in [1,2,3]:
   if gpu in active or not pending:continue
   capacity=available_bytes()-reserve_bytes-sum(need for _,need in active.values())
   task=pending[0];need=task['memory_bytes']
   if need>capacity:
    if not active:raise MemoryError('Insufficient RAM for the next score job; preserve existing results')
    break
   process=launch(task,gpu);pending.pop(0);active[gpu]=(process,need)
  health()
  if pending or active:sleep(1)
