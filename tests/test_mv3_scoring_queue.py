"""Scheduling safety regression checks with simulated processes, CPU only."""
import importlib.util,unittest
from pathlib import Path

spec=importlib.util.spec_from_file_location('mv3_scoring_queue',Path(__file__).resolve().parents[1]/'src/mv3/scoring_queue.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class ScoringQueueTests(unittest.TestCase):
 def simulate(self,needs,capacity=1000,failing=None):
  now=[0];launched=[];peak=[0];running={}
  class Process:
   def __init__(self,identifier,gpu):self.pid=identifier;self.gpu=gpu;self.finish=now[0]+2
   def poll(self):
    if now[0]>=self.finish:return 1 if self.pid==failing else 0
    return None
  def launch(task,gpu):
   alive=[p for p in running.values() if p.poll() is None]
   self.assertNotIn(gpu,[p.gpu for p in alive]);self.assertIn(gpu,[1,2,3])
   p=Process(task['id'],gpu);running[p.pid]=p;launched.append(task['id']);peak[0]=max(peak[0],len(alive)+1);return p
  def sleep(seconds):now[0]+=seconds
  module.score_queue([{'id':i,'memory_bytes':n} for i,n in enumerate(needs)],launch,lambda:None,lambda:capacity,sleep=sleep,reserve_bytes=0)
  return launched,peak[0]
 def test_all_jobs_once_at_most_three_workers_on_allowed_gpus(self):
  launched,peak=self.simulate([10]*8);self.assertEqual(sorted(launched),list(range(8)));self.assertEqual(peak,3)
 def test_aggregate_memory_reservation_limits_concurrency(self):
  launched,peak=self.simulate([60]*3,capacity=100);self.assertEqual(len(launched),3);self.assertEqual(peak,1)
 def test_failed_worker_stops_further_scheduling(self):
  with self.assertRaises(RuntimeError):self.simulate([10]*8,failing=0)
 def test_unaffordable_job_preserves_outputs_and_fails_explicitly(self):
  with self.assertRaises(MemoryError):self.simulate([101],capacity=100)

if __name__=='__main__':unittest.main()
