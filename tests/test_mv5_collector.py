"""Passive collection boundaries: deadline, auth, and no repeated publication."""
import contextlib,importlib.util,io,sys,tempfile,types,unittest
from pathlib import Path
from unittest import mock
ROOT=Path(__file__).resolve().parents[1]
class SSHFailure(Exception):pass
class AuthFailure(SSHFailure):pass
class MV5CollectorTests(unittest.TestCase):
 def setUp(self):
  self.ssh=types.SimpleNamespace(SSHClient=mock.Mock(),AutoAddPolicy=mock.Mock(),SSHException=SSHFailure,AuthenticationException=AuthFailure)
  spec=importlib.util.spec_from_file_location('mv5_collector_test',ROOT/'scripts/watch_mv5_results.py');self.m=importlib.util.module_from_spec(spec)
  with mock.patch.dict(sys.modules,{'paramiko':self.ssh}):spec.loader.exec_module(self.m)
  self.now=0
  def sleep(seconds):self.now+=seconds
  self.m.time=types.SimpleNamespace(monotonic=lambda:self.now,sleep=sleep)
 def test_dirty_checkout_preserves_collection_without_any_writes_or_push(self):
  self.m.git=mock.Mock(side_effect=[self.m.BRANCH,' M user-file.md'])
  self.assertEqual(self.m.publish(Path('preserved'),{'status':'complete'})['status'],'COLLECTED_CHECKOUT_NOT_CLEAN_EXPECTED_BRANCH')
  self.assertEqual(self.m.git.call_count,2)
 def test_network_reconnect_does_not_extend_deadline(self):
  client=mock.MagicMock();client.connect.side_effect=ConnectionResetError('fake');self.ssh.SSHClient.return_value=client
  with contextlib.redirect_stdout(io.StringIO()):result=self.m.watch('test-only',65)
  self.assertEqual(self.now,65);self.assertEqual(client.connect.call_count,3);self.assertEqual(result['status'],'PASSIVE_DEADLINE_NO_EXPERIMENT_ACTION')
 def test_auth_error_never_retries(self):
  client=mock.MagicMock();client.connect.side_effect=AuthFailure();self.ssh.SSHClient.return_value=client
  with self.assertRaises(AuthFailure):self.m.watch('test-only',65)
  client.connect.assert_called_once();client.close.assert_called_once()
 def test_stale_pr_snapshot_retries_without_republishing_or_changing_draft(self):
  api=mock.Mock(side_effect=[{'draft':True,'head':{'sha':'old'}},{'draft':False,'head':{'sha':'expected'}}])
  pr=self.m.wait_pr_head(api,'expected')
  self.assertEqual(pr['head']['sha'],'expected');self.assertFalse(pr['draft']);self.assertEqual(api.call_count,2);self.assertEqual(self.now,2)
 def test_persistently_stale_pr_receipt_has_finite_wait(self):
  api=mock.Mock(return_value={'head':{'sha':'old'}})
  with self.assertRaisesRegex(RuntimeError,'Branch push completed'):self.m.wait_pr_head(api,'expected')
  self.assertEqual(api.call_count,6);self.assertEqual(self.now,10)
 def test_publication_failure_does_not_repeat_collection_or_commit(self):
  client=mock.MagicMock();self.ssh.SSHClient.return_value=client
  self.m.remote_json=mock.Mock(return_value={'status':'complete'});self.m.publish=mock.Mock(side_effect=OSError('fake publication failure'))
  client.exec_command.return_value=(None,io.BytesIO(b'ActiveState=inactive\nMainPID=0\n'),None)
  with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
   self.m.collect=mock.Mock(return_value=Path(d))
   with self.assertRaises(OSError):self.m.watch('test-only',65)
  self.m.publish.assert_called_once();self.m.collect.assert_called_once()
if __name__=='__main__':unittest.main()
