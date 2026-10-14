"""Transport regression checks; fake SSH only, no credentials/network/experiments."""
import contextlib,importlib.util,io,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
class SSHFailure(Exception):pass
class AuthenticationFailure(SSHFailure):pass

class CollectorTests(unittest.TestCase):
 def setUp(self):
  self.stub=types.SimpleNamespace(SSHClient=mock.Mock(),AutoAddPolicy=mock.Mock(),SSHException=SSHFailure,AuthenticationException=AuthenticationFailure)
  spec=importlib.util.spec_from_file_location('mv3_collector_test',ROOT/'scripts/watch_mv3_results.py')
  self.module=importlib.util.module_from_spec(spec)
  with mock.patch.dict(sys.modules,{'paramiko':self.stub}):spec.loader.exec_module(self.module)
  self.now=0
  def sleep(seconds):self.now+=seconds
  self.module.time=types.SimpleNamespace(time=lambda:self.now,sleep=sleep)

 def test_connection_reset_reconnects_and_publishes_terminal_results_once(self):
  first,second=mock.MagicMock(),mock.MagicMock();self.stub.SSHClient.side_effect=[first,second]
  self.module.read_remote=mock.Mock(side_effect=[ConnectionResetError('simulated reset'),{'status':'complete'}])
  self.module.collect=mock.Mock(return_value=Path('collected'))
  self.module.publish=mock.Mock(return_value={'status':'PUSHED'})
  with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()):
   self.module.SCRATCH=Path(folder);self.module.watch('test-only',max_seconds=65,retry_seconds=30)
   self.assertEqual(json.loads((Path(folder)/'collector_result.json').read_text())['status'],'PUSHED')
  self.assertEqual(self.stub.SSHClient.call_count,2);self.assertEqual(self.now,30)
  self.module.publish.assert_called_once();first.close.assert_called_once();second.close.assert_called_once()

 def test_reconnects_do_not_extend_original_collection_deadline(self):
  client=mock.MagicMock();client.connect.side_effect=ConnectionResetError('simulated reset');self.stub.SSHClient.return_value=client
  with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(TimeoutError):self.module.watch('test-only',max_seconds=65,retry_seconds=30)
  self.assertEqual(self.now,65);self.assertEqual(client.connect.call_count,3)

 def test_authentication_failure_does_not_retry(self):
  client=mock.MagicMock();client.connect.side_effect=AuthenticationFailure();self.stub.SSHClient.return_value=client
  with self.assertRaises(AuthenticationFailure):self.module.watch('test-only',max_seconds=65)
  client.connect.assert_called_once();client.close.assert_called_once()

if __name__=='__main__':unittest.main()
