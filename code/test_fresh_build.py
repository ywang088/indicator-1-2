"""Fresh-inference entrypoint must never leave old success eligible after failure."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import fresh_build
class FreshFailureTests(unittest.TestCase):
 def test_missing_weights_invalidates_old_success(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);fresh_build.save(root/'results/execution.json',{'pipeline_mode':'fresh_sam3','completed':True})
   with patch.object(fresh_build,'ROOT',root),patch('sys.argv',['fresh_build.py']),patch.dict('os.environ',{'SAM3_CHECKPOINT':str(root/'absent.pt'),'SAM3_BPE':str(root/'absent.gz')}):
    with self.assertRaises(FileNotFoundError):fresh_build.main()
   self.assertFalse(json.loads((root/'results/execution.json').read_text())['completed'])
 def test_failed_smoke_preserves_completed_full_result(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);original={'pipeline_mode':'fresh_sam3','completed':True};fresh_build.save(root/'results/execution.json',original)
   with patch.object(fresh_build,'ROOT',root),patch('sys.argv',['fresh_build.py','--smoke-frames','2']),patch.dict('os.environ',{'SAM3_CHECKPOINT':str(root/'absent.pt'),'SAM3_BPE':str(root/'absent.gz')}):
    with self.assertRaises(FileNotFoundError):fresh_build.main()
   self.assertEqual(json.loads((root/'results/execution.json').read_text()),original)
if __name__=='__main__':unittest.main()
