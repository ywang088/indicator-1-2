import unittest,tempfile,json
from pathlib import Path
import numpy as np
from portal_connections import trajectory_transitions,validate_reviewed_connection
from map_pipeline import make_graph
import test_pipeline

class PortalContracts(unittest.TestCase):
 def fixture(self):
  frames={str(i):dict(id=str(i),timestamp=float(i),image_sha256='h'+str(i),T_world_camera=[[1,0,0,i],[0,1,0,0],[0,0,1,0],[0,0,0,1]]) for i in range(4)}
  config=dict(trajectories=[dict(id='T1',frame_ids=list(frames))]);states=dict(zip(frames,['A','A','B','B']));methods={i:'frozen_boundary' for i in frames};return config,frames,states,methods
 def test_door_alone_produces_no_portal(self):
  helper=test_pipeline.Contracts();c=helper.config();rows=helper.rows()
  for row in rows:row['detections'][0]['label']='door'
  with tempfile.TemporaryDirectory() as tmp:
   result=make_graph(c,rows,Path(tmp)/'map.sqlite',Path(tmp)/'updates.jsonl',{})
   self.assertEqual(result['portals'],0);self.assertEqual(result['objects'],1)
 def test_region_crossing_without_door_produces_connection(self):
  c,f,s,m=self.fixture();accepted,pending,audit=trajectory_transitions(c,s,m,f)
  self.assertEqual(len(accepted),1);self.assertEqual(accepted[0]['regions'],['A','B']);self.assertFalse(accepted[0]['requires_physical_door']);self.assertEqual(pending,[])
 def test_nearest_column_switch_remains_pending(self):
  c,f,s,m=self.fixture();m={i:'nearest_column_candidate' for i in f};accepted,pending,_=trajectory_transitions(c,s,m,f);self.assertEqual(accepted,[]);self.assertIn('camera_region_assignment_not_confirmed',pending[0]['reasons'])
 def test_different_replays_never_create_crossing(self):
  c,f,s,m=self.fixture();c['trajectories']=[dict(id='T1',frame_ids=['0','1']),dict(id='T2',frame_ids=['2','3'])];accepted,pending,_=trajectory_transitions(c,s,m,f);self.assertEqual(accepted,[]);self.assertEqual(pending,[])
 def test_rotation_without_translation_is_not_crossing(self):
  c,f,s,m=self.fixture()
  for row in f.values():row['T_world_camera']=np.eye(4).tolist()
  accepted,pending,_=trajectory_transitions(c,s,m,f);self.assertEqual(accepted,[]);self.assertIn('insufficient_motion_possible_camera_rotation',pending[0]['reasons'])
 def test_duplicate_region_or_unbound_photos_rejected(self):
  c,f,s,m=self.fixture();p=dict(id='portal:x',regions=['A','A'],reviewer='test',connection_confirmed=True,connection_type='open_passage')
  with self.assertRaises(ValueError):validate_reviewed_connection(p,c,f)
  p.update(regions=['A','B'],method='trajectory_crossing_review',trajectory_id='T1',evidence_frames=['1','2'],side_evidence=[dict(region_id='A',frame_id='1',image_sha256='wrong',basis='test'),dict(region_id='B',frame_id='2',image_sha256='h2',basis='test')])
  with self.assertRaises(ValueError):validate_reviewed_connection(p,c,f)
  p['side_evidence'][0]['image_sha256']='h1';validate_reviewed_connection(p,c,f)
if __name__=='__main__':unittest.main()

class SharedCorridorContracts(unittest.TestCase):
 def test_real_review_and_rejected_missing_sides_or_discontinuous_path(self):
  import copy
  c,frames,_,_=PortalContracts().fixture()
  p=dict(id='portal:test',regions=['A','B'],reviewer='test',connection_confirmed=True,connection_type='shared_corridor',method='shared_corridor_visual_review',opposite_sides_confirmed=True,open_access_confirmed=True,human_acceptance_completed=False,evidence_frames=list(frames),layout_evidence_frames=['1','2'],side_evidence=[dict(region_id='A',frame_id='1',image_sha256='h1',basis='label on left'),dict(region_id='B',frame_id='2',image_sha256='h2',basis='label on right')],observed_segments=[dict(trajectory_id='T1',frame_ids=list(frames),camera_path_xyz=[[i,0,0] for i in range(4)])])
  validate_reviewed_connection(p,c,frames)
  for mutation in ['side','layout','segment','pose']:
   q=copy.deepcopy(p)
   if mutation=='side':q['side_evidence']=q['side_evidence'][:1]
   if mutation=='layout':q['layout_evidence_frames']=[]
   if mutation=='segment':q['observed_segments'][0]['frame_ids'][1]='3'
   if mutation=='pose':q['observed_segments'][0]['camera_path_xyz'][0][0]+=10
   with self.assertRaises(ValueError):validate_reviewed_connection(q,c,frames)
