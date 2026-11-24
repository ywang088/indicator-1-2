import json,sqlite3,tempfile,unittest,subprocess,sys
from pathlib import Path
import numpy as np
from perception import zone_text,attach_text,projection,locate,dump
from map_pipeline import make_graph,load_query

class Contracts(unittest.TestCase):
 def config(self):
  return dict(scene='synthetic-test-only',image_size=[1280,720],min_confidence=.3,merge_radius_m=.8,region_radius_m=12,region_vote_min_frames=2,region_vote_min_motion_m=.5,regions=[],portals=[],frames=[dict(id=str(i),timestamp=i,image_sha256='test',T_world_camera=[[1,0,0,i],[0,1,0,0],[0,0,1,0],[0,0,0,1]]) for i in range(3)])
 def rows(self):
  return [dict(frame_id=str(i),detections=[dict(id=f'{i}:0',label='column',score=.9,bbox=[1,1,5,8],position=[1,1,5],texts=[dict(parsed=zone_text('B2-F12'),score=.95)])]) for i in range(3)]
 def test_geometry_cache_allows_paths_but_rejects_changed_pose(self):
  from geometry_io import geometry_scope_sha256
  import copy
  c=self.config();c.update(K=np.eye(3).tolist(),bounds_xyz=[[0,0,0],[10,10,10]],pointcloud=dict(path='/local/map',T_world_source=np.eye(4).tolist(),voxel_m=.1),trajectories=[dict(id='T01',frame_ids=['0','1','2'])])
  rebound=copy.deepcopy(c);rebound['pointcloud']['path']='/server/map';rebound['frames'][0]['image']='/server/0.png'
  self.assertEqual(geometry_scope_sha256(c),geometry_scope_sha256(rebound))
  rebound['frames'][0]['T_world_camera'][0][3]+=.1
  self.assertNotEqual(geometry_scope_sha256(c),geometry_scope_sha256(rebound))
 def test_text_does_not_parse_exit_sign_as_region(self):
  self.assertIsNone(zone_text('出口 F区向右'));self.assertIsNone(zone_text('车牌粤B12345'));self.assertEqual(zone_text('B2-F12')['zone'],'F');self.assertIsNone(zone_text('B2')['zone'])
 def test_ocr_must_overlap_column_mask(self):
  mask=np.zeros((100,100),bool);mask[20:80,20:60]=True
  d=dict(label='column',mask=mask);text=[dict(text='F',score=.9,bbox=[25,25,40,45]),dict(text='E',score=.9,bbox=[70,25,90,45])]
  self.assertEqual([t['parsed']['zone'] for t in attach_text(d,text)],['F'])
 def test_projection_nearest_surface(self):
  xyz=np.array([[0,0,2],[0,0,4],[0,0,-1],[30,0,2]])
  ids,uv,z=projection(xyz,dict(T_world_camera=np.eye(4).tolist()),[[10,0,5],[0,10,5],[0,0,1]],[10,10]);self.assertEqual(ids.tolist(),[0]);self.assertEqual(z.tolist(),[2])
 def test_graph_merge_regions_and_fresh_replay(self):
  with tempfile.TemporaryDirectory() as d:
   d=Path(d);s=make_graph(self.config(),self.rows(),d/'map.sqlite',d/'log.jsonl',{})
   self.assertEqual(s['objects'],1);self.assertEqual(s['regions'],1);q=load_query(d/'map.sqlite');self.assertEqual(q['node_counts']['frame'],3)
   s=make_graph(self.config(),self.rows()[:1],d/'fresh.sqlite',d/'fresh.jsonl',{});self.assertEqual(s['regions'],0)
   self.assertEqual(len(load_query(d/'fresh.sqlite')['node']['attributes']['observations']),1)
 def test_same_frame_instances_never_merge(self):
  r=self.rows()[:1];r[0]['detections'].append(dict(r[0]['detections'][0],id='0:1'))
  with tempfile.TemporaryDirectory() as d:
   d=Path(d);s=make_graph(self.config(),r,d/'db',d/'log',{});self.assertEqual(s['objects'],2)
 def test_conflicting_votes_stay_unresolved(self):
  r=self.rows();r[1]['detections'][0]['texts'][0]['parsed']=zone_text('E');r[2]['detections'][0]['texts'][0]['parsed']=zone_text('H')
  with tempfile.TemporaryDirectory() as d:
   d=Path(d);s=make_graph(self.config(),r,d/'db',d/'log',{});self.assertEqual(s['regions'],0)
 def test_static_view_not_multiview_confirmation(self):
  c=self.config()
  for f in c['frames']:f['T_world_camera']=np.eye(4).tolist()
  with tempfile.TemporaryDirectory() as d:
   d=Path(d);s=make_graph(c,self.rows(),d/'db',d/'log',{});self.assertEqual(s['regions'],0)
 def test_independent_process_and_missing_evidence_gate(self):
  from perception import sha
  with tempfile.TemporaryDirectory() as d:
   d=Path(d);cloud=d/'cloud.npy';np.save(cloud,np.random.default_rng(0).uniform(0,10,(10000,3)).astype('f4'))
   c=self.config();c.update(pointcloud=dict(path=str(cloud),sha256=sha(cloud),T_world_source=np.eye(4).tolist(),voxel_m=.1,scope_verified=False),bounds_xyz=[[0,0,0],[10,10,10]],trajectories=[dict(id='T01',frame_ids=['0','1','2'])],calibration_status='test_not_official',development_only=True)
   dump(d/'config.json',c);rows=self.rows()
   for r in rows:r.update(image_sha256='test',timestamp=int(r['frame_id']))
   (d/'predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
   subprocess.run([sys.executable,str(Path(__file__).with_name('map_pipeline.py')),'build','--config',str(d/'config.json'),'--predictions',str(d/'predictions.jsonl'),'--output',str(d/'out')],check=True,capture_output=True)
   r=json.loads((d/'out/report.json').read_text());self.assertNotIn('status',r);self.assertNotIn('evidence_gaps',r);self.assertGreater(r['reduction_percent'],20)
   self.assertTrue((d/'out/index.html').is_file());self.assertTrue((d/'out/audit/reload_queries.json').is_file())
if __name__=='__main__':unittest.main()
