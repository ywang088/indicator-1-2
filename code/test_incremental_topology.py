import unittest,copy
import numpy as np
from incremental_topology import IncrementalTopology
class IncrementalContracts(unittest.TestCase):
 def config(self):return dict(image_size=[1000,600],trajectories=[dict(id='T1',frame_ids=['0','1','2','3'])])
 def frame(self,i):
  T=np.eye(4);T[2,3]=i*1.5
  return dict(id=str(i),timestamp=float(i)*2,image_sha256='h'+str(i),T_world_camera=T.tolist())
 def column(self,i,zone,side,obj=None,score=.9):
  b=[100,100,180,550] if side=='left' else [800,100,880,550]
  return obj or zone+str(i),dict(id=zone+str(i),label='column',bbox=b,position=[-3 if side=='left' else 3,0,10],texts=[dict(text=zone,parsed={'zone':zone},score=score,source='test_ocr')])
 def test_different_columns_accumulate_zone_evidence(self):
  s=IncrementalTopology(self.config());self.assertEqual(s.update(self.frame(0),[self.column(0,'E','left',score=.5)])['regions_after'],0)
  self.assertEqual(s.update(self.frame(1),[self.column(1,'E','left',score=.5)])['regions_after'],1)
 def test_opposite_labels_and_motion_make_portal_during_replay(self):
  s=IncrementalTopology(self.config());s.update(self.frame(0),[])
  e=s.update(self.frame(1),[self.column(1,'E','left'),self.column(1,'F','right')]);self.assertEqual(e['portals_after'],1);self.assertEqual(len(e['new_portals']),1)
  e=s.update(self.frame(2),[self.column(2,'E','left'),self.column(2,'F','right')]);self.assertEqual(e['portals_after'],1);self.assertEqual(e['new_portals'],[])
 def test_same_side_or_same_zone_never_creates_portal(self):
  for labels in [[self.column(1,'E','right'),self.column(1,'F','right')],[self.column(1,'E','left'),self.column(1,'E','right')]]:
   s=IncrementalTopology(self.config());s.update(self.frame(0),[]);self.assertEqual(s.update(self.frame(1),labels)['portals_after'],0)
 def test_no_motion_and_cross_replay_do_not_connect(self):
  c=self.config();c['trajectories']=[dict(id='T1',frame_ids=['0']),dict(id='T2',frame_ids=['1','2'])];s=IncrementalTopology(c);s.update(self.frame(0),[])
  self.assertEqual(s.update(self.frame(1),[self.column(1,'E','left'),self.column(1,'F','right')])['portals_after'],0)
  f=self.frame(2);f['T_world_camera']=self.frame(1)['T_world_camera'];self.assertEqual(s.update(f,[self.column(2,'E','left'),self.column(2,'F','right')])['portals_after'],0)
 def test_prefix_is_identical_without_future_frames(self):
  inputs=[[],[self.column(1,'E','left')],[self.column(2,'F','right')],[]];s=IncrementalTopology(self.config());full=[s.update(self.frame(i),o) for i,o in enumerate(inputs)]
  for n in range(1,5):
   prefix=IncrementalTopology(self.config());out=[prefix.update(self.frame(i),inputs[i]) for i in range(n)];self.assertEqual(out,full[:n])
 def test_unreadable_or_conflicting_letters_do_not_create_region(self):
  s=IncrementalTopology(self.config());obj,d=self.column(0,'E','left');d['texts'].append(dict(parsed={'zone':'F'},score=.95));self.assertEqual(s.update(self.frame(0),[(obj,d)])['regions_after'],0)
 def test_missing_geometry_can_still_identify_zone(self):
  s=IncrementalTopology(self.config());obj,d=self.column(0,'E','left');d['position']=None;self.assertEqual(s.update(self.frame(0),[(obj,d)])['regions_after'],1)
 def test_passed_pillar_is_not_reused_on_old_image_side(self):
  s=IncrementalTopology(self.config());obj,d=self.column(0,'F','left');d['position']=[-3,0,.8]
  s.update(self.frame(0),[(obj,d)])
  self.assertEqual(s.update(self.frame(1),[self.column(1,'E','right')])['portals_after'],0)
 def test_letter_identity_does_not_survive_conflicting_track(self):
  s=IncrementalTopology(self.config());s.update(self.frame(0),[self.column(0,'E','left',obj='pillar')]);s.update(self.frame(1),[self.column(1,'F','left',obj='pillar')]);self.assertNotIn('pillar',s.zone_by_object)
if __name__=='__main__':unittest.main()
