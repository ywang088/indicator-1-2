import copy,unittest
import test_incremental_topology as oldtests
from color_topology import ColorTopology
class ColorContracts(unittest.TestCase):
 def setUp(self):self.helper=oldtests.IncrementalContracts();self.c=self.helper.config();self.c['color_regions']=True
 def col(self,i,color,letter=None,side='left'):
  obj,d=self.helper.column(i,letter or 'E',side,obj=color+str(i));d['texts']=d['texts'] if letter else [];d['pillar_color']={'color':color,'image_sha256':'h'+str(i)};return obj,d
 def test_provisional_then_stable_rename(self):
  s=ColorTopology(self.c);e=s.update(self.helper.frame(0),[self.col(0,'blue')]);rid=e['region_ids'][0];self.assertEqual(s.regions[rid]['label'],'蓝色')
  e=s.update(self.helper.frame(1),[self.col(1,'blue','E')]);self.assertEqual(e['region_ids'],[rid]);self.assertEqual(s.regions[rid]['label'],'E');self.assertEqual(len(s.regions[rid]['anchors']),2)
 def test_portal_endpoints_survive_rename(self):
  s=ColorTopology(self.c);s.update(self.helper.frame(0),[self.col(0,'red'),self.col(0,'yellow',side='right')]);s.update(self.helper.frame(1),[self.col(1,'red'),self.col(1,'yellow',side='right')]);pid=next(iter(s.portals));ends=list(s.portals[pid]['regions'])
  s.update(self.helper.frame(2),[self.col(2,'red','F'),self.col(2,'yellow','E','right')]);self.assertEqual(ends,s.portals[pid]['regions']);self.assertEqual(set(s.bindings.values()),{'E','F'})
 def test_conflict_suspends_binding(self):
  s=ColorTopology(self.c);s.update(self.helper.frame(0),[self.col(0,'red','F')]);s.update(self.helper.frame(1),[self.col(1,'red','E')]);self.assertIn('red',s.blocked);self.assertNotIn('red',s.bindings)
 def test_letter_binds_at_most_one_color(self):
  s=ColorTopology(self.c);s.update(self.helper.frame(0),[self.col(0,'red','F')]);s.update(self.helper.frame(1),[self.col(1,'blue','F')]);self.assertFalse(s.bindings);self.assertEqual(s.blocked,{'red','blue'})
 def test_prefix_and_reset(self):
  obs=[[self.col(0,'blue')],[self.col(1,'blue','E')],[self.col(2,'blue')]];s=ColorTopology(self.c);full=[s.update(self.helper.frame(i),o) for i,o in enumerate(obs)]
  for n in [1,2]:
   ss=ColorTopology(self.c);self.assertEqual([ss.update(self.helper.frame(i),o) for i,o in enumerate(obs[:n])],full[:n])
  fresh=ColorTopology(self.c);fresh.update(self.helper.frame(0),obs[0]);self.assertEqual(fresh.bindings,{})
 def test_existing_text_region_merges_into_color_identity(self):
  s=ColorTopology(self.c);obj,d=self.helper.column(0,'E','right');s.update(self.helper.frame(0),[(obj,d),self.col(0,'blue')]);rid='region:unknown:paint_blue'
  self.assertEqual(len(s.regions),2)
  s.update(self.helper.frame(1),[self.col(1,'blue','E')]);self.assertEqual(set(s.regions),{rid});self.assertIn(obj,s.regions[rid]['anchors']);self.assertEqual(s.zone_by_object[obj],rid)
 def test_paint_excludes_neutral_base_and_lower_stripes(self):
  import numpy as np
  from prepare_color_inputs import pillar_paint
  hsv=np.zeros((720,1280,3));hsv[:,:,2]=.6;mask=np.zeros((720,1280),bool);mask[100:650,100:220]=True
  self.assertIsNone(pillar_paint(hsv,mask,[100,100,220,650],.9))
  hsv[500:650,100:220]=[.12,.8,.8]
  self.assertIsNone(pillar_paint(hsv,mask,[100,100,220,650],.9))
  hsv[100:430,100:220]=[.12,.8,.8]
  self.assertEqual(pillar_paint(hsv,mask,[100,100,220,650],.9)['color'],'yellow')
 def test_neutral_color_is_not_region(self):
  s=ColorTopology(self.c);self.assertEqual(s.update(self.helper.frame(0),[self.col(0,'gray')])['regions_after'],0)
if __name__=='__main__':unittest.main()
