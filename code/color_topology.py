"""Causal scene-local paint identities, letter binding, and stable region IDs."""
import copy
from collections import defaultdict
from incremental_topology import IncrementalTopology

NAMES={'red':'红色','yellow':'黄色','blue':'蓝色','green':'绿色','purple':'紫色'}

class ColorTopology(IncrementalTopology):
 def __init__(self,c):
  super().__init__(c);self.bindings={};self.blocked=set();self.color_tokens={};self.letter_tokens={};self.binding_votes=defaultdict(lambda:defaultdict(set));self.binding_events=[];self.merge_events=[]
 def _merge(self,old,new):
  if old==new or old not in self.regions:return
  a=self.regions.pop(old);b=self.regions[new]
  for key in ['anchors','positions','evidence_frames','evidence_sources']:
   for v in a[key]:
    if v not in b[key]:b[key].append(v)
  self.evidence[new].extend(self.evidence.pop(old,[]))
  for e in self.evidence[new]:e['region']=new;e['zone']=new.split(':')[-1]
  for obj,rid in list(self.zone_by_object.items()):
   if rid==old:self.zone_by_object[obj]=new
  oldtoken=old.split(':')[-1];newtoken=new.split(':')[-1]
  for obj,zs in self.object_zones.items():
   if oldtoken in zs:zs.remove(oldtoken);zs.add(newtoken)
  for e in self.recent:
   if e['region']==old:e['region']=new;e['zone']=newtoken
  for pid,v in list(self.portals.items()):
   v['regions']=sorted({new if x==old else x for x in v['regions']})
   for e in v['side_evidence']:
    if e['region']==old:e['region']=new;e['zone']=newtoken
   if len(v['regions'])<2:del self.portals[pid]
  self.merge_events.append(dict(from_id=old,to_id=new))
 def update(self,frame,observations):
  before=set(self.regions);before_p=set(self.portals);events=[];merged_before=len(self.merge_events)
  # First collect direct text/paint coincidences from this frame only.
  letters_by_detection={}
  for obj,d in observations:
   if d['label']!='column':continue
   votes=[]
   for t in d.get('texts',[]):
    z=(t.get('parsed') or {}).get('zone');reviewed=t.get('review_status')=='assistant_visual_transcription_not_human_acceptance' and t.get('image_sha256')==frame['image_sha256'];score=t.get('score')
    if z in list('ABCDEFGH') and (reviewed or isinstance(score,(float,int)) and score>=.3):votes.append((z,reviewed or score>=.85))
   zs={z for z,strong in votes}
   if len(zs)!=1:continue
   letter=next(iter(zs));letters_by_detection[d['id']]=letter
   paint=d.get('pillar_color') or {};color=paint.get('color')
   if not color or paint.get('image_sha256')!=frame['image_sha256']:continue
   self.binding_votes[color][letter].add(frame['id'])
   strong=any(s for z,s in votes) or len(self.binding_votes[color][letter])>=2
   if not strong or color in self.blocked:continue
   other=next((k for k,v in self.bindings.items() if v==letter and k!=color),None)
   if color in self.bindings and self.bindings[color]!=letter or other:
    bad={color}|({other} if other else set())
    for k in bad:
     self.blocked.add(k);oldletter=self.bindings.pop(k,None)
     if oldletter:self.letter_tokens.pop(oldletter,None)
     rid='region:unknown:'+self.color_tokens.get(k,'paint_'+k)
     self.recent=[e for e in self.recent if e['region']!=rid]
     for oid,rr in list(self.zone_by_object.items()):
      if rr==rid:self.zone_by_object.pop(oid);self.object_zones[oid].clear()
     for pid,p in list(self.portals.items()):
      if rid in p['regions']:del self.portals[pid]
    events.append(dict(kind='binding_conflict',colors=sorted(bad),letter=letter,frame_id=frame['id'],detection_id=d['id']));continue
   if color not in self.bindings:
    token=self.color_tokens.get(color,self.letter_tokens.get(letter,letter if 'region:unknown:'+letter in self.regions else 'paint_'+color))
    self.color_tokens[color]=token
    existing=self.letter_tokens.get(letter,letter)
    self._merge('region:unknown:'+existing,'region:unknown:'+token) if 'region:unknown:'+token in self.regions else None
    self.letter_tokens[letter]=token;self.bindings[color]=letter
    events.append(dict(kind='color_bound_to_letter',color=color,letter=letter,region_id='region:unknown:'+token,frame_id=frame['id'],detection_id=d['id']))
  converted=[]
  for obj,d in observations:
   if d['label']!='column':converted.append((obj,d));continue
   dd=copy.deepcopy(d);paint=d.get('pillar_color') or {};color=paint.get('color');letter=letters_by_detection.get(d['id'])
   valid=color in NAMES and paint.get('image_sha256')==frame['image_sha256'] and color not in self.blocked
   if valid:
    token=self.color_tokens.setdefault(color,'paint_'+color)
    dd['texts']=[dict(text=NAMES[color],parsed={'zone':token},score=.9,source='pillar_paint_region',image_sha256=frame['image_sha256'])]
   elif letter and letter in self.letter_tokens and not any(v==letter for k,v in self.bindings.items() if k in self.blocked):
    token=self.letter_tokens[letter]
    for t in dd.get('texts',[]):
     if (t.get('parsed') or {}).get('zone')==letter:t['parsed']['zone']=token;t['source']='pillar_paint_region'
   converted.append((obj,dd))
  event=super().update(frame,converted)
  for color,token in self.color_tokens.items():
   rid='region:unknown:'+token
   if rid not in self.regions:continue
   r=self.regions[rid];letter=self.bindings.get(color)
   r.update(label=letter or NAMES[color],color=color,letter=letter,status='color_letter_bound' if letter else 'color_binding_conflict' if color in self.blocked else 'provisional_paint_region',binding_status='conflict' if color in self.blocked else 'bound' if letter else 'awaiting_letter')
  for p in self.portals.values():p['label']='↔'.join(self.regions[r]['label'] for r in p['regions'])+' 共享通道'
  # Color-only portals remain explicitly provisional; renaming never changes their endpoints.
  for p in self.portals.values():
   p['status']='trajectory_supported_connection_candidate' if all(self.regions[r].get('binding_status') in (None,'bound') for r in p['regions']) else 'color_supported_connection_candidate'
  self.binding_events.extend(events)
  event.update(regions_after=len(self.regions),region_ids=sorted(self.regions),new_regions=sorted(set(self.regions)-before),portals_after=len(self.portals),portal_ids=sorted(self.portals),new_portals=sorted(set(self.portals)-before_p),color_bindings=dict(self.bindings),color_binding_events=events,region_merges=self.merge_events[merged_before:],region_labels={rid:r['label'] for rid,r in self.regions.items()})
  return event

def make_topology(c):
 return ColorTopology(c) if c.get('color_regions') else IncrementalTopology(c)
