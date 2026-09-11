"""Measured reverse-curriculum spawn bank; no claim of dynamic stability."""
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[1]
m=mujoco.MjModel.from_xml_path(str(ROOT/'src/mjlab_microduck/robot/microduck/scene_rollers.xml'))
d=mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m,d,m.key('STAND').id)
base=d.qpos.copy(); jids=m.actuator_trnid[:,0]; qids=m.jnt_qposadr[jids]
home=base[qids].copy(); trunk=m.body('trunk_base').id
wids=[m.body(n).id for n in ('tire','tire_2','tire_3','tire_4')]
gids=[next(g for g in range(m.ngeom) if m.geom_bodyid[g]==b and m.geom_contype[g]) for b in wids]
verts=[]
for g in gids:
 mesh=m.geom_dataid[g]; start=m.mesh_vertadr[mesh]; count=m.mesh_vertnum[mesh]
 verts.append(m.mesh_vert[start:start+count].copy())

def put(x):
 d.qpos[:]=base;d.qpos[2]=x[0];d.qpos[3:7]=[np.cos(x[1]/2),np.sin(x[1]/2),0,0];d.qpos[qids]=x[2:]
 mujoco.mj_forward(m,d)

def geometry():
 pts=[v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g] for g,v in zip(gids,verts)]
 low=np.array([p[p[:,2].argmin()] for p in pts]); w=d.xpos[wids]
 # At nearly level camber, average the lowest points to avoid vertex-side jitter.
 support=np.array([p[p[:,2]<p[:,2].min()+.0002].mean(0) for p in pts[:2]]).mean(0)
 return low,w,support

lo=np.r_[.085,-.50,m.jnt_range[jids,0]+.045]
hi=np.r_[.185,.10,m.jnt_range[jids,1]-.045]
seed=json.loads((ROOT/'local/onefoot-seed.json').read_text())
x0=np.r_[seed['root_z'],seed['root_roll'],[seed['pose'][m.joint(int(j)).name] for j in jids]]
results={}
# Balance starts genuinely clear; preload and transfer start with both skates down.
for name,clearance,load_fraction in [('balance',.025,1.),('unload',0.,.95),('transfer_near',0.,.75),('transfer_mid',0.,.50),('transfer',0.,.0)]:
 def loss(x):
  put(x);low,w,support=geometry();com=d.subtree_com[trunk]
  target_y=load_fraction*support[1]+(1-load_fraction)*w[:,1].mean()
  return np.r_[low[:2,2]*180,(low[2:,2]-clearance)*160,
   (com[1]-target_y)*140,(com[0]-support[0])*80,
   (w[0,1]-w[1,1])*80,(w[2,1]-w[3,1])*50,
   (x[2:]-home)*.12,x[1]*.03]
 fit=least_squares(loss,x0,bounds=(lo,hi),max_nfev=1500,diff_step=1e-5)
 put(fit.x); low,w,support=geometry(); com=d.subtree_com[trunk]
 result=dict(scope='kinematic_spawn_only_not_stable_equilibrium',root_z=float(fit.x[0]),root_roll=float(fit.x[1]),
  pose={m.joint(int(j)).name:float(v) for j,v in zip(jids,fit.x[2:])},
  wheel_bottom_z_m=low[:,2].tolist(),wheel_centers=w.tolist(),com=com.tolist(),
  support_y=float(support[1]),com_support_offset_m=float(com[1]-support[1]),
  requested_load_fraction=load_fraction,cost=float(fit.cost))
 assert np.abs(low[:2,2]).max()<.0025,(name,result)
 assert np.abs(low[2:,2]-clearance).max()<.003,(name,result)
 results[name]=result;x0=fit.x
 print(name,'bottoms',low[:,2],'offset',result['com_support_offset_m'],'roll',result['root_roll'],flush=True)
path=ROOT/'local/onefoot-curriculum-poses.json';path.write_text(json.dumps(results,indent=2)+'\n')
