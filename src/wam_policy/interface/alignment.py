"""Physical-time alignment only; no dataset publication, image decoding or VAE cache."""
from __future__ import annotations
import copy
import numpy as np
from wam_policy.interface.validation import CAMERAS, require, validate_command, validate_frame


def align_actions(frames, commands, *, epoch, origin_ns, last_tick):
    per_camera={k:[] for k in CAMERAS}
    for row in frames:
        validate_frame(row,epoch=epoch,origin_ns=origin_ns,last_tick=last_tick)
        per_camera[row['camera_id']].append(row)
    require(type(last_tick) is int and last_tick>=0,'timing','last tick')
    require(all(len(per_camera[k])==last_tick//40+1 and all(r['capture_tick']==i*40 for i,r in enumerate(per_camera[k])) for k in CAMERAS), 'timing','complete synchronized RGB timeline required')
    expected=[r['capture_tick'] for r in per_camera[CAMERAS[0]]]
    by_start={};command_ids=set()
    for index,row in enumerate(commands):
        validate_command(row)
        require(row['command_id'] not in command_ids,'execution','duplicate command ID')
        command_ids.add(row['command_id'])
        require(row['start_tick'] not in by_start and row['start_tick']==index*10,'execution','command ordering')
        require(row['executed_until_tick']==max(row['start_tick'],min(last_tick,row['end_tick'])),'execution','command coverage')
        by_start[row['start_tick']]=(index,row)
    require(len(commands)*10>=last_tick,'execution','missing command coverage')
    t=1+(len(expected)-1)//4; n=1+4*(t-1)
    actions=np.zeros((9,t,16,1),np.float32);mask=np.zeros_like(actions,dtype=bool)
    sources=np.full((t,16),-1,np.int64);ids=[[None]*16 for _ in range(t)]
    for latent_index in range(1,t):
        for slot in range(16):
            tick=(latent_index-1)*160+slot*10
            require(tick in by_start,'timing','missing action interval')
            index,row=by_start[tick]
            require(row['status']=='completed','execution','image endpoint precedes uncompleted action')
            actions[:,latent_index,slot,0]=row['executed'];mask[:,latent_index,slot,0]=True
            sources[latent_index,slot]=index;ids[latent_index][slot]=row['command_id']
    endpoint=160*(t-1)
    return dict(actions=actions,actions_mask=mask,action_source_rows=sources,command_ids=ids,
                latent_endpoint_ticks=np.arange(t,dtype=np.int64)*160,
                rgb_source_indices=[[0]]+[list(range(1+4*i,5+4*i)) for i in range(t-1)],
                rgb_ticks=np.asarray(expected[:n],dtype=np.int64),
                unencoded_rgb_ticks=expected[n:],
                unencoded_command_ids=[r['command_id'] for r in commands if r['start_tick']>=endpoint],
                remainder_reason='preserved_unencoded_tail' if n<len(expected) or last_tick>endpoint else None)


def to_model_batch(aligned, *, latents, text_emb, normalizer):
    """Caller supplies frozen VAE/T5 outputs and explicit normalization statistics."""
    import torch
    from wam_policy.common.action_space import ActionNormalizer, validate_batch
    require(isinstance(normalizer,ActionNormalizer),'normalization','explicit ActionNormalizer required')
    raw=aligned['actions'];valid=aligned['actions_mask']
    encoded=normalizer.encode(raw[...,0].transpose(1,2,0)).transpose(2,0,1)[...,None]
    encoded=np.where(valid,encoded,0).astype(np.float32)
    batch=dict(latents=latents,actions=torch.from_numpy(encoded).unsqueeze(0).to(latents.device),
               actions_mask=torch.from_numpy(valid.copy()).unsqueeze(0).to(latents.device),text_emb=text_emb)
    validate_batch(batch)
    return batch
