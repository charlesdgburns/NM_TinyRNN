"""Recompute free-choice train+validation and held-out NLL using the saved training blocks.

Each block resets to the model's learned initial state. Evaluation is vectorized
over saved models and checked against their saved best validation scores.
"""
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
from NM_TinyRNN.code.measures import mechanistic_variability as mv
from NM_TinyRNN.code.models import datasets as ds

RAW=['forced_choice','choice','outcome','session_folder_name','sequence_block_idx','block_trial_idx']

def score_group(frame):
    torch.set_num_threads(1)
    frame=frame.sort_values(['outer_loop_n','inner_loop_idx']).reset_index(drop=True)
    first=pd.read_csv(frame.iloc[0].trials_data_path,sep='\t',usecols=RAW)
    kept=first.sequence_block_idx.notna().to_numpy()
    widths=first.loc[kept].groupby('sequence_block_idx').size().unique()
    if len(widths)!=1 or widths[0]<2:
        raise ValueError('Saved training blocks must have a single complete sequence length')
    width=int(widths[0])
    positions=np.flatnonzero(kept).reshape(-1,width)
    assert np.all(first.block_trial_idx.to_numpy()[positions]%width==np.arange(width))
    assert all(len(np.unique(first.sequence_block_idx.to_numpy()[indices]))==1 for indices in positions)
    assert all(first.session_folder_name.iloc[indices].nunique()==1 for indices in positions)
    raw=torch.tensor(first[['forced_choice','choice','outcome']].to_numpy()[positions],dtype=torch.float32)
    target=raw[:,1:,1].long();free=raw[:,1:,0].eq(0).numpy()
    models=[];masks=[];val_masks=[];eval_masks=[];continuous=[]
    for row in frame.itertuples():
        data=pd.read_csv(row.trials_data_path,sep='\t',usecols=RAW+['split','prob_A','prob_B'])
        pd.testing.assert_frame_equal(first,data[RAW])
        labels=data['split'].to_numpy()[positions[:,1:]]
        mask=np.isin(labels,['train','val'])&free
        assert mask.sum()>0
        masks.append(mask)
        val_masks.append((labels=='val')&free)
        eval_masks.append((labels=='eval')&free)
        probabilities=data[['prob_A','prob_B']].to_numpy()[positions[:,:-1]]
        correct=np.take_along_axis(probabilities,target.numpy()[...,None],axis=-1)[...,0]
        continuous.append(float(-np.log(np.clip(correct[mask],np.finfo(float).tiny,1)).mean()))
        with open(row.model_pickle_path,'rb') as handle:
            models.append(mv.CPUUnpickler(handle).load().cpu().eval())
    inputs=ds.input_encoder(raw[:,:-1],models[0].input_encoding,models[0].input_forced_choice)
    parameters,buffers=torch.func.stack_module_state(models)
    template=deepcopy(models[0]).to('meta')
    def forward(p,b,x):
        return torch.func.functional_call(template,(p,b),(x,))[0]
    sums=np.zeros(len(models));val_sums=np.zeros(len(models));val_masks=np.stack(val_masks)
    eval_sums=np.zeros(len(models));eval_masks=np.stack(eval_masks);eval_counts=eval_masks.sum(axis=(1,2))
    counts=np.stack(masks).sum(axis=(1,2))
    masks=np.stack(masks)
    with torch.no_grad():
        for start in range(0,len(inputs),8):
            x=inputs[start:start+8]
            logits=torch.vmap(forward,in_dims=(0,0,None))(parameters,buffers,x)
            expected=target[start:start+8][None,...,None].expand(len(models),-1,-1,-1)
            loss=-logits.log_softmax(-1).gather(-1,expected).squeeze(-1).double().numpy()
            sums+=(loss*masks[:,start:start+8]).sum(axis=(1,2))
            val_sums+=(loss*val_masks[:,start:start+8]).sum(axis=(1,2))
            eval_sums+=(loss*eval_masks[:,start:start+8]).sum(axis=(1,2))
            if start==0:
                direct=models[0](x)[0]
                torch.testing.assert_close(logits[0],direct,atol=2e-6,rtol=2e-6)
    frame['block_train_val_CE']=sums/counts
    frame['block_val_CE']=val_sums/val_masks.sum(axis=(1,2))
    # Older exports can predate trial-weighted validation; retain their freshly
    # computed score rather than requiring equality with the old saved metric.
    if 'sequence_length' in frame and frame.sequence_length.notna().all():
        np.testing.assert_allclose(frame.block_val_CE,frame.best_val_CE,atol=8e-5,rtol=8e-5)
    frame['block_eval_CE']=np.divide(eval_sums,eval_counts,out=np.full_like(eval_sums,np.nan),where=eval_counts>0)
    frame['block_eval_n_free']=eval_counts
    frame['continuous_matched_CE']=continuous
    frame['block_train_val_n_free']=counts
    return frame[['info_path','block_train_val_CE','block_val_CE','continuous_matched_CE','block_train_val_n_free','block_eval_CE','block_eval_n_free']]
