"""Reconstructed five functional components: SKU LightGBM heads, PyTorch
multi-horizon joint network, macro LightGBM, expert router, quantity corrector.
Original archived weights/source are NOT included here.
"""
from __future__ import annotations
import os,random
from pathlib import Path
import numpy as np
import lightgbm as lgb
import torch
from torch import nn
import torch.nn.functional as nnf
from data_utils import HORIZON

def seeds(seed=42,threads=2):
    random.seed(seed); np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(threads)


def train_origins(o,max_origins=36,min_origin=26):
    # A training example is only permitted when *all nine target weeks*
    # precede the forecast being evaluated (or the final origin=134).
    candidates=np.arange(min_origin,o-HORIZON+1,dtype='int32')
    assert len(candidates)>=2,(o,candidates)
    if max_origins>0 and len(candidates)>max_origins:
        pos=np.linspace(0,len(candidates)-1,max_origins).round().astype(int)
        candidates=candidates[pos]
    assert np.all(candidates+HORIZON-1<o)
    return candidates


def training_xy(panel,feat,origins,kind):
    x=[];y=[]
    for t in origins:
        if kind=='sku': xx=feat.sku_tabular(int(t));yy=panel.S[t:t+HORIZON]
        elif kind=='macro':xx=feat.macro_tabular(int(t));yy=panel.G[t:t+HORIZON]
        else:raise ValueError(kind)
        x.append(xx);y.append(yy.reshape(-1))
    return np.vstack(x).astype('float32'),np.concatenate(y).astype('float32')


def boost(kind,rounds=140,threads=2,seed=42,leaves=22):
    common={'n_estimators':rounds,'learning_rate':.052,
      'num_leaves':leaves,'min_child_samples':300,'reg_lambda':12,
      'colsample_bytree':.90,'max_bin':127,'n_jobs':threads,
      'random_state':seed,'verbosity':-1}
    if kind=='occ':return lgb.LGBMClassifier(objective='binary',**common)
    if kind=='quantity':return lgb.LGBMRegressor(objective='poisson',**common)
    if kind=='macro':return lgb.LGBMRegressor(objective='poisson',**common)
    if kind=='router':return lgb.LGBMClassifier(objective='binary',**{**common,'num_leaves':14,'min_child_samples':350})
    if kind=='corrector':return lgb.LGBMRegressor(objective='tweedie',tweedie_variance_power=1.3,**{**common,'num_leaves':14})
    raise ValueError(kind)


class Joint(nn.Module):
    def __init__(self,nfeat,hidden=96):
        super().__init__()
        self.body=nn.Sequential(nn.Linear(nfeat,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU())
        self.p_head=nn.Linear(hidden,HORIZON)
        self.q_head=nn.Linear(hidden,HORIZON)
    def forward(self,x):
        h=self.body(x)
        logits=self.p_head(h)
        prob=torch.sigmoid(logits)
        qty=nnf.softplus(self.q_head(h)) + 1e-4
        return logits,prob,qty


def train_joint(panel,feat,origins,epochs=12,batch=2048,threads=2,seed=42,device='auto'):
    seeds(seed,threads)
    device=('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
    if device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but torch.cuda.is_available() is false. Install CUDA PyTorch or use --device cpu.')
    print('PyTorch NN training device:',device,flush=True)
    inputs=[];targets=[]
    for t in origins:
        inputs.append(feat.sku_nn(int(t)))
        targets.append(panel.S[t:t+HORIZON].T.copy())
    xx=np.vstack(inputs).astype('float32')
    yy=np.vstack(targets).astype('float32')
    mean=xx.mean(0);std=np.maximum(xx.std(0),1e-3)
    xx=np.clip((xx-mean)/std,-7,7)
    model=Joint(xx.shape[1]).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=1.7e-3,weight_decay=.015)
    tx=torch.from_numpy(xx);ty=torch.from_numpy(yy)
    n=len(xx)
    for e in range(epochs):
        model.train();order=torch.randperm(n);total=0
        for start in range(0,n,batch):
            ix=order[start:start+batch];x=tx[ix].to(device);y=ty[ix].to(device)
            logit,p,q=model(x)
            ispos=(y>0).float()
            bce=nnf.binary_cross_entropy_with_logits(logit,ispos)
            loss_positive=(nnf.smooth_l1_loss(torch.log1p(q),torch.log1p(y),reduction='none')*ispos).sum()/ispos.sum().clamp(min=1)
            loss_expect=nnf.smooth_l1_loss(torch.log1p(p*q),torch.log1p(y))
            loss=bce+.45*loss_positive+.35*loss_expect
            optimizer.zero_grad();loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),3)
            optimizer.step();total+=float(loss.detach())*len(ix)
        print(f'NN epoch {e+1}/{epochs}: loss={total/n:.5f}',flush=True)
    return {'network':model,'mean':mean,'std':std,'nfeat':xx.shape[1],'origins':list(map(int,origins))}


def predict_nn(obj,feat,origin,batch=8192):
    x=feat.sku_nn(origin)
    x=np.clip((x-obj['mean'])/obj['std'],-7,7)
    model=obj['network'].eval();device=next(model.parameters()).device;ps=[];qs=[]
    with torch.no_grad():
        for t in range(0,len(x),batch):
            _,p,q=model(torch.from_numpy(x[t:t+batch]).to(device))
            ps.append(p.cpu().numpy());qs.append(q.cpu().numpy())
    return np.vstack(ps).T.copy(),np.vstack(qs).T.copy()


def save_joint(obj,path):
    torch.save({'state_dict':obj['network'].state_dict(),
       'mean':obj['mean'],'std':obj['std'],'nfeat':obj['nfeat'],
       'origins':obj['origins']},path)


def load_joint(path):
    obj=torch.load(path,map_location='cpu',weights_only=False)
    net=Joint(obj['nfeat']);net.load_state_dict(obj['state_dict'])
    obj['network']=net
    return obj


def fit_experts(panel,feat,o,conf,save_dir=None):
    origins=train_origins(o,conf.max_train_origins,conf.min_train_origin)
    print(f'EXPERT asof origin={o} | training origins={len(origins)} | last target week={origins[-1]+8}',flush=True)
    x,y=training_xy(panel,feat,origins,'sku')
    occ=boost('occ',conf.rounds,conf.threads)
    occ.fit(x,(y>0).astype('int8'))
    pos=y>0
    qty=boost('quantity',conf.rounds,conf.threads)
    qty.fit(x[pos],y[pos]);del x,y,pos
    mx,my=training_xy(panel,feat,origins,'macro')
    macro=boost('macro',conf.macro_rounds,conf.threads,leaves=15)
    macro.fit(mx,my);del mx,my
    joint=train_joint(panel,feat,origins,conf.nn_epochs,conf.nn_batch,conf.threads,device=conf.device)
    if save_dir is not None:
        save_dir=Path(save_dir);save_dir.mkdir(parents=True,exist_ok=True)
        occ.booster_.save_model(str(save_dir/'sku_occurrence.txt'))
        qty.booster_.save_model(str(save_dir/'sku_positive_quantity.txt'))
        macro.booster_.save_model(str(save_dir/'macro.txt'))
        save_joint(joint,save_dir/'joint.pt')
    return occ,qty,macro,joint


def predict_experts(fitted,panel,feat,o):
    occ,qty,macro,joint=fitted
    tx=feat.sku_tabular(o)
    bp=occ.predict_proba(tx)[:,1].reshape(HORIZON,panel.N)
    bq=np.maximum(0,qty.predict(tx).reshape(HORIZON,panel.N))
    # reconstructed analog of the v3/v4 two-stage regime probability mask.
    # Precise v4 regime weights unavailable without original weights.
    base=np.where(bp>=.55,bp*bq,0).astype('float32')
    legacy=np.where(bp>=.60,bp*bq,0).astype('float32')
    mm=np.maximum(0,macro.predict(feat.macro_tabular(o)).reshape(HORIZON,panel.NG)).astype('float32')
    npb,nq=predict_nn(joint,feat,o)
    return {'base':base,'legacy':legacy,'prob':npb.astype('float32'),
            'qty':nq.astype('float32'),'macro':mm}


def macro_shape(p,macro):
    if p.sum()<=0:return p
    target=p.sum()*macro.sum(1)/max(macro.sum(),1e-5)
    weekly=p.sum(1)
    scale=np.clip((target+.05)/(weekly+.05),.65,1.50)
    return p*scale[:,None]


def baseline_v6(pred):
    b=pred['base'];p=pred['prob'];q=pred['qty'];m=pred['macro']
    mask=(b>0)&(p>=.45)
    return macro_shape(np.where(mask,.75*b+.25*p*q,0),m)


def two_experts(pred):
    b=pred['base'];p=pred['prob'];q=pred['qty'];m=pred['macro']
    mask=(b>0)&(p>=.45)
    return (macro_shape(np.where(mask,b,0),m),macro_shape(np.where(mask,p*q,0),m))


def train_meta(oof,feat,training_indices,threads=2,rounds=140):
    xrouter=[];yr=[];weights=[];xq=[];yq=[]
    for i in training_indices:
        f=oof[i];o=int(f['origin']);pred={k:f[k] for k in ('base','legacy','prob','qty','macro')}
        a,b=two_experts(pred)
        x=feat.meta(o,**{'base':pred['base'],'legacy':pred['legacy'],
          'prob':pred['prob'],'qty':pred['qty'],'macro':pred['macro']})
        y=f['truth'].ravel();a=a.ravel();b=b.ravel()
        sel=(a>0)&(np.abs(a-b)>.03)
        if sel.any():
            xrouter.append(x[sel]);yr.append((np.abs(y[sel]-a[sel])<np.abs(y[sel]-b[sel])).astype('int8'))
            weights.append(np.clip(np.abs(a[sel]-b[sel]),.08,15))
        selq=(f['base'].ravel()>0)
        xq.append(x[selq]);yq.append(y[selq])
    assert xrouter and xq
    router=boost('router',rounds,threads);router.fit(np.vstack(xrouter),np.concatenate(yr),sample_weight=np.concatenate(weights))
    calibrator=boost('corrector',rounds,threads);calibrator.fit(np.vstack(xq),np.concatenate(yq))
    return router,calibrator


def predict_meta(f,panel,feat,origin,router,calibrator):
    pred={k:f[k] for k in ('base','legacy','prob','qty','macro')}
    a,b=two_experts(pred)
    X=feat.meta(origin,**pred)
    confidence=router.predict_proba(X)[:,1].reshape(HORIZON,panel.N)
    w=np.clip(.75+.25*(confidence-.75),.01,.99)
    routed=a*w+b*(1-w)
    quantity=np.maximum(0,calibrator.predict(X).reshape(HORIZON,panel.N))
    disagreement=np.abs(np.log1p(pred['base'])-np.log1p(pred['prob']*pred['qty']))>=.50
    out={'v6_baseline':baseline_v6(pred),'v7_routed':routed}
    for alpha in (.15,.30):
        out[f'v7_hybrid_{int(100*alpha)}']=np.where(disagreement&(routed>0),
                      (1-alpha)*routed+alpha*quantity,routed)
    for k in out:
        assert np.isfinite(out[k]).all() and (out[k]>=0).all()
    return out
