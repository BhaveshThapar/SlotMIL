import json, torch, numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from slotmil.data.feature_cache import FeatureBagDataset, collate_bags
from slotmil.models.mil import build_model

sp = json.load(open('data/lidc/splits.json'))
al = json.load(open('runs/lidc/alignment_slot_div=0.5/alignment_slot.json'))
assign = {int(k): int(v) for k, v in al['assignment'].items()}
LES = [k for k, v in assign.items() if v == 0][0]
print(f"frozen assignment {assign} -> lesion slot {LES} (fit on VAL before test)")

ds = FeatureBagDataset('data/lidc/features_dinov2_vitb14.h5', keys=sp['test'], return_mask=True)
m = build_model(pooling='slot', input_dim=ds.dim, dim=256, num_classes=2,
                num_slots=8, readout='gated')
m.load_state_dict(torch.load('runs/lidc/slot_div=0.5/seed0/best.pt', map_location='cpu'))
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
m.eval().to(dev)

maxslot, frozen, oracle, apf, prev = [], [], [], [], []
with torch.no_grad():
    for b in DataLoader(ds, batch_size=1, collate_fn=collate_bags):
        n = int(b['lengths'][0]); tgt = (b['patch_target'][0,:n].numpy() > 0).astype(int)
        if tgt.sum() == 0 or tgt.sum() == n: continue
        a = m(b['features'].to(dev), b['pad_mask'].to(dev))['attn'][0,:,:n].float().cpu().numpy()
        maxslot.append(roc_auc_score(tgt, a.max(axis=0)))
        frozen.append(roc_auc_score(tgt, a[LES]))
        oracle.append(max(roc_auc_score(tgt, a[k]) for k in range(a.shape[0])))
        apf.append(average_precision_score(tgt, a[LES])); prev.append(tgt.mean())
f = lambda v: f"{np.mean(v):.4f} +/- {np.std(v):.4f}"
print(f"\nover {len(frozen)} annotated test bags:")
print(f"  max-over-slots (what I reported) : {f(maxslot)}")
print(f"  FROZEN lesion slot {LES}            : {f(frozen)}")
print(f"  oracle best slot (upper bound)   : {f(oracle)}")
print(f"\n  supervised probe ceiling         : 0.9102")
print(f"  avg precision (frozen)           : {np.mean(apf):.5f}  vs prevalence {np.mean(prev):.5f}"
      f"  = {np.mean(apf)/np.mean(prev):.1f}x lift")
