import torch
import torch.nn.functional as F
from rankseg import RankSEG

probs = torch.load('./tests/data/demo_probs.pt')
labels = torch.load('./tests/data/demo_labels.pt')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
probs, labels = probs.to(device), labels.to(device)

rankseg_ba_obj = RankSEG(metric='dice', solver='BA')
preds = rankseg_ba_obj.predict(probs)

rankseg_rma_obj = RankSEG(metric='dice', solver='RMA', allow_overlap=True)
preds_rma_overlap = rankseg_rma_obj.predict(probs)
rankseg_rma_obj = RankSEG(metric='dice', solver='RMA', allow_overlap=False)
preds_rma = rankseg_rma_obj.predict(probs)

print("=" * 70)
print(f"{'Class':<8} {'RankSEG-BA':<15} {'RankSEG-RMA (overlap)':<15} {'RankSEG-RMA (non-overlap)':<15}")
print("=" * 70)

dice_ba_list = []
dice_rma_overlap_list = []
dice_rma_list = []

for k in range(21):
    # RankSEG-BA
    tp_new = ((preds[:, k] == 1) & (labels == k)).sum(dim=(-2,-1)).float()
    p = (labels == k).sum(dim=(-2,-1)).float()
    t_new = (preds[:, k] == 1).sum(dim=(-2,-1)).float()
    dice_new = (2*tp_new) / (t_new + p + 1e-8)
    dice_ba_mean = dice_new.mean().item()
    
    # RankSEG-RMA (overlap)
    tp_old = ((preds_rma_overlap[:, k] == 1) & (labels == k)).sum(dim=(-2,-1)).float()
    t_old = (preds_rma_overlap[:, k] == 1).sum(dim=(-2,-1)).float()
    dice_old = (2*tp_old) / (t_old + p + 1e-8)
    dice_rma_overlap_mean = dice_old.mean().item()

    # RankSEG-RMA (non-overlap)
    tp_rma = ((preds_rma == k) & (labels == k)).sum(dim=(-2,-1)).float()
    p_rma = (labels == k).sum(dim=(-2,-1)).float()
    t_rma = (preds_rma == k).sum(dim=(-2,-1)).float()
    dice_rma = (2*tp_rma) / (t_rma + p_rma + 1e-8)
    dice_rma_mean = dice_rma.mean().item()

    dice_ba_list.append(dice_ba_mean)
    dice_rma_overlap_list.append(dice_rma_overlap_mean)
    dice_rma_list.append(dice_rma_mean)

    print(f"{k:<8} {dice_ba_mean:<15.4f} {dice_rma_overlap_mean:<15.4f} {dice_rma_mean:<15.4f}")

print("=" * 70)
print(f"{'Mean':<8} {sum(dice_ba_list)/21:<15.4f} {sum(dice_rma_overlap_list)/21:<15.4f} {sum(dice_rma_list)/21:<15.4f}")
print("=" * 70)
