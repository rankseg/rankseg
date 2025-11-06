import torch
import torch.nn.functional as F
from rankseg import rankdice_ba, rank_dice

probs = torch.load('./tests/data/demo_probs.pt')
labels = torch.load('./tests/data/demo_labels.pt')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
probs, labels = probs.to(device), labels.to(device)

preds = rankdice_ba(probs, solver='BA', eps=1e-4)

preds_old, _, _ = rank_dice(output=probs, device=probs.device, app=2, smooth=0.0)

print("=" * 70)
print(f"{'Class':<8} {'New Method':<15} {'Old Method':<15} {'Difference':<15}")
print("=" * 70)

dice_new_list = []
dice_old_list = []

for k in range(21):
    # New method
    tp_new = ((preds[:, k] == 1) & (labels == k)).sum(dim=(-2,-1)).float()
    p = (labels == k).sum(dim=(-2,-1)).float()
    t_new = (preds[:, k] == 1).sum(dim=(-2,-1)).float()
    dice_new = (2*tp_new) / (t_new + p + 1e-8)
    dice_new_mean = dice_new.mean().item()
    
    # Old method
    tp_old = ((preds_old[:, k] == 1) & (labels == k)).sum(dim=(-2,-1)).float()
    t_old = (preds_old[:, k] == 1).sum(dim=(-2,-1)).float()
    dice_old = (2*tp_old) / (t_old + p + 1e-8)
    dice_old_mean = dice_old.mean().item()
    
    diff = dice_new_mean - dice_old_mean
    
    dice_new_list.append(dice_new_mean)
    dice_old_list.append(dice_old_mean)
    
    print(f"{k:<8} {dice_new_mean:<15.4f} {dice_old_mean:<15.4f} {diff:+.4f}")

print("=" * 70)
print(f"{'Mean':<8} {sum(dice_new_list)/21:<15.4f} {sum(dice_old_list)/21:<15.4f} {(sum(dice_new_list)-sum(dice_old_list))/21:+.4f}")
print("=" * 70)