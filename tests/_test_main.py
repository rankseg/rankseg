import torch
import torch.nn.functional as F
from rankseg import RankSEG
from torchmetrics.segmentation import DiceScore
from torchmetrics.functional.segmentation import dice_score

probs = torch.load('./tests/data/demo_probs.pt')
labels = torch.load('./tests/data/demo_labels.pt')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
probs, labels = probs.to(device), labels.to(device)

labels_oh = F.one_hot(labels, num_classes=-1)
labels_oh = labels_oh.permute(0, -1, 1, 2)[:, :-1]

num_batch, num_class, *image_shape = probs.shape


rankseg_ba_obj = RankSEG(metric='dice', solver='BA')
preds = rankseg_ba_obj.predict(probs)
rankseg_rma_obj = RankSEG(metric='dice', solver='RMA', return_binary_masks=True)
preds_rma_overlap = rankseg_rma_obj.predict(probs)
rankseg_rma_obj = RankSEG(metric='dice', solver='RMA', return_binary_masks=False)
preds_rma = rankseg_rma_obj.predict(probs)
preds_rma = F.one_hot(preds_rma, num_classes=-1)
preds_rma = preds_rma.permute(0, -1, 1, 2)


dice_ba = dice_score(preds, labels_oh, num_classes=num_class, average='none').nanmean(dim=0)
dice_rma_overlap = dice_score(preds_rma_overlap, labels_oh, num_classes=num_class, average='none').nanmean(dim=0)
dice_rma = dice_score(preds_rma, labels_oh, num_classes=num_class, average='none').nanmean(dim=0)

print(f"{'Class':<8} {'pro':<10} {'Dice-BA':<20} {'Dice-RMA (overlap)':<25} {'Dice-RMA (non-overlap)':<20}")
for i in range(num_class):
    p = labels_oh[:,i].sum(dim=(-1,-2))
    pro = len(p[p>0]) / len(p)
    print(f"{i:<8} {pro:<10.2%} {dice_ba[i]:<20.4f} {dice_rma_overlap[i]:<25.4f} {dice_rma[i]:<20.4f}")

print("=" * 85)
print(f"{'Mean':<8} {'':<10} {dice_ba.nanmean():<20.4f} {dice_rma_overlap.nanmean():<25.4f} {dice_rma.nanmean():<20.4f}")
print("=" * 85)



