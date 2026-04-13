import pytest
import torch
import torch.nn.functional as F


@pytest.fixture(scope="session")
def demo_data():
    labels = torch.tensor(
        [
            [[0, 0, 1, 1], [0, 2, 2, 1], [0, 2, 1, 1], [0, 0, 1, 2]],
            [[2, 2, 1, 1], [2, 2, 1, 0], [0, 1, 1, 0], [0, 0, 2, 2]],
        ],
        dtype=torch.int64,
    )
    labels_oh = F.one_hot(labels, num_classes=3).movedim(-1, 1).float()
    probs = labels_oh * 0.98 + (1 - labels_oh) * 0.01

    # Keep a few ambiguous pixels so the BA solver remains non-trivial on this tiny fixture.
    probs[0, :, 0, 0] = torch.tensor([0.80, 0.19, 0.01])
    probs[0, :, 1, 1] = torch.tensor([0.10, 0.05, 0.85])
    probs[1, :, 2, 2] = torch.tensor([0.10, 0.80, 0.10])
    probs[1, :, 0, 3] = torch.tensor([0.10, 0.80, 0.10])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return probs.to(device), labels.to(device)
