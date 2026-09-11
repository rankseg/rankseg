import pytest
import torch
import torch.nn.functional as F

import rankseg._rankseg_algo as rankseg_algo
from rankseg._rankseg_algo import rankdice_ba, rankseg_rma
from rankseg.distribution import RefinedNormalPB


def _demo_probs():
    labels = torch.tensor(
        [
            [[0, 0, 1, 1], [0, 2, 2, 1], [0, 2, 1, 1], [0, 0, 1, 2]],
            [[2, 2, 1, 1], [2, 2, 1, 0], [0, 1, 1, 0], [0, 0, 2, 2]],
        ],
        dtype=torch.int64,
    )
    labels_oh = F.one_hot(labels, num_classes=3).movedim(-1, 1).float()
    probs = labels_oh * 0.98 + (1 - labels_oh) * 0.01
    probs[0, :, 0, 0] = torch.tensor([0.80, 0.19, 0.01])
    probs[0, :, 1, 1] = torch.tensor([0.10, 0.05, 0.85])
    probs[1, :, 2, 2] = torch.tensor([0.10, 0.80, 0.10])
    probs[1, :, 0, 3] = torch.tensor([0.10, 0.80, 0.10])
    return probs


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        (torch.device("cpu"), False),
        (torch.device("cuda"), True),
        (torch.device("mps"), True),
        (torch.device("xpu"), True),
    ],
)
def test_rankdice_refined_normal_device_policy_covers_all_accelerators(device, expected):
    assert rankseg_algo._uses_host_refined_normal(device) is expected


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_selected_pixel_count_preserves_working_dtype(dtype):
    mask = torch.tensor([[[True, False, True], [False, True, True]]])

    counts = rankseg_algo._count_selected_pixels(mask, dtype)

    assert counts.dtype == dtype
    assert torch.equal(counts, torch.tensor([[[2.0], [2.0]]], dtype=dtype))


def test_selected_pixel_count_is_exact_above_float32_integer_limit():
    num_selected = 2**24 + 1
    mask = torch.ones((1, 1, num_selected), dtype=torch.bool)

    count = rankseg_algo._count_selected_pixels(mask, torch.float64)

    assert count.item() == num_selected
    assert mask.float().sum().item() != num_selected


@pytest.mark.parametrize("solver", ["BA", "TRNA", "BA+TRNA"])
@pytest.mark.parametrize(
    "device_type",
    [
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available"),
        ),
        pytest.param(
            "xpu",
            marks=pytest.mark.skipif(
                not (hasattr(torch, "xpu") and torch.xpu.is_available()),
                reason="XPU is not available",
            ),
        ),
    ],
)
def test_rankdice_optional_accelerator_matches_cpu(device_type, solver):
    probs = torch.tensor([[[0.8, 0.2, 0.6, 0.4]]], dtype=torch.float32)
    expected = rankdice_ba(probs, solver=solver, smooth=0.2, pruning_prob=0.0)

    preds = rankdice_ba(
        probs.to(device_type),
        solver=solver,
        smooth=0.2,
        pruning_prob=0.0,
    )

    assert preds.device.type == device_type
    assert torch.equal(preds.cpu(), expected)


def _rankdice_formula_oracle(probs, solver, smooth, eps=1e-4):
    """Evaluate the paper's BA/TRNA scalar formulas without solver code."""
    working_probs = probs.float() if probs.dtype in (torch.float16, torch.bfloat16) else probs
    sorted_prob, top_index = torch.sort(working_probs, descending=True)
    dim = sorted_prob.numel()
    cumsum_prob = sorted_prob.cumsum(0)
    if dim == 1:
        up_tau = 1
    else:
        search_denom = torch.arange(1, dim) + smooth + dim
        if smooth > 1e6:
            stop_search = cumsum_prob[:-1] / search_denom >= sorted_prob[1:]
        else:
            stop_search = cumsum_prob[:-1] >= search_denom * sorted_prob[1:]
        up_tau = int(torch.where(stop_search)[0][0].item()) + 1 if bool(stop_search.any()) else dim

    var_eps = torch.finfo(sorted_prob.dtype).eps
    pb_mean = sorted_prob.sum()
    pb_var = torch.sum(sorted_prob * (1 - sorted_prob))
    pb_var_safe = torch.clamp(pb_var, min=var_eps)
    pb_m3 = torch.sum(sorted_prob * (1 - sorted_prob) * (1 - 2 * sorted_prob))
    pb_skew = pb_m3 / pb_var_safe ** (3 / 2)
    rv = RefinedNormalPB(dim=dim, loc=pb_mean, scale=pb_var_safe.sqrt(), skew=pb_skew)
    lower, upper = rv.interval(eps)
    support = torch.arange(int(lower.item()), int(upper.item()) + 1)
    full_pmf = rv.pmf(support)
    full_pmf = full_pmf / full_pmf.sum()

    use_scaled_scores = smooth > 1e6
    scores = torch.empty(up_tau + 1, dtype=sorted_prob.dtype, device=sorted_prob.device)
    if use_scaled_scores:
        scores[0] = -torch.sum(support / (1.0 + support / smooth) * full_pmf)
    else:
        scores[0] = smooth * torch.sum(full_pmf / (support + smooth)) if smooth > 0 else 0.0
    if solver == "BA":
        for tau in range(1, up_tau + 1):
            if use_scaled_scores:
                omega = 2 * cumsum_prob[tau - 1] * torch.sum(full_pmf / (1.0 + (tau + support + 1) / smooth))
                nu_offset = tau + support
                nu = -torch.sum(nu_offset / (1.0 + nu_offset / smooth) * full_pmf)
            else:
                omega = 2 * cumsum_prob[tau - 1] * torch.sum(full_pmf / (tau + support + smooth + 1))
                nu = smooth * torch.sum(full_pmf / (tau + support + smooth))
            scores[tau] = omega + nu
    else:
        omega_weights = torch.zeros_like(full_pmf)
        for tau in range(1, up_tau + 1):
            prob = sorted_prob[tau - 1]
            excluded_var = torch.clamp(pb_var - prob * (1 - prob), min=var_eps)
            excluded_m3 = pb_m3 - prob * (1 - prob) * (1 - 2 * prob)
            excluded_rv = RefinedNormalPB(
                dim=dim - 1,
                loc=pb_mean - prob,
                scale=excluded_var.sqrt(),
                skew=excluded_m3 / excluded_var ** (3 / 2),
            )
            excluded_pmf = excluded_rv.pmf(support)
            excluded_pmf = excluded_pmf / excluded_pmf.sum()
            omega_weights += prob * excluded_pmf
            if use_scaled_scores:
                omega = torch.sum(2 * omega_weights / (1.0 + (tau + support + 1) / smooth))
                nu_offset = tau + support
                nu = -torch.sum(nu_offset / (1.0 + nu_offset / smooth) * full_pmf)
            else:
                omega = torch.sum(2 * omega_weights / (tau + support + smooth + 1))
                nu = smooth * torch.sum(full_pmf / (tau + support + smooth))
            scores[tau] = omega + nu

    expected = torch.zeros(dim, dtype=torch.bool, device=sorted_prob.device)
    expected[top_index[: int(scores.argmax().item())]] = True
    return expected


def _rma_dice_multiclass_objective_oracle(probs, smooth, pruning_prob=0.0):
    """Resolve RMA masks by directly subtracting its before/after Dice objectives."""
    masks = rankseg_rma(
        probs,
        metric="dice",
        smooth=smooth,
        output_mode="multilabel",
        pruning_prob=pruning_prob,
    ).flatten(2)
    working_probs = probs.flatten(2).to(torch.float64)
    class_counts = masks.sum(dim=1)
    safe_to_predict = masks & (class_counts == 1).unsqueeze(1)
    mu = (working_probs * safe_to_predict).sum(dim=2, keepdim=True)
    tau = safe_to_predict.sum(dim=2, keepdim=True).to(torch.float64)
    pb_mean = working_probs.sum(dim=2, keepdim=True)
    volume_and_mean = tau + pb_mean

    if smooth == 0:
        score_before = 2 * mu / (volume_and_mean + 1)
        score_after = 2 * (mu + working_probs) / (volume_and_mean + 2)
    else:
        # This is smooth * score with the common additive constant removed.
        # It is algebraically equivalent to the documented RMA Dice objective
        # but remains distinguishable when smooth is much larger than the mask.
        score_before = 2 * mu / (1 + (volume_and_mean + 1) / smooth)
        score_before -= volume_and_mean / (1 + volume_and_mean / smooth)
        score_after = 2 * (mu + working_probs) / (1 + (volume_and_mean + 2) / smooth)
        score_after -= (volume_and_mean + 1) / (1 + (volume_and_mean + 1) / smooth)
    increments = score_after - score_before

    active_classes = working_probs.amax(dim=2) > pruning_prob
    unassigned_eligible_classes = active_classes | ~active_classes.any(dim=1, keepdim=True)
    eligible_classes = torch.where(
        (class_counts > 0).unsqueeze(1),
        masks,
        unassigned_eligible_classes.unsqueeze(2),
    )
    return torch.where(eligible_classes, increments, float("-inf")).argmax(dim=1)


def test_rankseg_rma_binary_multiclass_returns_foreground_mask():
    probs = torch.tensor(
        [
            [
                [[0.80, 0.20], [0.30, 0.70]],
                [[0.20, 0.80], [0.70, 0.30]],
            ]
        ],
        dtype=torch.float32,
    )

    preds = rankseg_rma(probs, metric="dice", output_mode="multiclass", pruning_prob=0.0)

    assert preds.shape == (1, 2, 2)
    assert preds.dtype == torch.int64
    assert torch.equal(preds, torch.tensor([[[0, 1], [1, 0]]]))


def test_rankseg_rma_pruning_can_return_all_zero_masks():
    probs = torch.full((1, 3, 2, 2), 0.2, dtype=torch.float32)

    preds = rankseg_rma(probs, metric="iou", output_mode="multilabel", pruning_prob=0.5)

    assert preds.shape == probs.shape
    assert preds.dtype == torch.bool
    assert int(preds.sum().item()) == 0


def test_pruning_skips_classes_at_threshold_for_ba_and_rma():
    probs = torch.full((1, 1, 2, 2), 0.5, dtype=torch.float32)

    preds_ba = rankdice_ba(probs, solver="BA", pruning_prob=0.5)
    preds_rma = rankseg_rma(probs, metric="dice", output_mode="multilabel", pruning_prob=0.5)

    assert int(preds_ba.sum().item()) == 0
    assert int(preds_rma.sum().item()) == 0


def test_rankseg_rma_multiclass_unassigned_policy_max_score():
    probs = torch.tensor(
        [
            [
                [[0.95, 0.01, 0.01, 0.01]],
                [[0.01, 0.95, 0.95, 0.10]],
                [[0.01, 0.55, 0.55, 0.01]],
            ]
        ],
        dtype=torch.float32,
    )

    preds = rankseg_rma(probs, metric="dice", output_mode="multiclass", pruning_prob=0.5)

    assert preds.shape == (1, 1, 4)
    assert preds.dtype == torch.int64
    assert torch.equal(preds[0, 0, :3], torch.tensor([0, 1, 1]))
    assert preds[0, 0, 3].item() == 1


@pytest.mark.parametrize("metric", ["dice", "iou"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")),
    ],
)
def test_rankseg_rma_overlap_winner_selected_the_pixel(metric, device):
    # At pixel 6, classes 1 and 2 select the pixel. Class 0 has the largest
    # unconstrained increment, so the old all-active resolution incorrectly
    # returned class 0 even though its binary mask did not select that pixel.
    probs = torch.tensor(
        [
            [
                [0.189101, 0.524612, 0.277966, 0.124355, 0.165713, 0.002576, 0.138480, 0.119291],
                [0.175384, 0.374397, 0.685354, 0.821371, 0.578314, 0.038970, 0.356818, 0.039937],
                [0.635515, 0.100990, 0.036681, 0.054274, 0.255974, 0.958454, 0.504701, 0.840771],
            ]
        ],
        dtype=torch.float32,
        device=device,
    )

    binary_masks = rankseg_rma(probs, metric=metric, output_mode="multilabel", pruning_prob=0.5)
    preds = rankseg_rma(probs, metric=metric, output_mode="multiclass", pruning_prob=0.5)

    overlap_mask = binary_masks.sum(dim=1) > 1
    winner_selected_pixel = binary_masks.gather(1, preds.unsqueeze(1)).squeeze(1)
    assert bool(overlap_mask.any())
    assert bool(winner_selected_pixel[overlap_mask].all())
    assert torch.equal(binary_masks[0, :, 6], torch.tensor([False, True, True], device=device))
    assert preds[0, 6].item() == 2


@pytest.mark.parametrize(
    ("smooth", "seed"),
    [
        (0.0, 354),
        (0.01, 1378),
        (0.2, 354),
        (1.0, 1500),
        (10.0, 357),
        (1e8, 354),
        (1e308, 354),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")),
    ],
)
def test_rankseg_rma_dice_multiclass_matches_increment_of_its_objective(smooth, seed, dtype, device):
    generator = torch.Generator().manual_seed(seed)
    probs = torch.softmax(torch.randn((2, 4, 16), generator=generator), dim=1).to(dtype=dtype, device=device)
    expected = _rma_dice_multiclass_objective_oracle(probs, smooth)

    preds = rankseg_rma(
        probs,
        metric="dice",
        smooth=smooth,
        output_mode="multiclass",
        pruning_prob=0.0,
    ).flatten(1)

    assert torch.equal(preds, expected)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")),
    ],
)
def test_rankseg_rma_positive_smooth_overlap_regression(device):
    # The legacy formula added smooth twice and chose class 2 at pixel 11;
    # the actual RMA Dice objective increment selects class 0.
    generator = torch.Generator().manual_seed(354)
    probs = torch.softmax(torch.randn((2, 4, 16), generator=generator), dim=1).to(device)
    binary_masks = rankseg_rma(
        probs,
        metric="dice",
        smooth=0.2,
        output_mode="multilabel",
        pruning_prob=0.0,
    )

    preds = rankseg_rma(
        probs,
        metric="dice",
        smooth=0.2,
        output_mode="multiclass",
        pruning_prob=0.0,
    )

    assert torch.equal(binary_masks[0, :, 11], torch.tensor([True, False, True, True], device=device))
    assert preds[0, 11].item() == 0


@pytest.mark.parametrize("metric", ["dice", "iou"])
@pytest.mark.parametrize("smooth", [0.0, 0.2, 1e8])
def test_rankseg_rma_multiclass_winner_respects_nonempty_binary_masks(metric, smooth):
    generator = torch.Generator().manual_seed(20260902)
    probs = torch.softmax(torch.randn((8, 5, 4, 5), generator=generator), dim=1)

    binary_masks = rankseg_rma(
        probs,
        metric=metric,
        smooth=smooth,
        output_mode="multilabel",
        pruning_prob=0.5,
    )
    preds = rankseg_rma(
        probs,
        metric=metric,
        smooth=smooth,
        output_mode="multiclass",
        pruning_prob=0.5,
    )

    has_selected_class = binary_masks.any(dim=1)
    winner_selected_pixel = binary_masks.gather(1, preds.unsqueeze(1)).squeeze(1)
    assert bool(winner_selected_pixel[has_selected_class].all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("dim", [3, 4])
@pytest.mark.parametrize("output_mode", ["multilabel", "multiclass"])
def test_rankseg_rma_cuda_mask_paths_match_cpu_across_vectorization_boundary(monkeypatch, dim, output_mode):
    # Use a small test-only boundary so both CUDA paths are exercised without
    # allocating tensors at the production 524,288-pixel threshold.
    monkeypatch.setattr(rankseg_algo, "_RMA_CUDA_VECTORIZED_MASK_MAX_DIM", 3)
    probs = torch.tensor(
        [
            [
                [0.8, 0.2, 0.6, 0.4],
                [0.2, 0.8, 0.4, 0.6],
                [0.5, 0.5, 0.5, 0.5],
            ]
        ],
        dtype=torch.float32,
    )[..., :dim]
    expected = rankseg_rma(
        probs,
        metric="dice",
        output_mode=output_mode,
        pruning_prob=0.5,
    )

    preds = rankseg_rma(
        probs.cuda(),
        metric="dice",
        output_mode=output_mode,
        pruning_prob=0.5,
    )

    assert torch.equal(preds.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_rankseg_rma_cuda_single_channel_keeps_low_memory_mask_path():
    probs = torch.tensor([[[0.8, 0.2, 0.6, 0.4]]], dtype=torch.float32)
    expected = rankseg_rma(probs, metric="dice", output_mode="multilabel", pruning_prob=0.0)

    preds = rankseg_rma(probs.cuda(), metric="dice", output_mode="multilabel", pruning_prob=0.0)

    assert torch.equal(preds.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("output_mode", ["multilabel", "multiclass"])
def test_rankseg_rma_cuda_vectorized_mask_path_allows_empty_batches(output_mode):
    probs = torch.empty((0, 2, 11), dtype=torch.float32, device="cuda")

    preds = rankseg_rma(probs, metric="dice", output_mode=output_mode)

    expected_shape = probs.shape if output_mode == "multilabel" else (0, 11)
    assert preds.shape == expected_shape
    assert preds.device.type == "cuda"


@pytest.mark.parametrize("metric", ["dice", "iou"])
def test_rankseg_rma_all_classes_pruned_uses_unmasked_increment_score(metric):
    probs = torch.tensor(
        [
            [
                [[0.45, 0.50, 0.50, 0.50]],
                [[0.40, 0.25, 0.25, 0.25]],
                [[0.15, 0.25, 0.25, 0.25]],
            ]
        ],
        dtype=torch.float32,
    )

    preds = rankseg_rma(probs, metric=metric, output_mode="multiclass", pruning_prob=0.5)

    assert torch.equal(probs.argmax(dim=1), torch.tensor([[[0, 0, 0, 0]]]))
    assert torch.equal(preds, torch.tensor([[[1, 0, 0, 0]]]))


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("class_order", [(0, 1, 2), (2, 0, 1), (0, 2, 1)])
def test_rankseg_rma_iou_all_classes_pruned_zero_class_uses_finite_increment_scores(class_order, dtype, device):
    if device == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA bfloat16 is not supported")

    base_probs = torch.tensor(
        [
            [0.40, 0.30, 0.20, 0.10],
            [0.10, 0.20, 0.30, 0.40],
            [0.00, 0.00, 0.00, 0.00],
        ],
        dtype=dtype,
        device=device,
    )
    probs = base_probs[list(class_order)].unsqueeze(0)
    original_expected = (0, 0, 1, 1)
    expected = torch.tensor(
        [[class_order.index(class_id) for class_id in original_expected]],
        dtype=torch.int64,
        device=device,
    )

    preds = rankseg_rma(
        probs,
        metric="iou",
        output_mode="multiclass",
        pruning_prob=0.5,
        smooth=0,
    )

    assert torch.equal(preds, expected)


def test_rankseg_rma_all_classes_pruned_fallback_is_per_sample():
    probs = torch.tensor(
        [
            [
                [[0.45, 0.50, 0.50, 0.50]],
                [[0.40, 0.25, 0.25, 0.25]],
                [[0.15, 0.25, 0.25, 0.25]],
            ],
            [
                [[0.95, 0.01, 0.01, 0.01]],
                [[0.01, 0.95, 0.95, 0.10]],
                [[0.01, 0.55, 0.55, 0.01]],
            ],
        ],
        dtype=torch.float32,
    )

    preds = rankseg_rma(probs, metric="dice", output_mode="multiclass", pruning_prob=0.5)

    assert torch.equal(preds, torch.tensor([[[1, 0, 0, 0]], [[0, 1, 1, 1]]]))


def test_rankseg_rma_all_classes_pruned_void_policy_returns_void_index():
    probs = torch.full((1, 3, 2, 2), 0.2, dtype=torch.float32)

    preds = rankseg_rma(
        probs,
        metric="dice",
        output_mode="multiclass",
        pruning_prob=0.5,
        unassigned_policy="void",
        void_index=99,
    )

    assert torch.equal(preds, torch.full((1, 2, 2), 99, dtype=torch.int64))


def test_rankseg_rma_multiclass_unassigned_policy_void():
    probs = torch.tensor(
        [
            [
                [[0.95, 0.01, 0.01, 0.01]],
                [[0.01, 0.95, 0.95, 0.10]],
                [[0.01, 0.55, 0.55, 0.01]],
            ]
        ],
        dtype=torch.float32,
    )

    preds = rankseg_rma(
        probs,
        metric="dice",
        output_mode="multiclass",
        pruning_prob=0.5,
        unassigned_policy="void",
        void_index=255,
    )

    assert torch.equal(preds, torch.tensor([[[0, 1, 1, 255]]]))


def test_rankseg_rma_rejects_unknown_metric():
    probs = _demo_probs()

    with pytest.raises(ValueError, match="metric should be iou or dice"):
        rankseg_rma(probs, metric="f1", output_mode="multiclass")


def test_rankseg_rma_rejects_unknown_output_mode():
    probs = _demo_probs()

    with pytest.raises(ValueError, match="output_mode should be multiclass or multilabel"):
        rankseg_rma(probs, metric="dice", output_mode="overlap")


def test_rankseg_rma_rejects_unknown_unassigned_policy():
    probs = _demo_probs()

    with pytest.raises(ValueError, match="unassigned_policy should be max_score or void"):
        rankseg_rma(probs, metric="dice", output_mode="multiclass", unassigned_policy="ignore")


def test_rankseg_rma_rejects_void_policy_for_multilabel_output():
    probs = _demo_probs()

    with pytest.raises(ValueError, match="only supported when output_mode='multiclass'"):
        rankseg_rma(
            probs,
            metric="dice",
            output_mode="multilabel",
            unassigned_policy="void",
        )


@pytest.mark.parametrize("output_mode", ["multiclass", "multilabel"])
def test_rankseg_rma_rejects_unused_nondefault_void_index(output_mode):
    probs = _demo_probs()

    with pytest.raises(ValueError, match="non-default void_index requires"):
        rankseg_rma(
            probs,
            metric="dice",
            output_mode=output_mode,
            unassigned_policy="max_score",
            void_index=99,
        )


def test_rankseg_rma_accepts_normalized_multiclass_void_configuration():
    probs = torch.full((1, 2, 2, 2), 0.2, dtype=torch.float32)

    preds = rankseg_rma(
        probs,
        metric="dice",
        output_mode=" MULTICLASS ",
        unassigned_policy=" VOID ",
        void_index=-7,
    )

    assert torch.equal(preds, torch.full((1, 2, 2), -7, dtype=torch.int64))


@pytest.mark.parametrize("void_index", [0, 1])
def test_rankseg_rma_rejects_void_index_that_collides_with_a_class(void_index):
    probs = torch.full((1, 2, 2), 0.2, dtype=torch.float32)

    with pytest.raises(ValueError, match=r"outside the valid class index range \[0, num_class\)"):
        rankseg_rma(
            probs,
            output_mode="multiclass",
            unassigned_policy="void",
            void_index=void_index,
        )


def test_rankseg_rma_rejects_default_void_index_for_256_classes():
    probs = torch.full((1, 256, 1), 0.2, dtype=torch.float32)

    with pytest.raises(ValueError, match=r"outside the valid class index range \[0, num_class\)"):
        rankseg_rma(
            probs,
            output_mode="multiclass",
            unassigned_policy="void",
        )


@pytest.mark.parametrize("void_index", [-(2**63) - 1, 2**63])
def test_rankseg_rma_rejects_void_index_outside_int64_range(void_index):
    probs = torch.full((1, 2, 2), 0.2, dtype=torch.float32)

    with pytest.raises(ValueError, match="representable as torch.int64"):
        rankseg_rma(
            probs,
            output_mode="multiclass",
            unassigned_policy="void",
            void_index=void_index,
        )


@pytest.mark.parametrize("void_index", [-(2**63), 2**63 - 1])
def test_rankseg_rma_accepts_void_index_at_int64_boundaries(void_index):
    probs = torch.full((1, 2, 2), 0.2, dtype=torch.float32)

    preds = rankseg_rma(
        probs,
        output_mode="multiclass",
        unassigned_policy="void",
        void_index=void_index,
    )

    assert torch.equal(preds, torch.full((1, 2), void_index, dtype=torch.int64))


@pytest.mark.parametrize("algorithm", [rankdice_ba, rankseg_rma])
def test_exported_rankseg_algorithms_reject_integer_probabilities(algorithm):
    probs = torch.ones((1, 1, 4), dtype=torch.int64)

    with pytest.raises(TypeError, match="real floating-point dtype"):
        algorithm(probs)


def test_exported_rankseg_algorithms_disable_autograd(monkeypatch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    probs = torch.tensor([[[0.8, 0.2, 0.6, 0.4]]], dtype=torch.float32, device=device, requires_grad=True)
    expected_ba = rankdice_ba(probs.detach(), solver="BA", pruning_prob=0.0)
    expected_rma = rankseg_rma(probs.detach(), output_mode="multilabel", pruning_prob=0.0)

    grad_states = []
    original_validate = rankseg_algo.validate_probability_tensor

    def record_grad_state(*args, **kwargs):
        grad_states.append(torch.is_grad_enabled())
        return original_validate(*args, **kwargs)

    monkeypatch.setattr(rankseg_algo, "validate_probability_tensor", record_grad_state)
    with torch.enable_grad():
        actual_ba = rankdice_ba(probs, solver="BA", pruning_prob=0.0)
        actual_rma = rankseg_rma(probs, output_mode="multilabel", pruning_prob=0.0)

    assert grad_states == [False, False]
    assert not actual_ba.requires_grad
    assert not actual_rma.requires_grad
    assert torch.equal(actual_ba, expected_ba)
    assert torch.equal(actual_rma, expected_rma)


def test_exported_rankdice_ba_rejects_degenerate_eps():
    probs = torch.tensor([[[0.8, 0.2, 0.6, 0.4]]], dtype=torch.float32)

    with pytest.raises(ValueError, match="eps must be in the range \\(0, 1\\)"):
        rankdice_ba(probs, eps=0.0)


def test_exported_rankseg_rma_rejects_single_channel_multiclass_input():
    probs = torch.tensor([[[0.8, 0.2, 0.6, 0.4]]], dtype=torch.float32)

    with pytest.raises(ValueError, match="Single-channel probabilities"):
        rankseg_rma(probs, output_mode="multiclass")


def test_rankdice_ba_trna_returns_binary_masks():
    probs = _demo_probs()

    preds = rankdice_ba(probs, solver="TRNA")

    assert preds.shape == probs.shape
    assert preds.dtype == torch.bool
    expected = F.one_hot(probs.argmax(dim=1), num_classes=probs.shape[1]).movedim(-1, 1).bool()
    assert torch.equal(preds, expected)


def test_rankdice_ba_plus_trna_executes_solver_selection():
    probs = _demo_probs()
    preds_trna = rankdice_ba(probs, solver="TRNA")

    preds = rankdice_ba(probs, solver="BA+TRNA")

    assert preds.shape == probs.shape
    assert preds.dtype == torch.bool
    assert torch.equal(preds, preds_trna)


def test_rankdice_ba_trna_supports_positive_smooth():
    probs = _demo_probs()

    preds = rankdice_ba(probs, solver="TRNA", smooth=0.2)

    assert preds.shape == probs.shape
    assert preds.dtype == torch.bool
    expected = F.one_hot(probs.argmax(dim=1), num_classes=probs.shape[1]).movedim(-1, 1).bool()
    assert torch.equal(preds, expected)


def test_rankdice_ba_handles_zero_variance_probabilities():
    probs = torch.tensor(
        [
            [
                [[1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, 1.0]],
            ]
        ],
        dtype=torch.float32,
    )

    preds = rankdice_ba(probs, solver="BA", pruning_prob=0.0)

    assert preds.shape == probs.shape
    assert preds.dtype == torch.bool
    assert torch.equal(preds, probs.bool())


@pytest.mark.parametrize("solver", ["BA", "TRNA"])
@pytest.mark.parametrize("smooth", [0.0, 0.2, 1e6, 1e308])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "values",
    [
        [0.8],
        [0.6, 0.2],
        [1.0, 0.0, 1.0, 0.0, 1.0],
        [0.91, 0.73, 0.52, 0.31, 0.14, 0.03],
        [0.99, 0.01, 0.99, 0.01, 0.99, 0.01],
    ],
)
def test_rankdice_matches_independent_scalar_formula_oracle(solver, smooth, dtype, values):
    probs = torch.tensor(values, dtype=dtype)
    expected = _rankdice_formula_oracle(probs, solver, smooth)

    preds = rankdice_ba(
        probs.view(1, 1, -1),
        solver=solver,
        smooth=smooth,
        pruning_prob=0.0,
    )

    assert torch.equal(preds[0, 0], expected)


@pytest.mark.parametrize("solver", ["BA", "TRNA"])
@pytest.mark.parametrize("smooth", [0.0, 0.2, 1e8])
def test_rankdice_float64_matches_independent_scalar_formula_oracle(solver, smooth):
    probs = torch.tensor([0.91000000003, 0.73000000002, 0.52000000001, 0.31, 0.14, 0.03], dtype=torch.float64)
    expected = _rankdice_formula_oracle(probs, solver, smooth)

    preds = rankdice_ba(
        probs.view(1, 1, -1),
        solver=solver,
        smooth=smooth,
        pruning_prob=0.0,
    )

    assert torch.equal(preds[0, 0], expected)


@pytest.mark.parametrize("solver", ["BA", "TRNA", "BA+TRNA"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")),
    ],
)
def test_rankdice_float64_preserves_pruning_boundary(solver, device):
    probs = torch.tensor([[[0.50000001]]], dtype=torch.float64, device=device)

    preds = rankdice_ba(probs, solver=solver, pruning_prob=0.5)
    rounded_preds = rankdice_ba(probs.float(), solver=solver, pruning_prob=0.5)

    assert preds.item()
    assert not rounded_preds.item()


@pytest.mark.parametrize("solver", ["BA", "TRNA", "BA+TRNA"])
@pytest.mark.parametrize("smooth", [0.0, 0.2, 1e6, 1e308])
def test_rankdice_single_pixel_and_deterministic_probabilities(solver, smooth):
    probs = torch.tensor([[[1.0, 0.0, 1.0, 0.0, 1.0]]])

    preds = rankdice_ba(probs, solver=solver, smooth=smooth, pruning_prob=0.0)

    assert torch.equal(preds, probs.bool())


@pytest.mark.parametrize("solver", ["BA", "TRNA", "BA+TRNA"])
def test_rankdice_large_smooth_preserves_score_order(solver):
    probs = torch.tensor([[[0.8, 0.2, 0.6, 0.4]]])
    expected = torch.tensor([[[True, False, True, False]]])

    preds = rankdice_ba(probs, solver=solver, smooth=1e308, pruning_prob=0.0)

    assert torch.equal(preds, expected)


@pytest.mark.parametrize("metric", ["dice", "iou"])
def test_rankseg_rma_large_smooth_preserves_score_order(metric):
    probs = torch.tensor([[[0.8, 0.2, 0.6, 0.4]]])
    expected = torch.tensor([[[True, False, True, False]]])

    preds = rankseg_rma(
        probs,
        metric=metric,
        output_mode="multilabel",
        smooth=1e308,
        pruning_prob=0.0,
    )

    assert torch.equal(preds, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("solver", ["BA", "TRNA", "BA+TRNA"])
def test_rankdice_large_smooth_cuda_matches_cpu(solver):
    probs = torch.tensor([[[0.81, 0.19, 0.61, 0.39]]])
    expected = rankdice_ba(probs, solver=solver, smooth=1e308, pruning_prob=0.0)

    preds = rankdice_ba(probs.cuda(), solver=solver, smooth=1e308, pruning_prob=0.0)

    assert torch.equal(preds.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("metric", ["dice", "iou"])
def test_rankseg_rma_large_smooth_cuda_matches_cpu(metric):
    probs = torch.tensor([[[0.81, 0.19, 0.61, 0.39]]])
    expected = rankseg_rma(
        probs,
        metric=metric,
        output_mode="multilabel",
        smooth=1e308,
        pruning_prob=0.0,
    )

    preds = rankseg_rma(
        probs.cuda(),
        metric=metric,
        output_mode="multilabel",
        smooth=1e308,
        pruning_prob=0.0,
    )

    assert torch.equal(preds.cpu(), expected)


@pytest.mark.parametrize("shape", [(1, 1, 4), (0, 1, 4)])
def test_rankdice_rejects_unknown_solver_before_data_dependent_work(shape):
    probs = torch.zeros(shape, dtype=torch.float32)

    with pytest.raises(ValueError, match="Unknown solver"):
        rankdice_ba(probs, solver="invalid")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rankdice_ba_plus_trna_mixed_batch_uses_matching_distribution_rows(dtype):
    generator = torch.Generator().manual_seed(20260901)
    ba_probs = torch.rand(256, generator=generator)
    trna_probs = torch.where(torch.arange(256) % 2 == 0, 0.99, 0.01)
    pruned_probs = torch.full((256,), 0.5)
    probs = torch.stack([trna_probs, ba_probs, pruned_probs, ba_probs, trna_probs]).unsqueeze(1).to(dtype=dtype)

    expected = torch.cat(
        [
            rankdice_ba(probs[0:1], solver="TRNA", smooth=0.2),
            rankdice_ba(probs[1:2], solver="BA", smooth=0.2),
            torch.zeros_like(probs[2:3], dtype=torch.bool),
            rankdice_ba(probs[3:4], solver="BA", smooth=0.2),
            rankdice_ba(probs[4:5], solver="TRNA", smooth=0.2),
        ]
    )

    preds = rankdice_ba(probs, solver="BA+TRNA", smooth=0.2)

    assert torch.equal(preds, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_rankdice_ba_plus_trna_cuda_mixed_batch_uses_matching_distribution_rows():
    generator = torch.Generator().manual_seed(20260901)
    ba_probs = torch.rand(256, generator=generator)
    trna_probs = torch.where(torch.arange(256) % 2 == 0, 0.99, 0.01)
    pruned_probs = torch.full((256,), 0.5)
    probs = torch.stack([trna_probs, ba_probs, pruned_probs, ba_probs, trna_probs]).unsqueeze(1).cuda()

    expected = torch.cat(
        [
            rankdice_ba(probs[0:1], solver="TRNA", smooth=0.2),
            rankdice_ba(probs[1:2], solver="BA", smooth=0.2),
            torch.zeros_like(probs[2:3], dtype=torch.bool),
            rankdice_ba(probs[3:4], solver="BA", smooth=0.2),
            rankdice_ba(probs[4:5], solver="TRNA", smooth=0.2),
        ]
    )

    preds = rankdice_ba(probs, solver="BA+TRNA", smooth=0.2)

    assert torch.equal(preds, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("dim", [127, 128, 129])
def test_rankdice_trna_cuda_matches_cpu_across_pmf_chunk_boundary(dim):
    # Uniform probabilities make Lemma 3 use the full [0, dim] search range.
    probs = torch.full((1, 1, dim), 0.6)
    expected = rankdice_ba(probs, solver="TRNA", smooth=0.2, pruning_prob=0.0)

    preds = rankdice_ba(probs.cuda(), solver="TRNA", smooth=0.2, pruning_prob=0.0)
    repeated = rankdice_ba(probs.cuda(), solver="TRNA", smooth=0.2, pruning_prob=0.0)

    assert torch.equal(preds.cpu(), expected)
    assert torch.equal(preds, repeated)


def test_rankdice_ba_rejects_unknown_solver():
    probs = _demo_probs()

    with pytest.raises(ValueError, match="Unknown solver"):
        rankdice_ba(probs, solver="invalid")


@pytest.mark.parametrize("solver", [None, 1])
def test_rankdice_ba_rejects_non_string_solver(solver):
    probs = _demo_probs()

    with pytest.raises(TypeError, match="solver must be a string"):
        rankdice_ba(probs, solver=solver)


@pytest.mark.parametrize(
    ("solver", "canonical_solver"),
    [
        ("ba", "BA"),
        (" Ba ", "BA"),
        ("trna", "TRNA"),
        (" TrNa ", "TRNA"),
        ("ba+trna", "BA+TRNA"),
        (" Ba+TrNa ", "BA+TRNA"),
    ],
)
def test_rankdice_ba_normalizes_solver_name(solver, canonical_solver):
    probs = _demo_probs()
    expected = rankdice_ba(probs, solver=canonical_solver)

    preds = rankdice_ba(probs, solver=solver)

    assert torch.equal(preds, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("smooth", [0.0, 0.2])
def test_rankdice_ba_plus_trna_cuda_mixed_selection_matches_independent_solvers(dtype, smooth):
    generator = torch.Generator().manual_seed(20260901)
    ba_probs = torch.rand(256, generator=generator)
    trna_probs = torch.where(torch.arange(256) % 2 == 0, 0.99, 0.01)
    pruned_probs = torch.full((256,), 0.5)
    probs = torch.stack([ba_probs, trna_probs, pruned_probs]).unsqueeze(0).to(dtype=dtype)

    expected_ba = rankdice_ba(
        probs[:, :1].cuda(),
        solver="BA",
        smooth=smooth,
        pruning_prob=0.5,
    )
    expected_trna = rankdice_ba(
        probs[:, 1:2].cuda(),
        solver="TRNA",
        smooth=smooth,
        pruning_prob=0.5,
    )
    expected = torch.cat([expected_ba, expected_trna, torch.zeros_like(expected_trna)], dim=1)

    preds = rankdice_ba(
        probs.cuda(),
        solver="BA+TRNA",
        smooth=smooth,
        pruning_prob=0.5,
    )
    repeated = rankdice_ba(
        probs.cuda(),
        solver="BA+TRNA",
        smooth=smooth,
        pruning_prob=0.5,
    )

    assert preds.device.type == "cuda"
    assert preds.dtype == torch.bool
    assert torch.equal(preds, expected)
    assert torch.equal(preds, repeated)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("solver", ["TRNA", "BA+TRNA"])
def test_rankdice_cuda_paths_allow_empty_batches(solver):
    probs = torch.empty((0, 2, 11), dtype=torch.float32, device="cuda")

    preds = rankdice_ba(probs, solver=solver)

    assert preds.shape == probs.shape
    assert preds.device.type == "cuda"
    assert preds.dtype == torch.bool
