# Author: Ben Dai <bendai@cuhk.edu.hk>, Zixun Wang <zixunwang@link.cuhk.edu.hk>
# License: BSD 3 clause

import warnings

import torch

from rankseg._rankseg_algo import rankdice_ba, rankseg_rma
from rankseg._validation import validate_finite_real, validate_probability_tensor

_SOLVER_DISPLAY_NAMES = {
    "rma": "RMA",
    "ba": "BA",
    "trna": "TRNA",
    "ba+trna": "BA+TRNA",
    "tr": "TR",
    "argmax": "argmax",
}


def _supported_solvers(metric, output_mode, num_class):
    if metric == "dice":
        if output_mode == "multiclass":
            return ("rma",)
        return ("rma", "ba", "trna", "ba+trna")
    if metric == "iou":
        return ("rma",)
    if num_class == 1:
        return ("tr",)
    if output_mode == "multiclass":
        return ("argmax",)
    return ("tr", "argmax")


def _validate_solver_compatibility(metric, output_mode, solver, num_class):
    if metric == "dice" and solver == "exact":
        raise ValueError("Exact solver is not implemented yet")

    supported = _supported_solvers(metric, output_mode, num_class)
    if solver not in supported:
        supported_names = ", ".join(_SOLVER_DISPLAY_NAMES[name] for name in supported)
        solver_name = _SOLVER_DISPLAY_NAMES.get(solver, solver)
        channel_context = ", single-channel input" if num_class == 1 else ""
        raise ValueError(
            f"solver {solver_name!r} is not supported for metric={metric!r} "
            f"with output_mode={output_mode!r}{channel_context}; supported solvers: {supported_names}"
        )


def _validate_solver_params(solver, output_mode, solver_params):
    if solver == "rma":
        allowed = {"safe_screening"}
        if output_mode == "multiclass":
            allowed |= {"unassigned_policy", "void_index"}
    elif solver in {"ba", "trna", "ba+trna"}:
        allowed = {"eps"}
    else:
        allowed = set()

    unexpected = sorted(set(solver_params) - allowed)
    if unexpected:
        names = ", ".join(unexpected)
        solver_name = _SOLVER_DISPLAY_NAMES[solver]
        if allowed:
            allowed_names = ", ".join(sorted(allowed))
            raise TypeError(
                f"solver {solver_name!r} does not accept solver parameters: {names}; "
                f"allowed parameters: {allowed_names}"
            )
        raise TypeError(f"solver {solver_name!r} does not accept solver parameters: {names}")

    if solver == "rma" and "void_index" in solver_params:
        policy = solver_params.get("unassigned_policy", "max_score")
        if not isinstance(policy, str) or policy.strip().lower() != "void":
            raise ValueError("void_index requires unassigned_policy='void'")


@torch.no_grad()
def rankseg(
    probs: torch.Tensor,
    metric: str = "dice",
    smooth: float = 0.0,
    output_mode: str = "multiclass",
    solver: str = "RMA",
    pruning_prob: float = 0.5,
    **solver_params,
) -> torch.Tensor:
    r"""Convert probability maps to segmentation predictions.

    Parameters
    ----------
    probs : torch.Tensor
        Probability maps of shape (batch_size, num_class, \*image_shape).
        Must use a real floating-point dtype. Values must be finite and lie in
        the range [0, 1]. The class and spatial dimensions must be non-empty;
        an empty batch is allowed.
        image_shape has no restriction on the number of dimensions,
        can be (height, width) for 2D images, or (height, width, depth) for 3D images, or others.

    metric : str, default='dice'
        The segmentation metric to optimize. String values are matched
        case-insensitively after stripping leading and trailing whitespace.
        Currently supported: 'dice', 'IoU', and 'Acc'/'accuracy'.

    smooth : float, default=0.0
        Smoothing parameter added to numerator and denominator to avoid
        division by zero and improve numerical stability. Must be finite and
        greater than or equal to 0. This parameter affects Dice and IoU only;
        valid values are accepted but ignored when optimizing Accuracy.

    output_mode : {'multiclass', 'multilabel'}, default='multiclass'
        Controls whether predictions are non-overlapping or overlapping.
        For single-channel binary probabilities, use 'multilabel' to obtain a
        foreground mask. To obtain a multiclass 0/1 label map, provide two
        channels containing background and foreground probabilities;
        single-channel 'multiclass' input is rejected.

    solver : str, default='RMA'
        The optimization solver to use. Solver compatibility is validated
        strictly: Dice multiclass and all IoU outputs require RMA; Dice
        multilabel also supports BA, TRNA, and BA+TRNA; Accuracy requires TR
        for single-channel input, argmax for multiclass output, and either TR
        or argmax for multi-channel multilabel output.

    pruning_prob : float, default=0.5
        Probability threshold for pruning. Classes with maximum probability
        less than or equal to this threshold are skipped to improve efficiency.
        Must be finite and lie in the range [0, 1]. This parameter affects
        Dice and IoU only; valid values are accepted but ignored when optimizing
        Accuracy.

    \*\*solver_params : dict
        Additional parameters passed to the specific solver. In RMA multiclass
        output, overlapping binary masks are resolved by the largest
        incremental score among the classes that selected the pixel; a class
        that did not select the pixel cannot win. This score is the RMA
        objective change from assigning the pixel, not its raw probability.
        `unassigned_policy='max_score'` assigns pixels selected by no class
        using the active classes' incremental scores; if pruning removes every
        class in a sample, all classes are reconsidered for that fallback. Use
        `unassigned_policy='void'` and `void_index` to preserve abstentions.
        A void index must fit in ``torch.int64`` and lie outside the valid
        class index range so it cannot be confused with a class prediction.
        ``safe_screening=True`` forces experimental screened sorting for
        RMA Dice with ``smooth=0``. ``'auto'`` screens CPU inputs and CUDA
        inputs with Triton and at least 1,280,000 probability values (B*C*D);
        smaller/no-Triton CUDA inputs use optimized full sort.
        All other metric/smooth combinations retain
        full sort. It defaults to False, retaining the original path.
        This option does not change class pruning or
        multiclass eligibility. Direct argmax breaks exactly equal computed
        maxima by the smallest searched volume; near ties do not trigger a
        retry. Bitwise mask equivalence is not promised.
        For BA, TRNA, and BA+TRNA, `eps` is the tail probability used to retain
        the central `1 - eps` refined-normal interval; it must be finite and
        lie strictly between 0 and 1.

    Returns
    -------
    preds : torch.Tensor
        If `output_mode == "multilabel"`, returns boolean masks of shape
        (batch_size, num_class, \*image_shape) with dtype ``torch.bool``.

        If `output_mode == "multiclass"`, returns class-index maps of shape
        (batch_size, \*image_shape) with dtype ``torch.int64``.

    Notes
    -----
    RankSEG produces discrete inference-time predictions and is not
    differentiable. Gradient recording is disabled internally, even when
    ``probs.requires_grad`` is ``True``.
    """
    # Solver implementations validate values. Structural validation happens
    # here so dispatch can safely inspect class and spatial dimensions without
    # scanning a potentially large tensor twice.
    validate_probability_tensor(probs, check_values=False)

    if not isinstance(metric, str):
        raise TypeError("metric must be a string")
    if not isinstance(output_mode, str):
        raise TypeError("output_mode must be a string")
    if not isinstance(solver, str):
        raise TypeError("solver must be a string")

    smooth = validate_finite_real("smooth", smooth)
    if smooth < 0:
        raise ValueError("smooth must be greater than or equal to 0")
    pruning_prob = validate_finite_real("pruning_prob", pruning_prob)
    if not 0 <= pruning_prob <= 1:
        raise ValueError("pruning_prob must be in the range [0, 1]")

    num_class = probs.shape[1]
    metric = metric.strip().lower()
    output_mode = output_mode.strip().lower()
    solver = solver.strip().lower()

    if metric not in ["dice", "iou", "acc", "accuracy"]:
        raise ValueError("Unknown metric: %s" % metric)
    if metric == "acc":
        metric = "accuracy"

    ## check output mode
    if output_mode not in ["multiclass", "multilabel"]:
        raise ValueError("Unknown output mode: %s" % output_mode)
    if output_mode == "multiclass" and num_class == 1:
        raise ValueError(
            "Single-channel probabilities cannot produce a binary multiclass label map; "
            "use output_mode='multilabel' or provide background and foreground channels"
        )

    _validate_solver_compatibility(metric, output_mode, solver, num_class)
    _validate_solver_params(solver, output_mode, solver_params)

    if metric == "dice":
        if solver == "rma":
            preds = rankseg_rma(
                probs,
                metric="dice",
                output_mode=output_mode,
                smooth=smooth,
                pruning_prob=pruning_prob,
                **solver_params,
            )
        else:
            if "eps" in solver_params:
                eps = validate_finite_real("eps", solver_params["eps"])
                if not 0 < eps < 1:
                    raise ValueError("eps must be in the range (0, 1)")
                solver_params["eps"] = eps
            preds = rankdice_ba(
                probs,
                solver={"ba": "BA", "trna": "TRNA", "ba+trna": "BA+TRNA"}[solver],
                smooth=smooth,
                pruning_prob=pruning_prob,
                **solver_params,
            )

    elif metric == "iou":
        preds = rankseg_rma(
            probs,
            metric="iou",
            output_mode=output_mode,
            smooth=smooth,
            pruning_prob=pruning_prob,
            **solver_params,
        )

    else:
        validate_probability_tensor(probs)
        if num_class == 1:
            ## simply take thresholding at 0.5 over classes
            preds = probs > 0.5
        else:
            if output_mode == "multilabel":
                if solver == "argmax":
                    ## Return one-hot masks in multilabel tensor format.
                    warnings.warn(
                        "The argmax solver returns non-overlapping one-hot masks in multilabel tensor format. "
                        "Use the TR solver when labels should be thresholded independently.",
                        stacklevel=2,
                    )
                    preds = torch.zeros_like(probs, dtype=torch.bool)
                    class_indices = torch.argmax(probs, dim=1, keepdim=True)
                    preds.scatter_(1, class_indices, True)

                elif solver == "tr":
                    ## simply take truncation at 0.5 over classes
                    preds = probs > 0.5
            else:
                ## simply take argmax over classes
                preds = torch.argmax(probs, dim=1)
    return preds
