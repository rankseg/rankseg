# Author: Ben Dai <bendai@cuhk.edu.hk>, Zixun Wang <zixunwang@link.cuhk.edu.hk>
# License: BSD 3 clause

import warnings

import torch

from rankseg._rankseg_algo import rankdice_ba, rankseg_rma


def rankseg_predict(
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
        Values must be finite and lie in the range [0, 1].
        image_shape has no restriction on the number of dimensions,
        can be (height, width) for 2D images, or (height, width, depth) for 3D images, or others.

    metric : str, default='dice'
        The segmentation metric to optimize. String values are matched
        case-insensitively after stripping leading and trailing whitespace.
        Currently supported: 'dice', 'IoU', and 'Acc'/'accuracy'.

    smooth : float, default=0.0
        Smoothing parameter added to numerator and denominator to avoid
        division by zero and improve numerical stability.

    output_mode : {'multiclass', 'multilabel'}, default='multiclass'
        Controls whether predictions are non-overlapping or overlapping.

    solver : str, default='RMA'
        The optimization solver to use.

    pruning_prob : float, default=0.5
        Probability threshold for pruning. Classes with maximum probability
        below this threshold may be skipped to improve efficiency.

    \*\*solver_params : dict
        Additional parameters passed to the specific solver.

    Returns
    -------
    preds : torch.Tensor
        If `output_mode == "multilabel"`, returns binary masks of shape
        (batch_size, num_class, \*image_shape).

        If `output_mode == "multiclass"`, returns class index maps of shape
        (batch_size, \*image_shape).

        Depending on the selected solver, the returned dtype may be boolean
        or integer.
    """
    if not isinstance(probs, torch.Tensor):
        raise TypeError("probs must be a torch.Tensor")
    if probs.ndim < 3:
        raise ValueError("probs must have shape (batch_size, num_class, *image_shape)")
    if not bool(torch.isfinite(probs).all()):
        raise ValueError("probs must contain only finite values")
    if bool(torch.any((probs < 0) | (probs > 1))):
        raise ValueError("probs must be in the range [0, 1]")

    num_class = probs.shape[1]
    metric = metric.strip().lower() if isinstance(metric, str) else metric
    output_mode = output_mode.strip().lower() if isinstance(output_mode, str) else output_mode
    solver = solver.strip().lower() if isinstance(solver, str) else solver

    ## check output mode
    if output_mode not in ["multiclass", "multilabel"]:
        raise ValueError("Unknown output mode: %s" % output_mode)

    if solver == "exact":
        raise ValueError("Exact solver is not implemented yet")  ## TODO: implement exact solver

    if metric == "dice":
        if solver not in ["ba", "trna", "ba+trna", "rma"]:
            warnings.warn(
                "Currently, we only support BA, TRNA, BA+TRNA, RMA solvers for Dice metric; this prediction uses `RMA`.",
                stacklevel=2,
            )
            solver = "rma"

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
            if (output_mode == "multiclass") and (num_class > 1):
                warnings.warn(
                    "For Dice metric with BA/TRNA/BA+TRNA solver, it only supports returning binary masks per class (multi-label segmentation). This prediction uses `RMA`.",
                    stacklevel=2,
                )
                solver = "rma"

                preds = rankseg_rma(
                    probs,
                    metric="dice",
                    smooth=smooth,
                    pruning_prob=pruning_prob,
                    output_mode=output_mode,
                    **solver_params,
                )
            else:
                preds = rankdice_ba(
                    probs,
                    solver={"ba": "BA", "trna": "TRNA", "ba+trna": "BA+TRNA"}[solver],
                    smooth=smooth,
                    pruning_prob=pruning_prob,
                    **solver_params,
                )

    elif metric == "iou":
        if solver != "rma":
            warnings.warn(
                "Currently, we only support RMA solver for IoU metric; this prediction uses `RMA`.", stacklevel=2
            )
            solver = "rma"

        preds = rankseg_rma(
            probs,
            metric="iou",
            output_mode=output_mode,
            smooth=smooth,
            pruning_prob=pruning_prob,
            **solver_params,
        )

    elif metric in ["acc", "accuracy"]:
        if num_class == 1:
            ## simply take thresholding at 0.5 over classes
            preds = torch.where(probs > 0.5, 1, 0)
        else:
            if output_mode == "multilabel":
                if solver == "argmax":
                    ## argmax produces non-overlapping output, inconsistent with multilabel mode
                    warnings.warn(
                        "Returning argmax binary masks. For multilabel segmentation with accuracy metric, argmax solver "
                        "produces non-overlapping predictions (multiclass output). Consider using 'TR' "
                        "solver for true multilabel outcome. ",
                        stacklevel=2,
                    )
                    preds = torch.zeros_like(probs)
                    class_indices = torch.argmax(probs, dim=1, keepdim=True)
                    preds.scatter_(1, class_indices, 1)

                elif solver == "tr":
                    ## simply take truncation at 0.5 over classes
                    preds = torch.where(probs > 0.5, 1, 0)
                else:
                    warnings.warn(
                        "Currently, only argmax and truncation solvers support overlapping multi-class segmentation, and this prediction uses the TR solver.",
                        stacklevel=2,
                    )
                    preds = torch.where(probs > 0.5, 1, 0)
            else:
                if solver != "argmax":
                    warnings.warn(
                        "Currently, only argmax solver supports non-overlapping multi-class segmentation, and this prediction uses the argmax solver.",
                        stacklevel=2,
                    )
                ## simply take argmax over classes
                preds = torch.argmax(probs, dim=1)
    else:
        raise ValueError("Unknown metric: %s" % metric)
    return preds


class RankSEG(object):
    r"""RankSEG segmentation prediction module for optimizing segmentation metrics :cite:p:`dai2023rankseg`, :cite:p:`wang2025rankseg`.

    This class provides methods to convert probability maps into segmentation
    predictions by optimizing segmentation metrics like AP, Dice, IoU, and Accuracy.

    Parameters
    ----------
    metric : str, default='dice'
        The segmentation metric to optimize. String values are matched
        case-insensitively after stripping leading and trailing whitespace.
        Currently supported:

        - 'dice': Dice coefficient
        - 'IoU': Intersection over Union
        - 'Acc': Accuracy

    smooth : float, default=0.0
        Smoothing parameter added to numerator and denominator to avoid
        division by zero and improve numerical stability.

    output_mode : {'multiclass', 'multilabel'}, default='multiclass'
        String values are matched case-insensitively after stripping leading
        and trailing whitespace.
        Controls whether predictions are non-overlapping or overlapping.
        - 'multiclass': non-overlapping; each pixel belongs to exactly one class.
        - 'multilabel': overlapping; pixels can belong to multiple classes (binary mask per class).

    solver : str, default='RMA'
        String values are matched case-insensitively after stripping leading
        and trailing whitespace.
        The optimization solver to use. Options:

        - When metric is 'dice':

          - 'exact': Exact solver (not yet implemented)
          - 'BA': Blind approximation
          - 'TRNA': Truncated refined normal approximation
          - 'BA+TRNA': Automatically select from 'BA' or 'TRNA' solver based on data information
          - 'RMA': Reciprocal moment approximation

        - When metric is 'IoU':

          - 'RMA': Reciprocal moment approximation

        - When metric is 'Acc':

          - 'argmax': argmax solver
          - 'TR': truncation solver

    pruning_prob : float, default=0.5
        Probability threshold for pruning. Classes with maximum probability
        below this threshold may be skipped to improve efficiency.
        Should be in range [0, 1].

    \*\*solver_params : dict
        Additional parameters passed to the specific solver.
        For 'BA', 'TRNA' or 'BA+TRNA': eps (1 - confidence intervals for refined
        normal approximation of poisson-binomial distributions).
        For 'RMA': unassigned_policy and void_index control multiclass output
        for pixels not selected as positive by any class.

    References
    ----------
    :cite:p:`dai2023rankseg` Dai, B., & Li, C. (2023). Rankseg: a consistent ranking-based framework for segmentation. Journal of Machine Learning Research, 24(224), 1-50.

    :cite:p:`wang2025rankseg` Wang, Z., & Dai, B. (2025). RankSEG-RMA: An Efficient Segmentation Algorithm via Reciprocal Moment Approximation. arXiv preprint arXiv:2510.15362.

    Examples
    --------
    >>> import torch
    >>> from rankseg import RankSEG
    >>>
    >>> # Create segmentation model
    >>> rankseg = RankSEG(metric='dice', output_mode='multilabel', solver='BA', pruning_prob=0.5, eps=1e-4)
    >>>
    >>> # Generate predictions from probability maps
    >>> probs = torch.softmax(torch.rand(4, 21, 256, 256), dim=1)  # (batch, classes, height, width)
    >>> preds = rankseg.predict(probs)                             # (batch, classes, height, width)
    """

    def __init__(
        self,
        metric: str = "dice",
        smooth: float = 0.0,
        output_mode: str = "multiclass",
        solver: str = "RMA",
        pruning_prob: float = 0.5,
        **solver_params,
    ):
        self.metric = metric
        self.smooth = smooth
        self.output_mode = output_mode
        self.solver = solver
        self.pruning_prob = pruning_prob
        self.solver_params = solver_params

    def predict(self, probs):
        r"""Convert probability maps to segmentation predictions.

        Parameters
        ----------
        probs : torch.Tensor
            Probability maps of shape (batch_size, num_class, \*image_shape).
            Values must be finite and lie in the range [0, 1].
            image_shape has no restriction on the number of dimensions,
            can be (height, width) for 2D images, or (height, width, depth) for 3D images, or others.

        Returns
        -------
        preds : torch.Tensor
            If `output_mode == "multilabel"`, returns binary masks of shape
            (batch_size, num_class, \*image_shape).

            If `output_mode == "multiclass"`, returns class index maps of shape
            (batch_size, \*image_shape).

            Depending on the selected solver, the returned dtype may be boolean
            or integer.
        """
        return self(probs)

    def __call__(self, probs):
        r"""Convert probability maps to segmentation predictions."""
        return rankseg_predict(
            probs,
            metric=self.metric,
            smooth=self.smooth,
            output_mode=self.output_mode,
            solver=self.solver,
            pruning_prob=self.pruning_prob,
            **self.solver_params,
        )
