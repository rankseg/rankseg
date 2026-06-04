# Author: Ben Dai <bendai@cuhk.edu.hk>, Zixun Wang <zixunwang@link.cuhk.edu.hk>
# License: BSD 3 clause

from rankseg.functional import rankseg


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
        return rankseg(
            probs,
            metric=self.metric,
            smooth=self.smooth,
            output_mode=self.output_mode,
            solver=self.solver,
            pruning_prob=self.pruning_prob,
            **self.solver_params,
        )
