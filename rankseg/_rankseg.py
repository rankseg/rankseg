# Author: Ben Dai <bendai@cuhk.edu.hk>, Zixun Wang <zixunwang@link.cuhk.edu.hk>
# License: BSD 3 clause

from rankseg.functional import rankseg


class RankSEG(object):
    r"""RankSEG segmentation prediction module for optimizing segmentation metrics :cite:p:`dai2023rankseg`, :cite:p:`wang2025rankseg`.

    This class converts probability maps into segmentation predictions targeting
    the supported Dice, IoU, and Accuracy metrics.

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
        division by zero and improve numerical stability. Must be finite and
        greater than or equal to 0. This parameter affects Dice and IoU only;
        valid values are accepted but ignored when optimizing Accuracy.

    output_mode : {'multiclass', 'multilabel'}, default='multiclass'
        String values are matched case-insensitively after stripping leading
        and trailing whitespace.
        Controls whether predictions are non-overlapping or overlapping.
        - 'multiclass': non-overlapping; each pixel belongs to exactly one class.
        - 'multilabel': overlapping; pixels can belong to multiple classes (binary mask per class).

        For single-channel binary probabilities, use 'multilabel' to obtain a
        foreground mask. To obtain a multiclass 0/1 label map, provide two
        channels containing background and foreground probabilities;
        single-channel 'multiclass' input is rejected.

    solver : str, default='RMA'
        String values are matched case-insensitively after stripping leading
        and trailing whitespace.
        The optimization solver to use. Options:

        - When metric is 'dice':

          - 'BA': Blind approximation, for 'multilabel' output
          - 'TRNA': Truncated refined normal approximation, for 'multilabel' output
          - 'BA+TRNA': Automatically select from 'BA' or 'TRNA', for 'multilabel' output
          - 'RMA': Reciprocal moment approximation, for 'multiclass' or 'multilabel' output

        - When metric is 'IoU':

          - 'RMA': Reciprocal moment approximation

        - When metric is 'Acc':

          - 'argmax': argmax solver, required for 'multiclass' output; with
            multi-channel 'multilabel' output it returns non-overlapping one-hot masks
          - 'TR': truncation solver, for single-channel or 'multilabel' output

        Unsupported metric, output mode, and solver combinations raise an
        error; RankSEG does not silently substitute a different solver.

    pruning_prob : float, default=0.5
        Probability threshold for pruning. Classes with maximum probability
        less than or equal to this threshold are skipped to improve efficiency.
        Must be finite and lie in the range [0, 1]. This parameter affects
        Dice and IoU only; valid values are accepted but ignored when optimizing
        Accuracy.

    \*\*solver_params : dict
        Additional parameters passed to the specific solver.
        For 'BA', 'TRNA' or 'BA+TRNA', ``eps`` is the tail probability used to
        retain the central ``1 - eps`` refined-normal interval; it must be
        finite and lie strictly between 0 and 1.
        For 'RMA', overlapping binary masks are resolved by the largest
        incremental score among the classes that selected the pixel; a class
        that did not select the pixel cannot win. The score is the RMA
        objective change from assigning the pixel, not its raw probability.
        unassigned_policy and void_index control pixels selected by no class.
        With 'max_score', all classes are reconsidered if pruning removes every
        class in a sample. A void index must fit in ``torch.int64`` and lie
        outside the valid class index range.
        ``safe_screening=True`` forces experimental screened sorting for
        RMA Dice with ``smooth=0``. ``'auto'`` screens CPU inputs and CUDA
        inputs with Triton and at least 1,280,000 probability values (B*C*D);
        smaller/no-Triton CUDA inputs use optimized full sort.
        The default is False, retaining the original path. Other metric/smooth
        combinations keep full sort. Direct argmax breaks exactly equal
        computed maxima by the smallest searched volume; masks need
        not match the default full-sort path bit for bit.
        Unsupported parameters raise an error rather than being ignored.

    References
    ----------
    :cite:p:`dai2023rankseg` Dai, B., & Li, C. (2023). Rankseg: a consistent ranking-based framework for segmentation. Journal of Machine Learning Research, 24(224), 1-50.

    :cite:p:`wang2025rankseg` Wang, Z., & Dai, B. (2025). RankSEG-RMA: An Efficient Segmentation Algorithm via Reciprocal Moment Approximation. Advances in Neural Information Processing Systems (NeurIPS 2025).

    Examples
    --------
    >>> import torch
    >>> from rankseg import RankSEG
    >>>
    >>> # Create a RankSEG prediction-time postprocessor
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
            Must use a real floating-point dtype. Values must be finite and lie
            in the range [0, 1]. The class and spatial dimensions must be
            non-empty; an empty batch is allowed.
            image_shape has no restriction on the number of dimensions,
            can be (height, width) for 2D images, or (height, width, depth) for 3D images, or others.

        Returns
        -------
        preds : torch.Tensor
            If `output_mode == "multilabel"`, returns boolean masks of shape
            (batch_size, num_class, \*image_shape) with dtype ``torch.bool``.

            If `output_mode == "multiclass"`, returns class-index maps of shape
            (batch_size, \*image_shape) with dtype ``torch.int64``.
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
