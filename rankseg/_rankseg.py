# Author: Ben Dai <bendai@cuhk.edu.hk>, Zixun Wang <zixunwang@link.cuhk.edu.hk>
# License: BSD 3 clause

import torch
from rankseg._rankseg_algo import rankdice_ba, rankseg_rma
import warnings

class RankSEG(object):
    """RankSEG segmentation prediction module for optimizing segmentation metrics :cite:p:`dai2023rankseg`, :cite:p:`wang2025rankseg`.
    
    This class provides methods to convert probability maps into segmentation
    predictions by optimizing segmentation metrics like AP, Dice, IoU, and Accuracy.
    
    Parameters
    ----------
    metric : str, default='dice'
        The segmentation metric to optimize. Currently supported:
        
        - 'dice': Dice coefficient
        - 'IoU': Intersection over Union
        - 'Acc': Accuracy 
    
    smooth : float, default=0.0
        Smoothing parameter added to numerator and denominator to avoid
        division by zero and improve numerical stability.

    output_mode : {'multiclass', 'multilabel'}, default='multiclass'
        Controls whether predictions are non-overlapping or overlapping.
        - 'multiclass': non-overlapping; each pixel belongs to exactly one class.
        - 'multilabel': overlapping; pixels can belong to multiple classes (binary mask per class).

    solver : str, default='RMA'
        The optimization solver to use. Options:
        
        - When metric is 'dice':
          
          - 'exact': Exact solver (not yet implemented)
          - 'BA': Blind approximation
          - 'TRNA': Tuncated refined normal approximation
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
        normal approximation of poissoon-binomial distributions)

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
    def __init__(self,
                 metric: str='dice',
                 smooth: float=0.,
                 output_mode: str='multiclass',
                 solver: str='RMA',
                 pruning_prob: float=0.5,
                 **solver_params):
        self.metric = metric
        self.smooth = smooth
        self.output_mode = output_mode
        self.solver = solver
        self.pruning_prob = pruning_prob
        self.solver_params = solver_params

    def predict(self, probs):
        """Convert probability maps to binary segmentation predictions.
        
        Parameters
        ----------
        probs : torch.Tensor
            Probability maps of shape (batch_size, num_class, \*image_shape).
            Values should be in range [0, 1].
            image_shape has no restriction on the number of dimensions,
            can be (height, width) for 2D images, or (height, width, depth) for 3D images, or others.
        
        Returns
        -------
        preds : torch.Tensor
            Binary segmentation predictions of shape (batch_size, num_class, \*image_shape).
            Values are 0 or 1 (or boolean True/False depending on solver).
        """
        batch_size, num_class, *image_shape = probs.shape
        
        ## check output mode
        if self.output_mode not in ['multiclass', 'multilabel']:
            raise ValueError('Unknown output mode: %s' % self.output_mode)
        
        if self.solver == 'exact':
            raise ValueError('Exact solver is not implemented yet') ## TODO: implement exact solver

        if self.metric in ['dice', 'Dice', 'DICE']:
            if self.solver not in ['BA', 'TRNA', 'BA+TRNA', 'RMA']:
                warnings.warn("Currently, we only support BA, TRNA, BA+TRNA, RMA solvers for Dice metric; `solver` is automatically set as `RMA`.")
                self.solver = 'RMA'

            if self.solver == 'RMA':
                preds = rankseg_rma(probs,
                                    metric=self.metric,
                                    output_mode=self.output_mode,
                                    smooth=self.smooth,
                                    pruning_prob=self.pruning_prob,
                                    **self.solver_params)
            else:
                if (self.output_mode == 'multiclass') and (num_class > 1):
                    warnings.warn('For Dice metric with BA/TRNA/BA+TRNA solver, it only supports returning binary masks per class (multi-label segmentation). Thus, `solver` is automatically set as `RMA`.')
                    self.solver = 'RMA'

                    preds = rankseg_rma(probs,
                                        metric=self.metric,
                                        smooth=self.smooth,
                                        pruning_prob=self.pruning_prob,
                                        output_mode=self.output_mode,
                                        **self.solver_params)
                else:
                    preds = rankdice_ba(probs, 
                                        solver=self.solver,
                                        smooth=self.smooth,
                                        pruning_prob=self.pruning_prob,
                                        **self.solver_params)

        elif self.metric in ['IoU', 'IOU', 'iou']:
            if self.solver != 'RMA':
                warnings.warn("Currently, we only support RMA solver for IoU metric; `solver` is automatically set as `RMA`.")
                self.solver = 'RMA'
            
            preds = rankseg_rma(probs,
                    metric=self.metric,
                    output_mode=self.output_mode,
                    smooth=self.smooth,
                    pruning_prob=self.pruning_prob,
                    **self.solver_params)

        elif self.metric in ['Acc', 'acc', 'accuracy', 'ACC', 'Accuracy']:
            if num_class == 1:
                ## simply take thresholding at 0.5 over classes
                preds = torch.where(probs > .5, 1, 0)
            else:
                if self.output_mode == 'multilabel':
                    if self.solver == 'argmax':
                        ## argmax produces non-overlapping output, inconsistent with multilabel mode
                        warnings.warn(
                            "Returning argmax binary masks. For multilabel segmentation with accuracy metric, argmax solver "
                            "produces non-overlapping predictions (multiclass output). Consider using 'TR' "
                            "solver for true multilabel outcome. "
                        )
                        preds = torch.zeros_like(probs)
                        class_indices = torch.argmax(probs, dim=1, keepdim=True)
                        preds.scatter_(1, class_indices, 1)

                    elif self.solver == 'TR':
                        ## simply take truncation at 0.5 over classes
                        preds = torch.where(probs > .5, 1, 0)
                    else:
                        warnings.warn('Currently, only argmax and truncation solvers support overlapping multi-class segmentation, and RankSEG automatically switches to the TR solver.')
                        preds = torch.where(probs > .5, 1, 0)
                else:
                    if self.solver != 'argmax':
                        warnings.warn('Currently, only argmax solver supports non-overlapping multi-class segmentation, and RankSEG automatically switches to the argmax solver.')
                    ## simply take argmax over classes
                    preds = torch.argmax(probs, dim=1)
        else:
            raise ValueError('Unknown metric: %s' % self.metric)
        return preds
