# Author: Ben Dai <bendai@cuhk.edu.hk>
# License: BSD 3 clause

import numpy as np
import scipy
import torch
import torch.nn.functional as F
from rankseg import rankdice_batch, rankseg_rma
import warnings

class RankSEG(object):
    """Rank-based Segmentation for optimizing segmentation metrics.
    
    This class provides methods to convert probability maps into binary segmentation
    predictions by optimizing ranking-based metrics like Dice coefficient.
    
    Parameters
    ----------
    metric : str, default='dice'
        The segmentation metric to optimize. Currently supported:
        - 'dice': Dice coefficient (F1 score)
        - 'IoU': Intersection over Union (not yet implemented)
    
    smooth : float, default=0.0
        Smoothing parameter added to numerator and denominator to avoid
        division by zero and improve numerical stability.

    return_binary_masks : bool, default=False
        Whether to return or allow binary masks per class (multi-label segmentation). Generally, this is only meaningful when segmentation datasets contain multiple labels.
        If False, performs multi-class segmentation where each pixel belongs to exactly one class.
        If True, performs multi-label segmentation where pixels can belong to multiple classes.

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
        - When metric is 'AP':
            - simply taking argmax or truncation (at 0.5) over classes
    
    pruning_prob : float, default=0.1
        Probability threshold for pruning. Classes with maximum probability
        below this threshold may be skipped to improve efficiency.
        Should be in range [0, 1].
    
    **solver_params : dict
        Additional parameters passed to the specific solver:
        - For 'BA', 'TRNA' or 'BA+TRNA': 
            eps (1 - confidence intervals for refined normal approximation of poissoon-binomial distributions)

    Examples
    --------
    >>> import torch
    >>> from rankseg import RankSEG
    >>> 
    >>> # Create segmentation model
    >>> rankseg = RankSEG(metric='dice', solver='BA', pruning_prob=0.5, eps=1e-4)
    >>> 
    >>> # Generate predictions from probability maps
    >>> probs = torch.rand(4, 21, 256, 256)  # (batch, classes, height, width)
    >>> preds = rankseg.predict(probs)
    """
    def __init__(self,
                 metric: str='dice',
                 smooth: float=0.,
                 return_binary_masks: bool=False,
                 solver: str='RMA',
                 pruning_prob: float=0.1,
                 **solver_params):
        self.metric = metric
        self.smooth = smooth
        self.return_binary_masks = return_binary_masks
        self.solver = solver
        self.pruning_prob = pruning_prob
        self.solver_params = solver_params

    def predict(self, probs):
        """Convert probability maps to binary segmentation predictions.
        
        Parameters
        ----------
        probs : torch.Tensor
            Probability maps of shape (batch_size, num_class, *image_shape).
            Values should be in range [0, 1].
            image_shape has no restriction on the number of dimensions,
                can be (height, width) for 2D images, or (height, width, depth) for 3D images, or others.
        
        Returns
        -------
        preds : torch.Tensor
            Binary segmentation predictions of shape (batch_size, num_class, *image_shape).
            Values are 0 or 1 (or boolean True/False depending on solver).
        """
        batch_size, num_class, *image_shape = probs.shape

        if self.metric == 'dice':
            if self.solver in ['BA', 'TRNA', 'BA+TRNA']:
                if (not self.return_binary_masks) and (num_class > 1):
                    warnings.warn('For Dice metric with BA/TRNA/BA+TRNA solver, it only supports returning binary masks per class (multi-label segmentation). Thus, `return_binary_masks` is automatically set as `return_binary_masks=True`.')
                    self.return_binary_masks = True
                preds = rankdice_batch(probs, 
                                    solver=self.solver,
                                    smooth=self.smooth,
                                    pruning_prob=self.pruning_prob,
                                    **self.solver_params)
            elif self.solver == 'exact':
                raise NotImplementedError('Exact solver is not implemented yet')
            elif self.solver == 'RMA':
                preds = rankseg_rma(probs,
                                    metric=self.metric,
                                    smooth=self.smooth,
                                    pruning_prob=self.pruning_prob,
                                    return_binary_masks=self.return_binary_masks,
                                    **self.solver_params)
            else:
                raise ValueError('Unknown solver: %s' % self.solver)
        elif self.metric == 'IoU':
            if self.solver == 'RMA':
                pass
            else:
                warnings.warn('Currently, only RMA supports IoU optimization, and RankSEG automatically switches to the RMA solver.')
            
            preds = rankseg_rma(probs,
                    metric=self.metric,
                    smooth=self.smooth,
                    pruning_prob=self.pruning_prob,
                    return_binary_masks=self.return_binary_masks,
                    **self.solver_params)

        elif self.metric == 'AP':
            if (self.return_binary_masks) or (num_class == 1):
                ## simply take thresholding at 0.5 over classes
                preds = torch.where(probs > .5, True, False)
            else:
                ## simply take argmax over classes
                preds = torch.zeros_like(probs)
                class_indices = torch.argmax(probs, dim=1, keepdim=True)
                preds.scatter_(1, class_indices, 1)

        else:
            raise ValueError('Unknown metric: %s' % self.metric)
        return preds