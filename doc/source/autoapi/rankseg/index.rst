rankseg
=======

.. py:module:: rankseg


Classes
-------

.. autoapisummary::

   rankseg.RefinedNormalPB
   rankseg.RefinedNormal
   rankseg.RankSEG


Functions
---------

.. autoapisummary::

   rankseg.rankdice_ba
   rankseg.rankseg_rma


Package Contents
----------------

.. py:class:: RefinedNormalPB(dim: Union[torch.Tensor, int], loc: Union[torch.Tensor, float], scale: Union[torch.Tensor, float], skew: Union[torch.Tensor, float], validate_args: Optional[bool] = None)

   Bases: :py:obj:`torch.distributions.Distribution`


   Refined Normal distribution to approximate Poisson binomial distribution.

   The CDF is defined as:

   .. math::
       F(k; skew) = G( (k + 0.5 - loc) / scale ); \quad G(x) = \Phi(x) + skew * (1 - x^2) * \phi(x) / 6

   where:
   - \Phi(x) is the standard normal CDF
   - \phi(x) is the standard normal PDF
   - skew is the skewness parameter

   The PDF is defined as:

   Args:
       skew: Skewness parameter controlling the third moment correction term.
       validate_args: Whether to validate the arguments.


   .. py:method:: expand(batch_shape, _instance=None)

      Returns a new distribution instance (or populates an existing instance
      provided by a derived class) with batch dimensions expanded to
      `batch_shape`. This method calls :class:`~torch.Tensor.expand` on
      the distribution's parameters. As such, this does not allocate new
      memory for the expanded distribution instance. Additionally,
      this does not repeat any args checking or parameter broadcasting in
      `__init__.py`, when an instance is first created.

      Args:
          batch_shape (torch.Size): the desired expanded size.
          _instance: new instance provided by subclasses that
              need to override `.expand`.

      Returns:
          New distribution instance with batch dimensions expanded to
          `batch_size`.



   .. py:method:: cdf(value)

      Returns the cumulative density/mass function evaluated at
      `value`.

      Args:
          value (Tensor):



   .. py:method:: pdf(value)


   .. py:method:: pmf(value)


   .. py:method:: icdf(p, max_iter=1000, tol=1e-06)

      Inverse CDF (quantile function) using Newton-Raphson method.

      Parameters
      ----------
      p : torch.Tensor
          Probability values (0 < p < 1)
      max_iter : int, optional
          Maximum number of iterations (default: 50)
      tol : float, optional
          Tolerance for convergence (default: 1e-6)
          
      Returns
      -------
      torch.Tensor
          Quantile values corresponding to probabilities p



   .. py:method:: interval(p)

      Compute the confidence interval [lq, uq] such that P(lq <= X <= uq) = 1 - p.



.. py:class:: RefinedNormal(momtype=1, a=None, b=None, xtol=1e-14, badvalue=None, name=None, longname=None, shapes=None, seed=None)

   Bases: :py:obj:`scipy.stats.rv_continuous`


   Refined Normal distribution to approximate Poisson binomial distribution.

   This class extends the continuous random variable class from SciPy to implement
   a modified normal distribution with a skewness correction term. The distribution
   is particularly effective for approximating the Poisson binomial distribution
   (the sum of independent but non-identical Bernoulli random variables).

   The CDF is defined as:
   F(x; skew) = Φ(x) + skew * (1 - x²) * φ(x) / 6

   where:
   - Φ(x) is the standard normal CDF
   - φ(x) is the standard normal PDF
   - skew is the skewness parameter

   Parameters
   ----------
   skew : float
       Skewness parameter controlling the third moment correction term.
       Must be a finite value.

   Notes
   -----
   This refined approximation offers improved accuracy over the standard normal
   approximation.

   In the context of RankSEG, this distribution is used to efficiently compute
   approximate the Poisson binomial distribution.

   References
   ----------
   .. [1] Volkova, A.Y., 1996. A refinement of the central limit theorem for sums 
          of independent random indicators. Theory of Probability and its 
          Applications 40, 791-794.


.. py:function:: rankdice_ba(probs: torch.Tensor, solver: str = 'BA', smooth: float = 0.0, eps: float = 0.0001, pruning_prob: float = 0.5)

   Produce the predicted segmentation by `rankdice` based on the estimated output probability.

   Parameters
   ----------
   probs : Tensor, shape (batch_size, num_class, \*image_shape)
       The estimated probability tensor. 

   solver : str, {'exact', 'TRNA', 'BA', 'BA+TRNA'}
       The approximate algorithm used to implement `RankDice`. 
       `exact` indicates exact evaluation (under development),
       `TRNA` indicates the truncated refined normal approximation (T-RNA), and 
       `BA` indicates the blind approximation (BA),
       `BA+TRNA` indicates a combination of both BA and TRNA.
       
       - we use Cohen's d to determine if we use BA or TRNA
       - if Cohen's d is less than 0.2, we use BA; otherwise, we use TRNA

   smooth : float, default=0.0
       A smooth parameter in the Dice metric.

   eps : float, default=1e-4
       The threshold for truncation of the pmf of posisson-binomial distribution, 
       if the probability is less than `eps`, we truncate it to 0.

   pruning_prob : float, default=0.5
       The threshold for pruning, if all probabilities are less than `pruning_prob`, 
       we skip the class.

   Returns
   -------
   predict : Tensor, shape (batch_size, num_class, \*image_shape)
       The predicted segmentation based on `rankdice`.

   References
   ----------
   Dai, B., & Li, C. (2023). RankSEG: a consistent ranking-based framework for 
   segmentation. Journal of Machine Learning Research, 24(224), 1-50.


.. py:function:: rankseg_rma(probs: torch.Tensor, metric: str = 'dice', return_binary_masks: bool = False, smooth: float = 0.0, pruning_prob: float = 0.1) -> torch.Tensor

   Produce the predicted segmentation by `rankdice` based on the estimated output probability.

   Parameters
   ----------
   probs : Tensor, shape (batch_size, num_class, \*image_shape)
       The estimated probability tensor.

   metric : str, default='dice'
       The metric aim to optimize, either 'iou' or 'dice'.

   return_binary_masks : bool, default=False
       Whether to return or allow binary masks per class (multi-label segmentation).
       If False, performs multi-class segmentation where each pixel belongs to exactly one class.
       If True, performs multi-label segmentation where pixels can belong to multiple classes.

   smooth : float, default=0.0
       A smooth parameter in the Dice metric.

   pruning_prob : float, default=0.1
       The threshold for pruning, if all probabilities are less than `pruning_prob`, 
       we skip the class.

   Returns
   -------
   preds : Tensor
       Shape (batch_size, num_class, \*image_shape) if return_binary_masks is True,
       otherwise shape (batch_size, \*image_shape)


.. py:class:: RankSEG(metric: str = 'dice', smooth: float = 0.0, return_binary_masks: bool = False, solver: str = 'RMA', pruning_prob: float = 0.5, **solver_params)

   Bases: :py:obj:`object`


   Rank-based Segmentation for optimizing segmentation metrics.

   This class provides methods to convert probability maps into binary segmentation
   predictions by optimizing ranking-based metrics like AP, Dice, IoU segmentation metrics.

   Parameters
   ----------
   metric : str, default='dice'
       The segmentation metric to optimize. Currently supported:
       
       - 'dice': Dice coefficient (F1 score)
       - 'IoU': Intersection over Union (not yet implemented)
       - 'AP': Average Precision (not yet implemented)

   smooth : float, default=0.0
       Smoothing parameter added to numerator and denominator to avoid
       division by zero and improve numerical stability.

   return_binary_masks : bool, default=False
       Whether to return or allow binary masks per class (multi-label segmentation). 
       Generally, this is only meaningful when segmentation datasets contain multiple labels.
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

   pruning_prob : float, default=0.5
       Probability threshold for pruning. Classes with maximum probability
       below this threshold may be skipped to improve efficiency.
       Should be in range [0, 1].

   \*\*solver_params : dict
       Additional parameters passed to the specific solver.
       For 'BA', 'TRNA' or 'BA+TRNA': eps (1 - confidence intervals for refined 
       normal approximation of poissoon-binomial distributions)

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


   .. py:method:: predict(probs)

      Convert probability maps to binary segmentation predictions.

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



