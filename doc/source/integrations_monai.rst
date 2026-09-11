MONAI
=====

RankSEG is featured in the official `MONAI Tutorials
<https://github.com/Project-MONAI/tutorials/blob/main/modules/rankseg_integration.ipynb>`_
as an optional third-party post-processing method. The tutorial shows how to
wrap RankSEG as MONAI-style array and dictionary transforms and insert them into
a standard decollated inference pipeline.

.. important::

   The tutorial defines ``RankSEG`` and ``RankSEGd`` wrappers for integration
   with MONAI workflows. Their inclusion in the official MONAI Tutorials does
   not mean that RankSEG is built into every MONAI release. Install the
   ``rankseg`` package explicitly and follow the tutorial for the integration
   shown there.

What the tutorial covers
------------------------

- loading the public ``pancreas_ct_dints_segmentation`` MONAI Bundle;
- running inference on a 3D CT volume from Medical Segmentation Decathlon
  Task07 Pancreas;
- applying RankSEG and RankSEGd after softmax probabilities are produced;
- comparing RankSEG with ``AsDiscrete(argmax=True)`` on the same probabilities;
- preserving MONAI's channel-first and dictionary-transform conventions.

The model remains frozen: RankSEG replaces only the final probability-to-mask
decision and requires no retraining or method-specific fine-tuning.

Run the tutorial
----------------

- `View the notebook in the official MONAI Tutorials repository
  <https://github.com/Project-MONAI/tutorials/blob/main/modules/rankseg_integration.ipynb>`_
- `Open the notebook in Google Colab
  <https://colab.research.google.com/github/Project-MONAI/tutorials/blob/main/modules/rankseg_integration.ipynb>`_

Interpreting the comparison
---------------------------

The notebook demonstrates the integration and a paired comparison on a real
medical segmentation workflow. RankSEG's effect depends on the model's
probability quality, the selected metric, and the deployment distribution.
Validate both metric changes and post-processing cost on representative data
before adopting it in production or reporting a general performance claim.
