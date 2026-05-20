Executable Notebooks
====================

These notebooks are maintained as user-facing tutorials. They are intended to
be run from top to bottom and to show the exact point where RankSEG replaces a
standard segmentation prediction step.

.. list-table::
   :widths: 24 42 34
   :header-rows: 1

   * - Notebook
     - What it teaches
     - Best entry point
   * - `quickstart.ipynb <https://github.com/rankseg/rankseg/blob/main/notebooks/quickstart.ipynb>`__
     - A short PyTorch walkthrough with a pretrained DeepLabV3-style workflow:
       model output, probability map, baseline mask, RankSEG mask.
     - New users who want the fastest runnable example.
   * - `rankseg_with_transformers.ipynb <https://github.com/rankseg/rankseg/blob/main/notebooks/rankseg_with_transformers.ipynb>`__
     - A Hugging Face tutorial that first runs the official baseline and then
       changes only the final post-processing step to RankSEG.
     - Users with ``processor -> model -> outputs`` code.
   * - `rankseg_with_sam_family.ipynb <https://github.com/rankseg/rankseg/blob/main/notebooks/rankseg_with_sam_family.ipynb>`__
     - A SAM-family tutorial covering SAM1, SAM2, and SAM3 adapters, including
       geometry restoration before RankSEG is called.
     - Users working with prompt, instance, or semantic masks from SAM-family
       models.
   * - `rankseg_with_paddleseg.ipynb <https://github.com/rankseg/rankseg/blob/main/notebooks/rankseg_with_paddleseg.ipynb>`__
     - A PaddleSeg-oriented walkthrough that shows the probability conversion
       step before calling the PyTorch-based RankSEG predictor.
     - PaddleSeg users evaluating the external/community-maintained path.

Colab links
-----------

- `Open quickstart.ipynb in Colab <https://colab.research.google.com/github/rankseg/rankseg/blob/main/notebooks/quickstart.ipynb>`_
- `Open rankseg_with_transformers.ipynb in Colab <https://colab.research.google.com/github/rankseg/rankseg/blob/main/notebooks/rankseg_with_transformers.ipynb>`_
- `Open rankseg_with_sam_family.ipynb in Colab <https://colab.research.google.com/github/rankseg/rankseg/blob/main/notebooks/rankseg_with_sam_family.ipynb>`_
- `Open rankseg_with_paddleseg.ipynb in Colab <https://colab.research.google.com/github/rankseg/rankseg/blob/main/notebooks/rankseg_with_paddleseg.ipynb>`_

How to read the notebooks with the docs
---------------------------------------

Use :doc:`integrations` first to choose the correct backend. Then open the
matching notebook when you want a runnable, visual workflow. The conceptual
pages define the tensor shapes, probability conventions, option names, and
supported output families; the notebooks show those contracts in complete
inference code.
