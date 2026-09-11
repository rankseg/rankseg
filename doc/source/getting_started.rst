Getting Started
===============

RankSEG is a metric-aware post-processing tool designed to improve samplewise
Dice or IoU during inference without model retraining. Its effect depends on
the quality of the input probabilities and should be validated on representative
data.

Official Integration Paths
--------------------------

If you want the shortest route from an existing inference pipeline to RankSEG,
start with the official integrations overview:

- :doc:`integrations`

The first maintained integration path is ``PyTorch Native``, which is the
recommended starting point for users who already have a PyTorch semantic-
segmentation model and want to replace ``argmax`` with RankSEG.

If you already run Hugging Face semantic-segmentation models through
``processor -> model -> outputs``, start with :doc:`integrations_transformers`.
For SAM-family outputs, start with :doc:`integrations_sam`.
For a medical-imaging workflow using MONAI transforms, start with the official
tutorial described in :doc:`integrations_monai`.

Installation
------------

Install RankSEG using pip:

.. code-block:: bash

   pip install rankseg

Why RankSEG?
------------

Standard approaches like argmax (for multiclass) or 0.5 thresholding (for binary/multilabel) don't directly optimize for segmentation metrics like Dice or IoU. RankSEG bridges this gap with statistically consistent ranking methods designed to target the selected metric. Improvements are most likely to be useful when:

- Your model outputs uncertain probabilities
- You're working with complex segmentation tasks
- Ground truth regions are small to medium-sized

✨ Quick Start from Model to Prediction
---------------------------------------

In most semantic segmentation problems, models typically output **multiclass** probability maps ``probs`` with shape ``(batch_size, num_classes, *image_shape)``, where probabilities sum to 1 across classes. Our aim is to convert these probabilities into segmentation masks ``preds`` with shape ``(batch_size, *image_shape)``, assigning each pixel to one class, to optimize a given segmentation ``metric``.

.. note::
   RankSEG expects probabilities in range ``[0, 1]``. If your model outputs raw logits (unbounded values), apply the appropriate activation function first: ``torch.softmax(logits, dim=1)`` for multiclass or ``torch.sigmoid(logits)`` for multilabel/binary segmentation.

.. note::
   RankSEG accepts float16, bfloat16, float32, and float64 probability tensors.
   Solver calculations promote float16 and bfloat16 to float32 for numerical
   stability, while float32 and float64 retain their supplied precision. Output
   masks still use the boolean or integer dtype defined by ``output_mode``.

Here's how to use RankSEG to make segmentation predictions that target the Dice/IoU metric:

.. code-block:: python

    import torch
    import torch.nn.functional as F
    from rankseg import RankSEG
    ## input: `images` (batch_size, num_channels, *image_shape) is the input image tensor
    ## output: `preds` (batch_size, *image_shape) is an integer class-index map

    # Load your trained segmentation model
    model = torch.load('trained_model.pth')
    model.eval()

    # Get probability predictions from your model
    # probs shape: (batch_size, num_classes, *image_shape)
    logits = model(images)
    probs = F.softmax(logits, dim=1)

    # Make segmentation prediction targeting the Dice metric
    rankseg = RankSEG(metric='dice')  # use metric='iou' for IoU
    preds = rankseg(probs)            # shape: (batch_size, *image_shape)

The above code covers the common semantic segmentation case where multiclass
probabilities ``probs`` should become non-overlapping predictions
(``output_mode='multiclass'``). The older ``rankseg.predict(probs)`` form remains
supported.

You can also use the functional API for one-off prediction:

.. code-block:: python

    from rankseg.functional import rankseg

    preds = rankseg(probs, metric='dice')  # shape: (batch_size, *image_shape)

.. note::
   RankSEG is a discrete inference-time post-processing operation. Its public
   prediction functions disable PyTorch gradient recording internally, even
   when ``probs.requires_grad`` is ``True``. Returned boolean masks and integer
   class maps therefore do not participate in backpropagation.

**Key Benefits:**

- ✅ **No retraining required** - Works with pre-trained probabilistic segmentation models
- ✅ **Metric-aware** - Directly optimizes for your target metric (Dice, IoU, or Accuracy)
- ✅ **Statistically grounded** - Consistent for the target metric under the method's assumptions
- ✅ **Easy integration** - Just 2 lines of code to add to your inference pipeline

Advanced Use Cases
~~~~~~~~~~~~~~~~~~

Some scenarios require more advanced configurations:

- **Multilabel probabilities**: When ``probs`` contains independent per-class probabilities (e.g., from sigmoid activation)
- **Overlapping predictions**: When you want ``output_mode='multilabel'`` to allow pixels to belong to multiple classes simultaneously

For these cases, see the examples below organized by probability type and desired output mode.

.. container:: method-selection

  .. tab-set::
    :class: tabs-task


    .. tab-item:: ``multiclass`` (softmax activation)
      :class-label: task-multiclass

      .. tab-set::
        :class: tabs-pred

        .. tab-item:: ``multiclass`` (non-overlapping)
          :class-label: pred-multiclass

          .. code-block:: python

              import torch
              import torch.nn.functional as F
              from rankseg import RankSEG
              ## input: `images` (batch_size, num_channels, *image_shape) is the input image tensor
              ## output: `preds` (batch_size, *image_shape) is the output mask tensor

              # Load your trained segmentation model
              model = torch.load('trained_model.pth')
              model.eval()

              ## `probs` (batch_size, num_classes, *image_shape) is the model output probability tensor
              logits = model(images)
              probs = F.softmax(logits, dim=1)

              # Make segmentation prediction target the Dice metric
              ## you can also use `IoU` or `Acc` as the target metric
              rankseg = RankSEG(metric='dice', output_mode='multiclass')
              preds = rankseg.predict(probs)  # (batch, *image_shape)

        .. tab-item:: ``multilabel`` (overlapping)
          :class-label: pred-multilabel

          .. code-block:: python

              import torch
              import torch.nn.functional as F
              from rankseg import RankSEG
              ## input: `images` (batch_size, num_channels, *image_shape) is the input image tensor
              ## output: `preds` (batch_size, num_classes, *image_shape) is the binary mask per class output tensor

              # Load your trained segmentation model
              model = torch.load('trained_model.pth')
              model.eval()

              ## `probs` (batch_size, num_classes, *image_shape) is the model output probability tensor
              logits = model(images)
              probs = F.softmax(logits, dim=1)

              # Make segmentation prediction target the Dice metric
              rankseg = RankSEG(metric='dice', output_mode='multilabel')
              preds = rankseg.predict(probs)  # (batch, num_classes, *image_shape)

    .. tab-item:: ``multilabel`` (sigmoid activation)
      :class-label: task-multilabel

      .. tab-set::
        :class: tabs-pred

        .. tab-item:: ``multilabel`` (overlapping)
          :class-label: pred-multilabel

          .. code-block:: python

              import torch
              import torch.nn.functional as F
              from rankseg import RankSEG
              ## input: `images` (batch_size, num_channels, *image_shape) is the input image tensor
              ## output: `preds` (batch_size, num_classes, *image_shape) is the output binary mask tensor

              # Load your trained segmentation model
              model = torch.load('trained_model.pth')
              model.eval()

              ## `probs` (batch_size, num_classes, *image_shape) is the model output probability tensor
              logits = model(images)
              probs = F.sigmoid(logits)

              # Make segmentation prediction target the Dice metric
              ## you can also use `IoU` or `Acc` as the target metric
              rankseg = RankSEG(metric='dice', output_mode='multilabel')
              preds = rankseg.predict(probs)  # (batch, num_classes, *image_shape)


        .. tab-item:: ``multiclass`` (non-overlapping)
          :class-label: pred-multilabel

          .. code-block:: python

              import torch
              from rankseg import RankSEG
              ## input: `images` (batch_size, num_channels, *image_shape) is the input image tensor
              ## output: `preds` (batch_size, *image_shape) is the output 0/1 class-index tensor

              # Load your trained segmentation model
              model = torch.load('trained_model.pth')
              model.eval()

              ## `foreground_probs` (batch_size, 1, *image_shape) is the model output probability tensor
              logits = model(images)
              foreground_probs = torch.sigmoid(logits)

              # Multiclass output needs explicit background and foreground channels.
              probs = torch.cat((1 - foreground_probs, foreground_probs), dim=1)

              # Make segmentation prediction target the Dice metric
              ## you can also use `IoU` or `Acc` as the target metric
              rankseg = RankSEG(metric='dice', output_mode='multiclass')
              preds = rankseg.predict(probs)  # (batch, *image_shape)

.. note::
   For binary segmentation, a single probability channel represents the foreground
   mask and should use ``output_mode='multilabel'``. With RMA,
   ``output_mode='multiclass'`` returns class indices; a one-channel input has only
   class index 0 and therefore cannot represent foreground label 1. To obtain a
   multiclass 0/1 label map, pass two channels ``torch.cat((1 - probs, probs), dim=1)``.


⚙️ Advanced Configuration
-------------------------

Output Mode: Overlapping vs Non-overlapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RankSEG can produce either overlapping (multilabel) or non-overlapping (multiclass) masks via ``output_mode``, regardless of the input ``probs`` mode:

- **Non-overlapping (multiclass)**: Set ``output_mode='multiclass'``. Each pixel belongs to exactly one class. With RMA, the per-class binary masks are converted using the rules below.

  - Output shape: ``(batch, *image_shape)``
  - Output dtype: ``torch.int64`` (class indices)
  - Use for: Standard semantic segmentation where classes are mutually exclusive

- **Overlapping (multilabel)**: Set ``output_mode='multilabel'``. Pixels may belong to multiple classes. RMA returns its per-class binary masks directly and does not resolve overlaps.

  - Output shape: ``(batch, num_classes, *image_shape)``
  - Output dtype: ``torch.bool`` (binary mask per class)
  - Use for: Instance segmentation, medical imaging, or when objects can overlap


Example:

.. code-block:: python

   from rankseg import RankSEG

   # Non-overlapping masks (multi-class)
   rankseg = RankSEG(metric='dice', output_mode='multiclass')
   preds = rankseg.predict(probs)  # (batch, *image_shape)

   # Overlapping masks (multi-label)
   rankseg = RankSEG(metric='dice', output_mode='multilabel')
   preds = rankseg.predict(probs)  # (batch, num_classes, *image_shape)


How RMA converts overlapping masks to multiclass output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RMA first optimizes one binary mask per class. Converting those masks to
``output_mode='multiclass'`` uses the number of masks that selected each pixel:

.. list-table::
   :widths: 18 34 48
   :header-rows: 1

   * - Selecting masks
     - Eligible classes
     - Multiclass result
   * - Exactly one
     - The selecting class
     - Use that class directly.
   * - Two or more
     - Only the classes whose binary masks selected the pixel
     - Use the eligible class with the largest incremental score. A class that
       did not select the pixel cannot win, even if its unconstrained score or
       probability is larger.
   * - None
     - Active classes when ``unassigned_policy='max_score'``
     - Use the active class with the largest incremental score. With
       ``unassigned_policy='void'``, use ``void_index`` instead. The void index
       must fit in ``torch.int64`` and lie outside ``[0, num_classes)`` so it
       cannot be confused with a valid class prediction.

Here, an *active class* has a maximum probability strictly greater than
``pruning_prob``; a class whose probabilities are all less than or equal to the
threshold is pruned. If every class in a sample is pruned, the ``max_score``
fallback reconsiders all classes for otherwise unassigned pixels. This fallback
does not change the overlap rule: a pruned class cannot have selected a pixel.

The incremental score is the RMA objective change from assigning the pixel to
a class after fixing pixels selected by exactly one mask. It is not simply the
pixel probability or a probability ``argmax``. Equal scores are resolved by
the lowest eligible class index.

For example, suppose the masks selecting a pixel are
``[False, True, True]``. Classes 1 and 2 compete by incremental score; class 0
is ineligible even if it has the largest unconstrained score. This conversion
is skipped entirely for ``output_mode='multilabel'``, where both selected masks
remain ``True``.


Solver Selection
~~~~~~~~~~~~~~~~

RankSEG offers multiple solver algorithms, each optimized for specific metrics and output modes:

.. list-table::
   :widths: 15 15 15 15 55
   :header-rows: 1

   * - Solver
     - Metrics
     - Output Mode
     - Speed
     - Description
   * - ``'RMA'``
     - Dice, IoU
     - ``'multiclass'``, ``'multilabel'``
     - Fastest
     - **Recommended for most cases.** Reciprocal Moment Approximation. Works for both binary and multiclass segmentation. Good balance of speed and accuracy.
   * - ``'BA'``
     - Dice
     - ``'multilabel'``
     - Fast
     - Blind Approximation. Best for Dice metric when speed is critical. ``eps`` sets the tail probability excluded from the retained refined-normal interval.
   * - ``'TRNA'``
     - Dice
     - ``'multilabel'``
     - Slow
     - Truncated Refined Normal Approximation. More accurate than BA for complex cases. ``eps`` sets the tail probability excluded from the retained refined-normal interval.
   * - ``'BA+TRNA'``
     - Dice
     - ``'multilabel'``
     - Fast (adaptive)
     - Automatically selects between BA and TRNA based on data characteristics using Cohen's d.
   * - ``'TR'``
     - Acc
     - ``'multilabel'``
     - Fastest
     - Truncation solver: truncate at 0.5 threshold for binary and multilabel.
   * - ``'argmax'``
     - Acc
     - ``'multiclass'``, ``'multilabel'``
     - Fastest
     - Argmax solver: argmax over classes. In ``'multilabel'`` mode it returns
       non-overlapping one-hot masks in multilabel tensor format.

Example with solver parameters:

.. code-block:: python

   from rankseg import RankSEG

   # RMA solver (default; supports Dice and IoU)
   rankseg = RankSEG(metric='dice', solver='RMA')

   # BA solver with custom epsilon
   rankseg = RankSEG(metric='dice', solver='BA', output_mode='multilabel', eps=1e-4)

   # Automatic solver selection
   rankseg = RankSEG(metric='dice', solver='BA+TRNA', output_mode='multilabel', eps=1e-4)

   # Accuracy requires an explicit compatible solver
   rankseg = RankSEG(metric='accuracy', solver='argmax', output_mode='multiclass')

Solver compatibility is checked strictly. RankSEG raises an error for an
unsupported metric, output mode, and solver combination instead of silently
substituting another solver.

``smooth`` and ``pruning_prob`` affect the Dice and IoU solvers only. Accuracy
uses ``argmax`` or a fixed ``0.5`` threshold, so valid values supplied for
``smooth`` and ``pruning_prob`` through the shared API do not affect Accuracy
predictions.


Device behavior
---------------

RankSEG returns predictions on the same device as ``probs``. RMA runs natively
on that device. For Dice multilabel solvers on an accelerator, BA retains
ranking and convolution on the input device. TRNA and BA+TRNA also retain
ranking and scoring there, while their SciPy-backed probability calculations
are staged through the CPU and copied back in bounded chunks to avoid a device
transfer and synchronization for every search step. This applies to accelerator
backends such as CUDA, MPS, and XPU; backend support for individual floating-point
dtypes still follows PyTorch.

.. code-block:: python

   import torch
   import torch.nn.functional as F
   from rankseg import RankSEG

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   # Move model and data to GPU
   model = model.to(device)
   images = images.to(device)

   # RMA runs on the same device as the probabilities
   logits = model(images)
   probs = F.softmax(logits, dim=1)

   rankseg = RankSEG(metric='dice', solver='RMA')
   preds = rankseg.predict(probs)  # Computed on GPU

Best Practices
--------------

1. **Choose the right metric**: Use the same metric you'll evaluate your model with (Dice, IoU, or Acc).

2. **Select output mode**: Decide whether ``preds`` should allow overlapping classes (``output_mode='multilabel'``) or non-overlapping classes (``output_mode='multiclass'``).

3. **Start with RMA solver**: It works well for both Dice and IoU metrics and provides good speed-accuracy balance, especially for large images.

4. **For small images**: Consider using BA, TRNA, or BA+TRNA solvers for Dice metric to achieve better accuracy.

5. **Choose devices deliberately**: RMA and large BA problems can benefit from
   CUDA. TRNA is sequential and may remain faster on CPU for small inputs, so
   benchmark representative shapes in latency-sensitive pipelines.

❓ FAQ
------

**Q: My predictions look the same as argmax/threshold?**

A: This can happen in two scenarios:
   - **Confident predictions**: When your model's probabilities are already very confident or the segmentation task is simple, RankSEG provides minimal improvement. It works best with uncertain probabilities or complex tasks.

   - **Large segmentation regions**: When the ground truth region is very large, predictions may appear similar to argmax/threshold. This occurs because Dice/IoU metrics are less sensitive to large regions (due to the large denominator). To verify RankSEG's effectiveness, check predictions on images with smaller ground truth regions.

**Q: Should I use multiclass or multilabel mode?**

A: Use multiclass (``output_mode='multiclass'``) when classes are mutually exclusive (e.g., semantic segmentation). Use multilabel (``output_mode='multilabel'``) when objects can overlap (e.g., instance segmentation, medical imaging).

**Q: Which solver should I choose?**

A: Start with ``'RMA'`` for Dice or IoU. For Dice metric on small images, try
``'BA'``, ``'TRNA'``, or ``'BA+TRNA'`` with ``output_mode='multilabel'`` for
potentially better accuracy. For Accuracy metric, explicitly use ``'argmax'``
for multiclass or ``'TR'`` for multilabel.

**Q: What does the ``eps`` parameter do?**

A: For BA, TRNA, and BA+TRNA, ``eps`` is the tail probability excluded when
retaining the central ``1 - eps`` refined-normal interval. Smaller values
(e.g., ``1e-5``) retain wider PMF support, reducing truncation at the cost of
additional computation. The default is ``1e-4``.

**Q: Can I use RankSEG with binary segmentation?**

A: Yes. For a single foreground-probability channel with shape
   ``(batch, 1, *image_shape)``, use ``output_mode='multilabel'`` to obtain a
   binary foreground mask. If you need a ``(batch, *image_shape)`` multiclass
   label map containing labels 0 and 1, construct two-channel background/foreground
   probabilities with ``torch.cat((1 - probs, probs), dim=1)`` and use
   ``output_mode='multiclass'``.

**Q: Does RankSEG require retraining my model?**

A: No. RankSEG is a post-processing method for pre-trained probabilistic
   segmentation models. Apply it to the model's probability outputs during
   inference.

----

**Have more questions?** Contact us at bendai@cuhk.edu.hk


Next Steps
----------

- Check out the :doc:`API Reference </API>` for detailed parameter descriptions
- See :doc:`Citation </citation>` for how to cite RankSEG in your research
- Report issues or contribute on `GitHub <https://github.com/rankseg/rankseg>`_
