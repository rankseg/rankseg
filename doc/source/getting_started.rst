Getting Started
===============

RankSEG helps you get better segmentation masks from your models. It's a simple post-processing tool that improves your predictions during inference by using smart ranking methods, making your results more accurate for common segmentation metrics like Dice and IoU—without requiring any model retraining.

Installation
------------

Install RankSEG using pip:

.. code-block:: bash

   pip install rankseg

Why RankSEG?
------------

Standard approaches like argmax (for multiclass) or 0.5 thresholding (for binary/multilabel) don't directly optimize for segmentation metrics like Dice or IoU. RankSEG bridges this gap by using statistically consistent ranking methods that are specifically designed to maximize your target metric. This means better segmentation quality, especially when:

- Your model outputs uncertain probabilities
- You're working with complex segmentation tasks
- Ground truth regions are small to medium-sized

✨ Quick Start from Model to Prediction
---------------------------------------

In most semantic segmentation problems, models typically output **multiclass** probability maps ``probs`` with shape ``(batch_size, num_classes, *image_shape)``, where probabilities sum to 1 across classes. Our aim is to convert these probabilities into segmentation masks ``preds`` with shape ``(batch_size, *image_shape)``, assigning each pixel to one class, to optimize a given segmentation ``metric``.

.. note::
   RankSEG expects probabilities in range ``[0, 1]``. If your model outputs raw logits (unbounded values), apply the appropriate activation function first: ``torch.softmax(logits, dim=1)`` for multiclass or ``torch.sigmoid(logits)`` for multilabel/binary segmentation.

Here's how to use RankSEG to make segmentation predictions that target the Dice/IoU metric:

.. code-block:: python

    import torch
    import torch.nn.functional as F
    from rankseg import RankSEG
    ## input: `images` (batch_size, num_channels, *image_shape) is the input image tensor
    ## output: `preds` (batch_size, *image_shape) is the output binary mask tensor
    
    # Load your trained segmentation model
    model = torch.load('trained_model.pth')
    model.eval()

    # Get probability predictions from your model
    # probs shape: (batch_size, num_classes, *image_shape)
    logits = model(images)
    probs = F.softmax(logits, dim=1)

    # Make segmentation prediction targeting the Dice metric
    rankseg = RankSEG(metric='dice')  # or 'iou', 'acc'
    preds = rankseg.predict(probs)    # shape: (batch_size, *image_shape)

The above code handles **99% of semantic segmentation use cases** where we have multiclass probabilities ``probs`` and want non-overlapping predictions (``output_mode='multiclass'``).

**Key Benefits:**

- ✅ **No retraining required** - Works with any pre-trained logit/prob-outcome segmentation model
- ✅ **Metric-aware** - Directly optimizes for your target metric (Dice, IoU, or Accuracy)
- ✅ **Statistically consistent** - Theoretically guaranteed to improve performance
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
              import torch.nn.functional as F
              from rankseg import RankSEG
              ## input: `images` (batch_size, num_channels, *image_shape) is the input image tensor
              ## output: `preds` (batch_size, num_classes, *image_shape) is the output binary mask tensor

              # Load your trained segmentation model
              model = torch.load('trained_model.pth')
              model.eval()

              ## `probs` (batch_size, num_classes=1, *image_shape) is the model output probability tensor
              logits = model(images)
              probs = F.sigmoid(logits)
              
              # Make segmentation prediction target the Dice metric
              ## you can also use `IoU` or `Acc` as the target metric
              rankseg = RankSEG(metric='dice', output_mode='multiclass')
              preds = rankseg.predict(probs)  # (batch, *image_shape)

.. note::
   For binary segmentation, when ``num_classes=1`` for ``probs``, the ``preds`` output is identical for both ``output_mode='multiclass'`` and ``output_mode='multilabel'``.


⚙️ Advanced Configuration
-------------------------

Output Mode: Overlapping vs Non-overlapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RankSEG can produce either overlapping (multilabel) or non-overlapping (multiclass) masks via ``output_mode``, regardless of the input ``probs`` mode:

- **Non-overlapping (multiclass)**: Set ``output_mode='multiclass'``. Each pixel belongs to exactly one class.
  
  - Output shape: ``(batch, *image_shape)``
  - Use for: Standard semantic segmentation where classes are mutually exclusive

- **Overlapping (multilabel)**: Set ``output_mode='multilabel'``. Pixels may belong to multiple classes.
  
  - Output shape: ``(batch, num_classes, *image_shape)``
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
     - Blind Approximation. Best for Dice metric when speed is critical. Requires ``eps`` (error tolerance for the normal approximation).
   * - ``'TRNA'``
     - Dice
     - ``'multilabel'``
     - Slow
     - Truncated Refined Normal Approximation. More accurate than BA for complex cases. Requires ``eps`` (error tolerance for the normal approximation).
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
     - ``'multiclass'``
     - Fastest
     - Argmax solver: argmax over classes.

Example with solver parameters:

.. code-block:: python

   from rankseg import RankSEG

   # RMA solver (default, works for all metrics)
   rankseg = RankSEG(metric='dice', solver='RMA')

   # BA solver with custom epsilon
   rankseg = RankSEG(metric='dice', solver='BA', eps=1e-4)

   # Automatic solver selection
   rankseg = RankSEG(metric='dice', solver='BA+TRNA', eps=1e-4)


GPU Acceleration
----------------

RankSEG automatically uses GPU if your ``probs`` tensors are on GPU:

.. code-block:: python

   import torch
   import torch.nn.functional as F
   from rankseg import RankSEG

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   # Move model and data to GPU
   model = model.to(device)
   images = images.to(device)

   # Get predictions (RankSEG will use GPU automatically)
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

5. **Enable GPU acceleration**: For large images or batches, ensure your tensors are on GPU for faster processing.

❓ FAQ
------

**Q: My predictions look the same as argmax/threshold?**

A: This can happen in two scenarios:
   - **Confident predictions**: When your model's probabilities are already very confident or the segmentation task is simple, RankSEG provides minimal improvement. It works best with uncertain probabilities or complex tasks.

   - **Large segmentation regions**: When the ground truth region is very large, predictions may appear similar to argmax/threshold. This occurs because Dice/IoU metrics are less sensitive to large regions (due to the large denominator). To verify RankSEG's effectiveness, check predictions on images with smaller ground truth regions.

**Q: Should I use multiclass or multilabel mode?**

A: Use multiclass (``output_mode='multiclass'``) when classes are mutually exclusive (e.g., semantic segmentation). Use multilabel (``output_mode='multilabel'``) when objects can overlap (e.g., instance segmentation, medical imaging).

**Q: Which solver should I choose?**

A: Start with ``'RMA'`` for most cases. For Dice metric on small images, try ``'BA'``, ``'TRNA'``, or ``'BA+TRNA'`` for potentially better accuracy. For Accuracy metric, use ``'argmax'`` for multiclass or ``'TR'`` for multilabel.

**Q: What does the ``eps`` parameter do?**

A: The ``eps`` parameter controls the error tolerance for normal approximation in BA, TRNA, and BA+TRNA solvers. Smaller values (e.g., ``1e-5``) give more accurate results but slower computation. Default is ``1e-4``.

**Q: Can I use RankSEG with binary segmentation?**

A: Yes! For binary segmentation, set your ``probs`` shape to ``(batch, 1, *image_shape)`` and use ``output_mode='multiclass'`` or ``output_mode='multilabel'`` (they produce identical results for binary cases).

**Q: Does RankSEG require retraining my model?**

A: No! RankSEG is a post-processing method that works with any pre-trained logit/prob-outcome segmentation model. Simply apply it to your model's probability outputs during inference.

----

**Have more questions?** Contact us at bendai@cuhk.edu.hk


Next Steps
----------

- Check out the :doc:`API Reference </API>` for detailed parameter descriptions
- See :doc:`Citation </citation>` for how to cite RankSEG in your research
- Report issues or contribute on `GitHub <https://github.com/rankseg/rankseg>`_
