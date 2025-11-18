Getting Started
===============

RankSEG provides plug-and-play modules to improve segmentation results during inference by using ranking-based methods that are statistically consistent with popular segmentation metrics.

Installation
------------

Install RankSEG using pip:

.. code-block:: bash

   pip install rankseg

.. raw:: html

  <style>
    /* Make tabs more compact */
    .method-selection .sd-tab-set > input + label {
      padding: 0.2rem 0.7rem;
      font-size: 0.9rem;
      margin: 0 0.35rem 0.35rem 0;  
      border: 1px solid #ddd;
      border-radius: 9999px; /* pill shape */
      line-height: 1.2;
    }

    /* Hover and active states for tabs */
    .method-selection .sd-tab-set > input:checked + label {
      border-color: #964dd1ff;
      color: #964dd1ff;
    }
    
    /* Make task tabs split space equally */
    .method-selection .tabs-task > input + label {
      flex: 1;
      text-align: center;
    }
    
    /* Make metric tabs split space equally */
    .method-selection .tabs-pred > input + label {
      flex: 1;
      text-align: center;
    }
    
    @media screen and (min-width: 960px) {

      .method-selection .sd-tab-set.tabs-task::before {
        content: "Probs Mode:";
        font-weight: 600;
        font-size: 0.85rem;
        padding: 0.2rem 2.6rem 0.3rem 0.6rem;
        display: block;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }

      .method-selection .sd-tab-set.tabs-pred::before {
        content: "Preds Mode:";
        font-weight: 600;
        font-size: 0.85rem;
        padding: 0.2rem 2.6rem 0.3rem 0.6rem;
        display: block;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }
    }
    
    /* Nested tabs styling */
    .method-selection .sd-tab-set .sd-tab-set {
      margin-top: 0.5rem;
      margin-bottom: 0.5rem;
    }
    
  </style>

✨ Quick Start from Model to Prediction
---------------------------------------

In most semantic segmentation problems, models typically output **multiclass** probability maps ``probs`` with shape ``(batch_size, num_classes, *image_shape)``, where probabilities sum to 1 across classes. Our aim is to convert these probabilities into segmentation masks ``preds`` with shape ``(batch_size, *image_shape)``, assigning each pixel to one class, to optimize a given segmentation ``metric``. 

In this case, we can use following code to make segmentation prediction target the Dice/IoU metric.

.. code-block:: python

    import torch
    import torch.nn.functional as F
    from rankseg import RankSEG
    ## input: `images` (batch_size, num_channels, *image_shape) is the input image tensor
    ## output: `preds` (batch_size, *image_shape) is the output binary mask tensor

    # Load your trained segmentation model
    model = torch.load('trained_model.pth')
    model.eval()

    ## `probs` (batch_size, num_classes, *image_shape) is the model output probability tensor
    logits = model(images)
    probs = F.softmax(logits, dim=1)

    # Make segmentation prediction target the Dice metric
    rankseg = RankSEG(metric='dice') ## you can also use `IoU` or `Acc` as the target metric
    pred = rankseg.predict(probs)

The above code handles **99% of semantic segmentation use cases** where we have multiclass probabilities ``probs`` and want non-overlapping predictions (``output_mode='multiclass'``).

However, some scenarios require more advanced configurations, for example:

- **Multilabel probabilities**: When ``probs`` contains independent per-class probabilities (e.g., from sigmoid activation)
- **Overlapping predictions**: When you want ``output_mode='multilabel'`` to allow pixels to belong to multiple classes simultaneously

For these advanced cases, see the examples below organized by task type and target metric.

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
              pred = rankseg.predict(probs)

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
              pred = rankseg.predict(probs)

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
              pred = rankseg.predict(probs)
            
              
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
              pred = rankseg.predict(probs)

.. note::
   For binary segmentation, when ``num_classes=1`` for ``probs``, the ``preds`` output is identical for both ``output_mode='multiclass'`` and ``output_mode='multilabel'``.

**Key Benefits:**

- ✅ **No retraining required** - Works with any pre-trained segmentation model
- ✅ **Metric-aware** - Directly optimizes for your target metric (Dice, IoU, or Accuracy)
- ✅ **Statistically consistent** - Theoretically guaranteed to improve performance
- ✅ **Easy integration** - Just 2 lines of code to add to your inference pipeline



⚙️ Advanced Configuration
-------------------------

This section covers advanced configuration options for RankSEG.

Overlapping or Non-overlapping Segmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RankSEG can produce either overlapping (multilabel) or non-overlapping (multiclass) masks via ``output_mode``, no matter what the mode of the input ``probs`` is:

- **Non-overlapping (multiclass)**: Set ``output_mode='multiclass'``. Each pixel belongs to exactly one class.
  Output shape: ``(batch, *image_shape)``.

- **Overlapping (multilabel)**: Set ``output_mode='multilabel'``. Pixels may belong to multiple classes.
  Output shape: ``(batch, num_classes, *image_shape)``.


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

A: No! RankSEG is a post-processing method that works with any pre-trained segmentation model. Simply apply it to your model's probability outputs during inference.

----

**Have more questions?** Contact us at bendai@cuhk.edu.hk


Next Steps
----------

- Check out the :doc:`API Reference </API>` for detailed parameter descriptions
- See :doc:`Citation </citation>` for how to cite RankSEG in your research
- Report issues or contribute on `GitHub <https://github.com/rankseg/rankseg>`_
              