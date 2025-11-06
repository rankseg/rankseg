Getting Started
===============

RankSEG provides plug-and-play modules to improve segmentation results during inference by using ranking-based methods that are statistically consistent with popular segmentation metrics.

.. raw:: html

  <style>
    /* Improved compact layout for tabs */
    .method-selection .sd-tab-set {
      margin-bottom: 1rem;
    }
    
    /* Make tabs more compact */
    .method-selection .sd-tab-set > input + label {
      padding: 0.5rem 1rem;
      font-size: 0.95rem;
    }
    
    /* Reduce spacing in tab content */
    .method-selection .sd-tab-content {
      padding: 1rem 0.5rem;
    }
    
    @media screen and (min-width: 960px) {
      .method-selection .sd-tab-set {
        --tab-caption-width: 20%;
      }

      .method-selection .sd-tab-set.tabs-task::before {
        content: "Task:";
        font-weight: 600;
        font-size: 0.85rem;
        padding: 0.4rem 0.5rem 0.3rem 0.5rem;
        display: block;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }

      .method-selection .sd-tab-set.tabs-metric::before {
        content: "Metric:";
        font-weight: 600;
        font-size: 0.85rem;
        padding: 0.4rem 0.5rem 0.3rem 0.5rem;
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

Suppose you have already trained a segmentation model ``model`` (for example, a U-Net trained with cross-entropy loss), and you want to use it to predict the segmentation results for a new images ``inputs``.

.. container:: method-selection

  .. tab-set::
    :class: tabs-task

    .. tab-item:: Binary segmentation
      :class-label: task-binary

      .. tab-set::
        :class: tabs-metric

        .. tab-item:: AP
          :class-label: metric-ap
                    
          .. code-block:: python
          
              import torch
              from rankseg import RankSEG
              ## `probs` (batch_size, num_classes, *image_shape) is the input probability tensor
              
              # Make segmentation prediction target the AP metric
              rankseg = RankSEG(metric='AP')
              pred = rankseg.predict(probs)
              
        .. tab-item:: Dice
          :class-label: metric-dice
          
          .. code-block:: python
          
              import torch
              from rankseg import RankSEG
              ## `probs` (batch_size, num_classes, *image_shape) is the input probability tensor
              
              # Make segmentation prediction target the Dice metric
              rankseg = RankSEG(metric='Dice')
              pred = rankseg.predict(probs)
              
        .. tab-item:: IoU
          :class-label: metric-iou
                    
          .. code-block:: python
          
              import torch
              from rankseg import RankSEG
              ## `probs` (batch_size, num_classes, *image_shape) is the input probability tensor
              
              # Make segmentation prediction target the IoU metric
              rankseg = RankSEG(metric='IoU')
              pred = rankseg.predict(probs)

    .. tab-item:: Multiclass segmentation
      :class-label: task-multiclass

      .. tab-set::
        :class: tabs-metric

        .. tab-item:: AP
          :class-label: metric-ap
                    
          .. code-block:: python
          
              import torch
              from rankseg import RankSEG
              ## `probs` (batch_size, num_classes, *image_shape) is the input probability tensor
              
              # Make segmentation prediction target the AP metric
              rankseg = RankSEG(metric='AP')
              pred = rankseg.predict(probs)

        .. tab-item:: Dice
          :class-label: metric-dice
          
          .. code-block:: python
          
              import torch
              from rankseg import RankSEG
              ## `probs` (batch_size, num_classes, *image_shape) is the input probability tensor
              
              # Make segmentation prediction target the Dice metric
              rankseg = RankSEG(metric='Dice')
              pred = rankseg.predict(probs)

        .. tab-item:: IoU
          :class-label: metric-iou
                    
          .. code-block:: python
          
              import torch
              from rankseg import RankSEG
              ## `probs` (batch_size, num_classes, *image_shape) is the input probability tensor
              
              # Make segmentation prediction target the IoU metric
              rankseg = RankSEG(metric='IoU')
              pred = rankseg.predict(probs)
              