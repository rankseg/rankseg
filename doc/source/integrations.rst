Official Integrations
=====================

RankSEG is designed to fit into existing inference pipelines with minimal code
changes. This section collects the maintained integration paths officially
documented in this repository.

Available integrations
----------------------

.. toctree::
   :maxdepth: 1

   integrations_pytorch
   integrations_paddleseg

Guiding principle
-----------------

At the integration layer, the goal is to help users adopt RankSEG in the
framework they already use.

Solver details such as ``RMA`` remain part of the recommended configuration
inside each integration page, rather than the top-level navigation label.

At the moment, ``PyTorch Native`` is the main first-party maintained entry
point documented here, while ``PaddleSeg`` is provided as an
external/community-maintained path.
