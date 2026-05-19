.. _api:

.. meta::
   :description: Browse the rankseg API documentation.

=================
API
=================

Module Interface
================

.. autoapisummary::

   rankseg.RankSEG


Integration Interface
=====================

.. autoapisummary::

   rankseg.integration.transformers.postprocess
   rankseg.integration.transformers.restore_semantic_probs
   rankseg.integration.sam.Sam1
   rankseg.integration.sam.Sam2
   rankseg.integration.sam.Sam3


Algorithms
==========

.. autoapisummary::

   rankseg.rankdice_ba
   rankseg.rankseg_rma


Distribution Interface
======================

.. autoapisummary::

   rankseg.RefinedNormalPB
   rankseg.RefinedNormal


.. toctree::
   :hidden:

   autoapi/rankseg/rankseg/index
