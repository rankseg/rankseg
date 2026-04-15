PaddleSeg
=========

This page documents the current PaddleSeg integration status for RankSEG.

Current status
--------------

The PaddleSeg integration already exists, but it is currently maintained
outside the main ``rankseg`` branch.

At this stage, the main repository provides an entry point to that work rather
than duplicating or re-implementing the full PaddleSeg integration locally.

Who should use this path
------------------------

This path is useful if:

- you already deploy segmentation pipelines in PaddleSeg;
- you want to evaluate RankSEG as an inference-time post-processing module in
  the Paddle ecosystem;
- you are comfortable using an externally maintained integration branch.

Available entry points
----------------------

- External integration branch:
  `Leev1s/rankseg (paddleseg branch) <https://github.com/Leev1s/rankseg/tree/paddleseg/rankseg/paddleseg>`_
- Docker image:
  `ghcr.io/leev1s/rankseg <https://ghcr.io/leev1s/rankseg>`_

Scope and maintenance
---------------------

This integration is currently treated as an external/community-maintained path.

That means:

- the main RankSEG repository links to it;
- the main RankSEG repository does not yet treat it as a first-party official
  integration path;
- once the integration assets become stable enough, it can be promoted into a
  first-party maintained path later.

Relationship to official integrations
-------------------------------------

If you are new to RankSEG, the recommended first entry point remains:

- :doc:`integrations_pytorch`

That path is maintained directly in this repository and is the current default
official integration route.
