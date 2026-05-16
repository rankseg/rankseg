# RankSEG Package Layout

This directory contains the Python package modules behind the public `rankseg` API.

## Main modules

- `_rankseg.py`: defines the `RankSEG` class and the main prediction entry point
- `_rankseg_algo.py`: lower-level solver implementations used by `RankSEG`
- `distribution.py`: probability-distribution utilities used by the algorithms
- `integration/`: compatibility adapters for external model APIs
  - `transformers.py`: standard Hugging Face semantic segmentation helpers. Use `postprocess` for inference; use `restore_semantic_probs` when probability maps are needed directly.
  - `sam.py`: SAM-family adapters. Use `Sam1`, `Sam2`, or `Sam3` for family-specific mask restoration and postprocessing.

## Import path

```python
from rankseg.integration.transformers import postprocess, restore_semantic_probs
from rankseg.integration.sam import Sam1, Sam2, Sam3
```
