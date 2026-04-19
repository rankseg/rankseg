# RankSEG Package Layout

This directory contains the Python package modules behind the public `rankseg` API.

## Main modules

- `_rankseg.py`: defines the `RankSEG` class and the main prediction entry point
- `_rankseg_algo.py`: lower-level solver implementations used by `RankSEG`
- `distribution.py`: probability-distribution utilities used by the algorithms
- `transformers.py`: Hugging Face Transformers compatibility helper for restoring semantic probabilities and post-processing with `RankSEG`

## Import path

```python
from rankseg.transformers import postprocess, restore_semantic_probs
```
