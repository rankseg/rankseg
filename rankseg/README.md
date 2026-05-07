# RankSEG Package Layout

This directory contains the Python package modules behind the public `rankseg` API.

## Main modules

- `_rankseg.py`: defines the `RankSEG` class and the main prediction entry point
- `_rankseg_algo.py`: lower-level solver implementations used by `RankSEG`
- `distribution.py`: probability-distribution utilities used by the algorithms
- `transformers.py`: Hugging Face Transformers compatibility helpers. Use `postprocess` for inference; use the restore helpers only when probability maps are needed directly.

## Import path

```python
from rankseg.transformers import postprocess, restore_semantic_probs, restore_sam_mask_probs
```
