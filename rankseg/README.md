# RankSEG Package Layout

This directory contains the Python package modules behind the public `rankseg` API.

## Main modules

- `_rankseg.py`: defines the `rankseg_predict` functional API and the reusable `RankSEG` predictor class
- `_rankseg_algo.py`: lower-level solver implementations used by `rankseg_predict`
- `distribution.py`: probability-distribution utilities used by the algorithms
- `integration/`: compatibility adapters for external model APIs
  - `transformers.py`: standard Hugging Face semantic segmentation helpers. Use `transformers.postprocess` for inference; use `transformers.restore_semantic_probs` when probability maps are needed directly.
  - `sam.py`: SAM-family adapters. Use `sam.Sam1`, `sam.Sam2`, or `sam.Sam3` for family-specific mask restoration and postprocessing.

## Import path

```python
from rankseg.integration import transformers
from rankseg.integration import sam

preds = transformers.postprocess(outputs, target_sizes=target_sizes)
adapter = sam.Sam2(rankseg_kwargs={"metric": "dice"})
```

Explicit submodule imports are also supported:

```python
from rankseg.integration.transformers import postprocess, restore_semantic_probs
from rankseg.integration.sam import Sam1, Sam2, Sam3
```
