"""Official Transformers integration example.

This script shows the minimal insertion point for RankSEG in a standard
Hugging Face semantic-segmentation inference pipeline.

Typical workflow:
1. Load a segmentation processor and model from `transformers`.
2. Run the usual `processor -> model -> outputs` inference path.
3. Replace manual post-processing with `rankseg.transformers.postprocess`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import requests
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor

from rankseg.transformers import postprocess


def predict_with_rankseg(
    outputs: object,
    image_size: tuple[int, int],
    *,
    metric: str = "dice",
):
    """Post-process Hugging Face segmentation outputs with RankSEG."""
    return postprocess(
        outputs,
        target_sizes=image_size[::-1],
        rankseg_kwargs={"metric": metric},
    )


def main() -> None:
    processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

    url = (
        "https://plus.unsplash.com/premium_photo-1673210886161-bfcc40f54d1f"
        "?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MXx8cGVyc29uJTIwc3RhbmRpbmd8ZW58MHx8MHx8&w=1000&q=80"
    )
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(images=image, return_tensors="pt")

    outputs = model(**inputs)
    logits = outputs.logits.cpu()
    upsampled_logits = F.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    baseline_pred = upsampled_logits.argmax(dim=1)[0]
    rankseg_pred = predict_with_rankseg(outputs, image.size)[0].cpu()

    print("Image size:    ", image.size[::-1])
    print("Logits shape:  ", tuple(outputs.logits.shape))
    print("Baseline shape:", tuple(baseline_pred.shape))
    print("RankSEG shape: ", tuple(rankseg_pred.shape))
    print()
    print("Integration summary:")
    print("- Keep the standard Hugging Face processor and model unchanged.")
    print("- Run the usual processor -> model -> outputs inference path.")
    print("- Replace manual argmax post-processing with rankseg.transformers.postprocess(...).")
    print("- Pass target_sizes as the original image size in (H, W) order.")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Input image")
    axes[1].imshow(baseline_pred)
    axes[1].set_title("Baseline argmax")
    axes[2].imshow(rankseg_pred)
    axes[2].set_title("RankSEG postprocess")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
