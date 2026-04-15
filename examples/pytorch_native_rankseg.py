"""Official PyTorch Native integration example.

This script shows the minimal insertion point for RankSEG in a standard
PyTorch semantic-segmentation inference pipeline.

Typical workflow:
1. Run your segmentation model to obtain logits of shape
   ``(batch_size, num_classes, *image_shape)``.
2. Convert logits to probabilities with softmax for multiclass tasks.
3. Apply RankSEG during inference-time post-processing.

For multilabel tasks, replace softmax with sigmoid and use
`output_mode="multilabel"` when constructing RankSEG.
"""

from __future__ import annotations

import torch

from rankseg import RankSEG


def extract_logits(model_output: object) -> torch.Tensor:
    """Normalize common PyTorch segmentation outputs to a logits tensor."""
    if isinstance(model_output, torch.Tensor):
        return model_output
    if isinstance(model_output, dict):
        if "out" in model_output:
            return model_output["out"]
        if "logits" in model_output:
            return model_output["logits"]
        raise KeyError("Model output dict must contain either 'out' or 'logits'.")
    if hasattr(model_output, "logits"):
        return model_output.logits
    raise TypeError("Unsupported model output type. Expected Tensor, dict, or object with a .logits attribute.")


def predict_with_rankseg(
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    metric: str = "dice",
    solver: str = "RMA",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a PyTorch semantic-segmentation model and post-process with RankSEG.

    Parameters
    ----------
    model
        A PyTorch model whose forward pass returns logits directly, a dict
        containing `out` / `logits`, or an object exposing `.logits`.
    images
        Input images with shape ``(batch_size, channels, *image_shape)``.
    metric
        Segmentation metric optimized by RankSEG. `dice` is the recommended
        default for multiclass semantic segmentation.
    solver
        RankSEG solver. `RMA` is the recommended default for inference.

    Returns
    -------
    preds
        RankSEG predictions with shape ``(batch_size, *image_shape)`` for
        multiclass segmentation.
    logits
        Raw logits returned by the model before softmax, with shape
        ``(batch_size, num_classes, *image_shape)``.
    """
    model.eval()
    with torch.inference_mode():
        model_output = model(images)
        logits = extract_logits(model_output)
        if logits.ndim != 4:
            raise ValueError("Expected logits to have shape (batch_size, num_classes, *image_shape).")

        # Multiclass semantic-segmentation models should be converted with softmax.
        probs = torch.softmax(logits, dim=1)

        rankseg = RankSEG(metric=metric, solver=solver, output_mode="multiclass")
        preds = rankseg.predict(probs)

    return preds, logits


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Replace this toy model with your own pretrained segmentation network.
    # The only contract RankSEG needs is logits with shape
    # (batch_size, num_classes, *image_shape).
    model = torch.nn.Conv2d(in_channels=3, out_channels=21, kernel_size=1).to(device)
    images = torch.randn(2, 3, 256, 256, device=device)

    preds, logits = predict_with_rankseg(model, images)

    print("Input shape: ", tuple(images.shape))
    print("Logits shape:", tuple(logits.shape))
    print("Preds shape: ", tuple(preds.shape))
    print("Preds dtype: ", preds.dtype)
    print()
    print("Integration summary:")
    print("- Keep your existing PyTorch model unchanged.")
    print("- Feed RankSEG probabilities produced by softmax(logits, dim=1).")
    print("- Logits/probabilities are expected to have shape (batch_size, num_classes, *image_shape).")
    print('- Use output_mode="multiclass" for standard semantic segmentation.')
    print("- 2D images use image_shape=(H, W); 3D volumes use image_shape=(H, W, D).")
    print('- For multilabel tasks, use sigmoid probabilities and output_mode="multilabel".')


if __name__ == "__main__":
    main()
