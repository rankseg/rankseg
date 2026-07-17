"""Run the fixed DiNTS model once and cache softmax probabilities per case."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from common import (
    DEFAULT_CONFIG,
    atomic_torch_save,
    cache_dtype,
    case_cache_path,
    config_sha256,
    load_config,
    resolve_device,
    root_path,
    seed_everything,
    sha256,
    software_info,
    validate_probability_payload,
)
from huggingface_hub import hf_hub_download
from monai.data import decollate_batch
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import DiNTS, TopologyInstance
from monai.transforms import (
    Activations,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--device", help="Override config device, for example cuda:0 or cpu")
    parser.add_argument("--case-limit", type=int, help="Only cache the first N configured cases")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def preprocessing() -> Compose:
    return Compose(
        [
            LoadImaged(keys=("image", "label")),
            EnsureChannelFirstd(keys=("image", "label")),
            Orientationd(keys=("image", "label"), axcodes="RAS"),
            Spacingd(keys=("image", "label"), pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            ScaleIntensityRanged(keys="image", a_min=-87, a_max=199, b_min=0, b_max=1, clip=True),
            EnsureTyped(keys=("image", "label")),
        ]
    )


def verify_checkpoints(config: dict) -> tuple[Path, Path]:
    model_config = config["model"]
    bundle_dir = root_path(model_config["bundle_dir"])
    checkpoint = bundle_dir / model_config["checkpoint"]
    architecture = bundle_dir / model_config["architecture_checkpoint"]
    expected = {
        checkpoint: model_config["checkpoint_sha256"],
        architecture: model_config["architecture_checkpoint_sha256"],
    }
    for path, expected_hash in expected.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing bundle file: {path}")
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"SHA-256 mismatch for {path}: expected {expected_hash}, got {actual_hash}")
    return checkpoint, architecture


def build_network(config: dict, device: torch.device) -> DiNTS:
    checkpoint_path, architecture_path = verify_checkpoints(config)
    architecture = torch.load(architecture_path, map_location=device, weights_only=False)
    topology = TopologyInstance(
        channel_mul=1,
        num_blocks=12,
        num_depths=4,
        use_downsample=True,
        arch_code=[architecture["arch_code_a"], architecture["arch_code_c"]],
        device=str(device),
    )
    network = DiNTS(
        dints_space=topology,
        in_channels=1,
        num_classes=3,
        use_downsample=True,
        node_a=torch.as_tensor(architecture["node_a"], device=device),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    network.load_state_dict(checkpoint.get("model", checkpoint))
    network.eval()
    return network


def ensure_case_files(config: dict, case_id: str, allow_download: bool) -> tuple[Path, Path]:
    dataset = config["dataset"]
    data_dir = root_path(dataset["data_dir"])
    image = data_dir / "imagesTr" / f"{case_id}.nii.gz"
    label = data_dir / "labelsTr" / f"{case_id}.nii.gz"
    for subdir, path in (("imagesTr", image), ("labelsTr", label)):
        if path.exists():
            continue
        if not allow_download:
            raise FileNotFoundError(f"Missing {path}; rerun without --no-download")
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {subdir}/{case_id}.nii.gz", flush=True)
        hf_hub_download(
            repo_id=dataset["mirror_repo"],
            repo_type="dataset",
            filename=f"{subdir}/{case_id}.nii.gz",
            local_dir=data_dir,
        )
    return image, label


@torch.inference_mode()
def infer_probabilities(
    network: DiNTS, image: torch.Tensor, config: dict, device: torch.device
) -> tuple[torch.Tensor, float]:
    model_config = config["model"]
    inferer = SlidingWindowInferer(
        roi_size=tuple(model_config["roi_size"]),
        sw_batch_size=int(model_config["sw_batch_size"]),
        overlap=float(model_config["overlap"]),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    logits = inferer(image.unsqueeze(0).to(device), network)
    samples = decollate_batch(logits)
    probabilities = Activations(softmax=True)(samples[0])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return probabilities.detach().cpu(), elapsed_ms


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["experiment"]["seed"]))
    device = resolve_device(args.device, config["decoding"]["device"])
    case_ids = list(config["dataset"]["case_ids"])
    if args.case_limit is not None:
        case_ids = case_ids[: args.case_limit]

    cases: list[tuple[str, Path, Path]] = []
    for case_id in case_ids:
        image, label = ensure_case_files(config, case_id, not args.no_download)
        cases.append((case_id, image, label))

    transform = preprocessing()
    network: DiNTS | None = None
    probability_type = cache_dtype(config["outputs"]["probability_dtype"])
    image_type = cache_dtype(config["outputs"]["image_dtype"])
    for index, (case_id, image_path, label_path) in enumerate(cases, start=1):
        destination = case_cache_path(config, case_id)
        if destination.exists() and not args.overwrite:
            print(f"[{index}/{len(cases)}] Using existing {destination}", flush=True)
            continue
        if network is None:
            network = build_network(config, device)
        print(f"[{index}/{len(cases)}] Inferring {case_id}", flush=True)
        data = transform({"image": str(image_path), "label": str(label_path)})
        image = data["image"].as_tensor()
        label = data["label"].as_tensor().squeeze(0).to(torch.uint8)
        probabilities, inference_ms = infer_probabilities(network, image, config, device)
        payload = {
            "schema_version": 1,
            "case_id": case_id,
            "probabilities": probabilities.to(probability_type).contiguous(),
            "image": image.squeeze(0).to(image_type).cpu().contiguous(),
            "label": label.cpu().contiguous(),
            "metadata": {
                "config_sha256": config_sha256(config),
                "source_image": str(image_path.relative_to(root_path(config["dataset"]["data_dir"]))),
                "source_label": str(label_path.relative_to(root_path(config["dataset"]["data_dir"]))),
                "spatial_shape": list(label.shape),
                "probability_dtype": str(probability_type),
                "inference_ms": inference_ms,
                "no_retraining_or_finetuning": True,
                "software": software_info(),
            },
        }
        validate_probability_payload(payload, num_classes=len(config["dataset"]["class_names"]))
        atomic_torch_save(payload, destination)
        size_gib = destination.stat().st_size / 1024**3
        print(f"Saved {destination} ({size_gib:.2f} GiB, inference {inference_ms:.1f} ms)", flush=True)


if __name__ == "__main__":
    main()
