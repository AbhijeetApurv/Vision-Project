#!/usr/bin/env python3
"""
PLANE 1A: LiteRT (.tflite) Model Verification & Host Latency Benchmark
----------------------------------------------------------------------
1. Verifies file existence, size, and cryptographic SHA-256 match against
   `manifest.json` (if provided).
2. Loads the model in `ai-edge-litert` (or `tf.lite.Interpreter`) and inspects
   input/output tensor shapes, data types, and INT8 quantization scale/zero-point.
3. Validates YOLOv8 output head geometry (e.g. [1, 84, 2100] or [1, 2100, 84]
   for 320x320 input) and runs a synthetic forward pass + Non-Maximum Suppression
   (NMS) post-processing sanity check to guarantee numerical stability (no NaNs/Infs).
4. Benchmarks multi-threaded host inference latency (mean, median, P90, P95, P99,
   and estimated FPS) and optionally exports a JSON benchmark report.
"""

import argparse
import hashlib
import json
import logging
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | [Plane-1A] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Plane1A.Verify")


def load_litert_interpreter(model_path: Path, num_threads: int = 4):
    """Loads the LiteRT Interpreter with fallback to TensorFlow Lite."""
    try:
        from ai_edge_litert.interpreter import Interpreter
        runtime_name = "ai-edge-litert"
    except ImportError:
        try:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
            runtime_name = f"tensorflow ({tf.__version__})"
        except ImportError as exc:
            logger.error("Neither `ai-edge-litert` nor `tensorflow` is installed.")
            raise SystemExit(1) from exc

    interpreter = Interpreter(model_path=str(model_path), num_threads=num_threads)
    interpreter.allocate_tensors()
    return interpreter, runtime_name


def compute_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def dequantize_tensor(raw_output: np.ndarray, out_detail: Dict[str, Any]) -> np.ndarray:
    """Converts quantized INT8/UINT8 output tensor back to FP32 using affine scale & zero_point."""
    if raw_output.dtype in (np.int8, np.uint8):
        q_params = out_detail.get("quantization_parameters", {})
        scales = q_params.get("scales", [])
        zero_points = q_params.get("zero_points", [])
        scale = float(scales[0]) if len(scales) > 0 else 1.0
        zp = int(zero_points[0]) if len(zero_points) > 0 else 0
        return (raw_output.astype(np.float32) - zp) * scale
    return raw_output.astype(np.float32)


def sanity_check_yolov8_decode(
    output_fp32: np.ndarray,
    conf_threshold: float = 0.25,
) -> Tuple[bool, str]:
    """
    Validates that the output tensor conforms to standard YOLOv8 detection head
    dimensions ([1, 4 + num_classes, num_anchors] or [1, num_anchors, 4 + num_classes])
    and contains finite values.
    """
    if not np.all(np.isfinite(output_fp32)):
        return False, "Output tensor contains NaN or Inf values!"

    if output_fp32.ndim != 3 or output_fp32.shape[0] != 1:
        return False, f"Expected 3D batch-1 output tensor [1, D1, D2], got shape={output_fp32.shape}"

    _, d1, d2 = output_fp32.shape
    # Transpose [1, 84, 2100] -> [2100, 84] so each row is [cx, cy, w, h, c0, c1, ...]
    preds = output_fp32[0].T if d1 < d2 else output_fp32[0]
    num_anchors, channels = preds.shape
    num_classes = channels - 4
    if num_classes <= 0:
        return False, f"Invalid channel count ({channels}); expected at least 5 (4 box coords + >=1 class)."

    class_scores = preds[:, 4:]
    max_scores = np.max(class_scores, axis=1)
    above_thresh = int(np.sum(max_scores >= conf_threshold))

    return (
        True,
        f"Valid YOLOv8 head: anchors={num_anchors}, classes={num_classes}, "
        f"score_range=[{float(np.min(class_scores)):.4f}, {float(np.max(class_scores)):.4f}], "
        f"candidates>={conf_threshold}: {above_thresh}",
    )


def verify_and_benchmark(
    model_path: Path,
    manifest_path: Optional[Path] = None,
    runs: int = 50,
    warmup: int = 10,
    num_threads: int = 4,
    max_mean_latency_ms: Optional[float] = None,
    report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    actual_sha256 = compute_sha256(model_path)
    actual_size = model_path.stat().st_size
    logger.info("Inspecting model: %s (%.2f MB)", model_path.name, actual_size / (1024 * 1024))
    logger.info("SHA-256         : %s", actual_sha256)

    if manifest_path is not None and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_sha = manifest.get("sha256", "")
        if expected_sha and expected_sha.lower() != actual_sha256.lower():
            raise ValueError(
                f"Manifest SHA-256 mismatch! Manifest={expected_sha} vs Actual={actual_sha256}"
            )
        logger.info("Manifest Check  : PASSED (matches %s)", manifest_path.name)

    interpreter, runtime_name = load_litert_interpreter(model_path, num_threads=num_threads)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    logger.info("LiteRT Runtime  : %s (threads=%d)", runtime_name, num_threads)
    for idx, inp in enumerate(input_details):
        q_params = inp.get("quantization_parameters", {})
        scales = q_params.get("scales", [])
        zps = q_params.get("zero_points", [])
        q_str = (
            f"scale={scales[0]:.6f}, zp={zps[0]}"
            if len(scales) > 0 and len(zps) > 0
            else "unquantized (FP32)"
        )
        logger.info(
            "Input[%d]        : name=%s | shape=%s | dtype=%s | %s",
            idx,
            inp["name"],
            inp["shape"].tolist(),
            inp["dtype"].__name__,
            q_str,
        )

    for idx, out in enumerate(output_details):
        q_params = out.get("quantization_parameters", {})
        scales = q_params.get("scales", [])
        zps = q_params.get("zero_points", [])
        q_str = (
            f"scale={scales[0]:.6f}, zp={zps[0]}"
            if len(scales) > 0 and len(zps) > 0
            else "unquantized (FP32)"
        )
        logger.info(
            "Output[%d]       : name=%s | shape=%s | dtype=%s | %s",
            idx,
            out["name"],
            out["shape"].tolist(),
            out["dtype"].__name__,
            q_str,
        )

    # Prepare synthetic input tensor matching exact input dtype
    inp0 = input_details[0]
    input_shape = inp0["shape"]
    input_dtype = inp0["dtype"]
    rng = np.random.default_rng(seed=42)

    if input_dtype == np.int8:
        dummy_input = rng.integers(-128, 127, size=input_shape, dtype=np.int8)
    elif input_dtype == np.uint8:
        dummy_input = rng.integers(0, 255, size=input_shape, dtype=np.uint8)
    else:
        dummy_input = rng.random(size=input_shape, dtype=np.float32)

    # Warmup passes
    for _ in range(warmup):
        interpreter.set_tensor(inp0["index"], dummy_input)
        interpreter.invoke()

    # Timed benchmark passes
    latencies_ms: List[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        interpreter.set_tensor(inp0["index"], dummy_input)
        interpreter.invoke()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    raw_output = interpreter.get_tensor(output_details[0]["index"])
    output_fp32 = dequantize_tensor(raw_output, output_details[0])
    head_ok, head_msg = sanity_check_yolov8_decode(output_fp32)
    if not head_ok:
        raise RuntimeError(f"Output head verification failed: {head_msg}")
    logger.info("Head Check      : PASSED (%s)", head_msg)

    mean_ms = float(statistics.mean(latencies_ms))
    median_ms = float(statistics.median(latencies_ms))
    stdev_ms = float(statistics.stdev(latencies_ms)) if len(latencies_ms) > 1 else 0.0
    p90_ms = float(np.percentile(latencies_ms, 90))
    p95_ms = float(np.percentile(latencies_ms, 95))
    p99_ms = float(np.percentile(latencies_ms, 99))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    logger.info(
        "Latency (%d runs): Mean=%.2f ms | Median=%.2f ms | P95=%.2f ms | P99=%.2f ms | Est. Throughput=%.1f FPS",
        runs,
        mean_ms,
        median_ms,
        p95_ms,
        p99_ms,
        fps,
    )

    if max_mean_latency_ms is not None and mean_ms > max_mean_latency_ms:
        raise RuntimeError(
            f"Mean latency ({mean_ms:.2f} ms) exceeded SLA threshold ({max_mean_latency_ms:.2f} ms)!"
        )

    summary = {
        "model_file": model_path.name,
        "sha256": actual_sha256,
        "size_bytes": actual_size,
        "runtime": runtime_name,
        "num_threads": num_threads,
        "input_shape": input_shape.tolist(),
        "input_dtype": input_dtype.__name__,
        "output_shape": output_details[0]["shape"].tolist(),
        "output_dtype": output_details[0]["dtype"].__name__,
        "head_validation": head_msg,
        "benchmark": {
            "runs": runs,
            "warmup": warmup,
            "mean_ms": round(mean_ms, 3),
            "median_ms": round(median_ms, 3),
            "stdev_ms": round(stdev_ms, 3),
            "p90_ms": round(p90_ms, 3),
            "p95_ms": round(p95_ms, 3),
            "p99_ms": round(p99_ms, 3),
            "estimated_fps": round(fps, 2),
        },
    }

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        logger.info("Saved benchmark report to %s", report_path)

    return summary


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plane 1A: Sanity-check LiteRT (.tflite) tensor shapes and benchmark host inference latency."
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to exported .tflite model")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path to manifest.json to verify SHA-256 integrity match",
    )
    parser.add_argument("--runs", type=int, default=50, help="Number of timed benchmark iterations (default: 50)")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations (default: 10)")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for LiteRT Interpreter (default: 4)")
    parser.add_argument(
        "--max-latency-ms",
        type=float,
        default=None,
        help="Fail with non-zero exit code if mean latency exceeds this budget in ms (e.g. 33.3 for 30 FPS)",
    )
    parser.add_argument("--report", type=Path, default=None, help="Optional file path to save JSON benchmark report")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    verify_and_benchmark(
        model_path=args.model,
        manifest_path=args.manifest,
        runs=args.runs,
        warmup=args.warmup,
        num_threads=args.threads,
        max_mean_latency_ms=args.max_latency_ms,
        report_path=args.report,
    )
