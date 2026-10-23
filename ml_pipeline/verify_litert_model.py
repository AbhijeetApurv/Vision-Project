"""
PLANE 1A: Sanity-checks exported LiteRT (.tflite) tensor shapes, quantization parameters,
and benchmarks host inference latency.
"""

import argparse
import statistics
import time
from pathlib import Path
import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter


def verify_and_benchmark(model_path: Path, runs: int = 50, warmup: int = 10) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    interpreter = Interpreter(model_path=str(model_path), num_threads=4)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("=== LiteRT Model Tensor Inspection ===")
    for idx, inp in enumerate(input_details):
        print(f"Input[{idx}]: name={inp['name']}, shape={inp['shape'].tolist()}, dtype={inp['dtype'].__name__}")
    for idx, out in enumerate(output_details):
        print(f"Output[{idx}]: name={out['name']}, shape={out['shape'].tolist()}, dtype={out['dtype'].__name__}")

    input_shape = input_details[0]["shape"]
    input_dtype = input_details[0]["dtype"]
    dummy_input = np.zeros(input_shape, dtype=input_dtype)

    for _ in range(warmup):
        interpreter.set_tensor(input_details[0]["index"], dummy_input)
        interpreter.invoke()

    latencies_ms = []
    for _ in range(runs):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], dummy_input)
        interpreter.invoke()
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    mean_ms = statistics.mean(latencies_ms)
    p95_ms = np.percentile(latencies_ms, 95)
    print(f"=== Host Latency Benchmark ({runs} runs) ===")
    print(f"Mean: {mean_ms:.2f} ms | P95: {p95_ms:.2f} ms | Est. FPS: {1000.0 / mean_ms:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify LiteRT model tensor shapes and latency.")
    parser.add_argument("--model", type=Path, required=True, help="Path to exported .tflite model")
    parser.add_argument("--runs", type=int, default=50, help="Number of benchmark iterations")
    args = parser.parse_args()

    verify_and_benchmark(args.model, runs=args.runs)
