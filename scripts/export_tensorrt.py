import os
import sys
import time
import argparse
import numpy as np
import torch

BENCH_RUNS  = 200
WARMUP_RUNS = 20
IMG_SIZE    = 640
BATCH_SIZE  = 1


# 1. Check NVIDIA GPU

def check_gpu():
    if not torch.cuda.is_available():
        print(" No NVIDIA GPU detected — TensorRT export requires CUDA.")
        print("   Falling back: skipping TensorRT, raw PyTorch only.")
        return False
    print(f" GPU detected: {torch.cuda.get_device_name(0)}")
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"   Memory: {mem:.1f} GB\n")
    return True


# 2. Export ONNX → TensorRT engine

def export_tensorrt(onnx_path: str, engine_path: str, fp16: bool = True) -> bool:
    """
    Build TensorRT engine from ONNX file.
    Returns True if successful.
    """
    try:
        import tensorrt as trt
    except ImportError:
        print(" TensorRT not installed. Run: pip install tensorrt")
        return False

    print(f"[EXPORT] Building TensorRT engine...")
    print(f"  ONNX   : {onnx_path}")
    print(f"  Engine : {engine_path}")
    print(f"  FP16   : {fp16}\n")

    logger  = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser  = trt.OnnxParser(network, logger)
    config  = builder.create_builder_config()

    # Memory pool
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2 GB

    # FP16 (half precision) — faster on modern GPUs
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  [INFO] FP16 enabled")

    # Parse ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  [ERROR] {parser.get_error(i)}")
            return False

    # Build engine
    print("  [INFO] Building engine (this may take 1-5 minutes)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("  [ERROR] Engine build failed.")
        return False

    # Save engine
    os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)

    size_mb = os.path.getsize(engine_path) / 1e6
    print(f"   Engine saved → {engine_path}  ({size_mb:.1f} MB)\n")
    return True


# 3. Benchmark PyTorch (baseline)


def benchmark_pytorch(weights_path: str) -> float:
    """Returns average FPS for raw PyTorch inference."""
    print("[BENCHMARK] Raw PyTorch...")
    device = torch.device("cuda:0")

    sys.path.insert(0, "yolov5")
    from models.common import DetectMultiBackend  # noqa
    model = DetectMultiBackend(weights_path, device=device, fp16=False)
    model.eval()

    dummy = torch.rand(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE).to(device)

    # Warmup
    for _ in range(WARMUP_RUNS):
        with torch.no_grad():
            model(dummy)
    torch.cuda.synchronize()

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(BENCH_RUNS):
        with torch.no_grad():
            model(dummy)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    fps = (BENCH_RUNS * BATCH_SIZE) / elapsed
    ms  = elapsed * 1000 / BENCH_RUNS
    print(f"  PyTorch  → {fps:.1f} FPS  ({ms:.2f} ms/frame)\n")
    return fps


# 4. Benchmark TensorRT engine

def benchmark_tensorrt(engine_path: str) -> float:
    """Returns average FPS for TensorRT engine inference."""
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa
    except ImportError:
        print(" pycuda / tensorrt not installed.")
        return 0.0

    print("[BENCHMARK] TensorRT engine...")

    logger  = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # Allocate buffers
    input_shape  = (BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
    output_shape = (BATCH_SIZE, 25200, 8)  # YOLOv5s default output

    h_input  = np.random.rand(*input_shape).astype(np.float32)
    h_output = np.empty(output_shape, dtype=np.float32)

    d_input  = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    bindings = [int(d_input), int(d_output)]
    stream   = cuda.Stream()

    # Warmup
    for _ in range(WARMUP_RUNS):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v2(bindings, stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(BENCH_RUNS):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v2(bindings, stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
    elapsed = time.perf_counter() - t0

    fps = (BENCH_RUNS * BATCH_SIZE) / elapsed
    ms  = elapsed * 1000 / BENCH_RUNS
    print(f"  TensorRT → {fps:.1f} FPS  ({ms:.2f} ms/frame)\n")
    return fps


 
# 5. Summary report
 
def print_summary(pytorch_fps: float, trt_fps: float, engine_path: str):
    speedup    = trt_fps / pytorch_fps if pytorch_fps > 0 else 0
    meets_target = speedup >= 2.0

    print("=" * 55)
    print("  TENSORRT EXPORT SUMMARY")
    print("=" * 55)
    print(f"  PyTorch  FPS : {pytorch_fps:.1f}")
    print(f"  TensorRT FPS : {trt_fps:.1f}")
    print(f"  Speedup      : {speedup:.2f}x  {' TARGET MET (≥2x)' if meets_target else ' TARGET NOT MET (<2x)'}")
    print(f"  Engine path  : {engine_path}")
    print("=" * 55)

    # Save simple report
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/TENSORRT_REPORT.md"
    with open(report_path, "w") as f:
        f.write("# TensorRT Export Report\n\n")
        f.write(f"| | FPS |\n|---|---|\n")
        f.write(f"| PyTorch (baseline) | {pytorch_fps:.1f} |\n")
        f.write(f"| TensorRT | {trt_fps:.1f} |\n")
        f.write(f"| **Speedup** | **{speedup:.2f}x** |\n\n")
        f.write(f"Target (≥ 2x): {'✅ MET' if meets_target else '❌ NOT MET'}\n")
    print(f"\n  Report saved → {report_path}")


# 6. Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx",    default="models/yolov5s.onnx",   help="Path to ONNX model")
    parser.add_argument("--weights", default="models/yolov5s.pt",     help="Path to PyTorch weights")
    parser.add_argument("--engine",  default="models/yolov5s.engine", help="Output TensorRT engine path")
    parser.add_argument("--no-fp16", action="store_true",             help="Disable FP16")
    args = parser.parse_args()

    print("\n" + "="*55)
    print("  ONNX → TensorRT Export & Benchmark")
    print("="*55 + "\n")

    # Step 1: Check GPU
    has_gpu = check_gpu()
    if not has_gpu:
        sys.exit(1)

    # Step 2: Export
    success = export_tensorrt(args.onnx, args.engine, fp16=not args.no_fp16)
    if not success:
        sys.exit(1)

    # Step 3: Benchmark both
    pytorch_fps = benchmark_pytorch(args.weights)
    trt_fps     = benchmark_tensorrt(args.engine)

    # Step 4: Summary
    print_summary(pytorch_fps, trt_fps, args.engine)


if __name__ == "__main__":
    main()
