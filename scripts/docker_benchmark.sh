#!/bin/bash
# Docker container resource benchmark
# Measures image size, startup time, RAM/CPU usage during inference
set -e

IMAGE="ghcr.io/abdallah4z/road-sense:latest"
CONTAINER_NAME="road-sense-benchmark"
RESULTS_FILE="experiments/docker_benchmark_results.json"

echo "============================================"
echo "DOCKER CONTAINER RESOURCE BENCHMARK"
echo "============================================"

# 1. Image size
echo ""
echo "[1/4] Checking image size..."
SIZE=$(docker images --format "{{.Size}}" "$IMAGE" 2>/dev/null || echo "not found")
if [ "$SIZE" = "not found" ]; then
    echo "  Image not found locally. Pulling..."
    docker pull "$IMAGE" > /dev/null 2>&1
    SIZE=$(docker images --format "{{.Size}}" "$IMAGE")
fi
echo "  Image size: $SIZE"

# 2. Startup time
echo ""
echo "[2/4] Measuring startup time..."
START=$(date +%s%N)
docker run --name "$CONTAINER_NAME" -d --rm -p 8001:8000 "$IMAGE" > /dev/null 2>&1

# Wait for health endpoint
for i in $(seq 1 30); do
    sleep 1
    if curl -sf http://localhost:8001/health > /dev/null 2>&1; then
        break
    fi
done
END=$(date +%s%N)
STARTUP_MS=$(( (END - START) / 1000000 ))
echo "  Startup time: ${STARTUP_MS}ms"

# 3. RAM and CPU during inference
echo ""
echo "[3/4] Measuring RAM/CPU during inference..."
docker stats "$CONTAINER_NAME" --no-stream --format "{{.MemUsage}}|{{.CPUPerc}}" > /tmp/docker_stats.txt
STATS=$(cat /tmp/docker_stats.txt)
MEM=$(echo "$STATS" | cut -d'|' -f1)
CPU=$(echo "$STATS" | cut -d'|' -f2)

# Run inference test
echo "  Running inference test..."
python3 -c "
import requests, time
img = b'0' * (640 * 640 * 3)
times = []
for _ in range(20):
    t0 = time.time()
    r = requests.post('http://localhost:8001/detect', files={'file': ('test.jpg', img)}, timeout=30)
    times.append((time.time() - t0) * 1000)
print(f'  Avg latency: {sum(times)/len(times):.1f}ms')
"

docker stats "$CONTAINER_NAME" --no-stream --format "{{.MemUsage}}|{{.CPUPerc}}" > /tmp/docker_stats_load.txt
STATS_LOAD=$(cat /tmp/docker_stats_load.txt)
MEM_LOAD=$(echo "$STATS_LOAD" | cut -d'|' -f1)
CPU_LOAD=$(echo "$STATS_LOAD" | cut -d'|' -f2)

# 4. Cleanup
echo ""
echo "[4/4] Cleaning up..."
docker stop "$CONTAINER_NAME" > /dev/null 2>&1 || true

# Results
echo ""
echo "============================================"
echo "BENCHMARK RESULTS"
echo "============================================"
echo "  Image size:       $SIZE"
echo "  Startup time:     ${STARTUP_MS}ms"
echo "  RAM (idle):       $MEM"
echo "  CPU (idle):       $CPU"
echo "  RAM (load):       $MEM_LOAD"
echo "  CPU (load):       $CPU_LOAD"
echo "============================================"

# Save JSON
mkdir -p "$(dirname "$RESULTS_FILE")"
cat > "$RESULTS_FILE" <<EOF
{
  "image_size": "$SIZE",
  "startup_ms": $STARTUP_MS,
  "ram_idle": "$MEM",
  "cpu_idle": "$CPU",
  "ram_load": "$MEM_LOAD",
  "cpu_load": "$CPU_LOAD"
}
EOF
echo "Results saved: $RESULTS_FILE"
