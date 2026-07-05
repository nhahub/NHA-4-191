import io
import os
import numpy as np
import cv2
from locust import HttpUser, task, between, events

# Helpers — generate dummy images in memory

def make_dummy_image_bytes(width=640, height=375) -> bytes:
    """Create a random RGB image encoded as JPEG bytes."""
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", img)
    return buffer.tobytes()

SINGLE_IMAGE = make_dummy_image_bytes()
BATCH_IMAGES = [make_dummy_image_bytes() for _ in range(4)]   # 4-image batch


# User behavior

class DetectionAPIUser(HttpUser):
    """
    Simulates a user sending requests to the detection API.
    wait_time: each user waits 0.1–0.5s between requests.
    """
    wait_time = between(0.1, 0.5)

    # ── Single image (higher weight = called more often) ──
    @task(3)
    def predict_single(self):
        """POST /predict — single image upload."""
        self.client.post(
            "/predict",
            files={"file": ("test.jpg", SINGLE_IMAGE, "image/jpeg")},
            name="/predict [single]",
        )

    #  Batch images 
    @task(1)
    def predict_batch(self):
        """POST /predict/batch — multiple images."""
        files = [
            ("files", (f"img_{i}.jpg", img_bytes, "image/jpeg"))
            for i, img_bytes in enumerate(BATCH_IMAGES)
        ]
        self.client.post(
            "/predict/batch",
            files=files,
            name="/predict/batch [4 images]",
        )

    #  Health check (lightweight baseline) 
    @task(1)
    def health_check(self):
        """GET /health — sanity check endpoint."""
        self.client.get("/health", name="/health")


# Custom stats logging on test finish

@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    """Print p50/p95/p99 summary and pass/fail against targets."""
    stats = environment.runner.stats

    print("\n" + "="*55)
    print("  LOAD TEST RESULTS")
    print("="*55)

    TARGET_P95_MS    = 150
    TARGET_RPS       = 10

    all_pass = True

    for name, entry in stats.entries.items():
        if entry.num_requests == 0:
            continue

        p50 = entry.get_response_time_percentile(0.50)
        p95 = entry.get_response_time_percentile(0.95)
        p99 = entry.get_response_time_percentile(0.99)
        rps = entry.current_rps

        p95_ok  = p95 < TARGET_P95_MS
        rps_ok  = stats.total.current_rps >= TARGET_RPS

        print(f"\n  Endpoint : {name[1]}")
        print(f"  Requests : {entry.num_requests} | Failures: {entry.num_failures}")
        print(f"  p50      : {p50:.1f} ms")
        print(f"  p95      : {p95:.1f} ms  {' if p95_ok else ' > 150ms TARGET'}")
        print(f"  p99      : {p99:.1f} ms")
        print(f"  RPS      : {rps:.1f}  {' if rps_ok else ' < 10 req/s TARGET'}")

        if not p95_ok or not rps_ok:
            all_pass = False

    print("\n" + "-"*55)
    print(f"  Overall: {' ALL TARGETS MET' if all_pass else ' SOME TARGETS FAILED'}")
    print("="*55 + "\n")

    os.makedirs("reports", exist_ok=True)
    print("  HTML Report → reports/load_test_report.html")
