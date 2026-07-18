import argparse
import json
import subprocess
import time
import sys


def get_image_size_mb(image: str) -> float:
    out = subprocess.run(
        ["docker", "image", "inspect", image, "--format={{.Size}}"],
        capture_output=True, text=True, check=True
    )
    size_bytes = int(out.stdout.strip())
    return round(size_bytes / (1024 * 1024), 2)


def run_container(image: str, container_name: str, port: int) -> None:
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
    subprocess.run(
        ["docker", "run", "-d", "--name", container_name, "-p", f"{port}:8000", image],
        check=True, capture_output=True,
    )


def wait_for_health(port: int, timeout: int = 30) -> float:
    """Returns time-to-healthy in seconds (cold start proxy)."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            result = subprocess.run(
                ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", f"http://127.0.0.1:{port}/health"],
                capture_output=True, text=True, timeout=2,
            )
            if result.stdout.strip() == "200":
                return round(time.time() - start, 2)
        except Exception:
            pass
        time.sleep(0.2)
    raise TimeoutError(f"Container on port {port} never became healthy within {timeout}s")


def get_memory_usage_mb(container_name: str) -> float:
    out = subprocess.run(
        ["docker", "stats", container_name, "--no-stream", "--format", "{{.MemUsage}}"],
        capture_output=True, text=True, check=True,
    )
    # Format looks like "45.3MiB / 1.943GiB"
    mem_str = out.stdout.strip().split("/")[0].strip()
    if "MiB" in mem_str:
        return round(float(mem_str.replace("MiB", "")), 2)
    if "GiB" in mem_str:
        return round(float(mem_str.replace("GiB", "")) * 1024, 2)
    return -1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="benchmarks/results.json")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()

    container_name = "edge-inference-bench"
    results = {"image": args.image}

    print(f"Measuring image size for {args.image}...")
    results["image_size_mb"] = get_image_size_mb(args.image)

    print(f"Starting container from {args.image}...")
    run_container(args.image, container_name, args.port)

    print("Waiting for health check (cold start timing)...")
    results["cold_start_seconds"] = wait_for_health(args.port)

    print("Measuring idle memory usage...")
    time.sleep(1)
    results["idle_memory_mb"] = get_memory_usage_mb(container_name)

    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
