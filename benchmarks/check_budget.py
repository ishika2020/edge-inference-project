import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--max-size-mb", type=float, required=True)
    parser.add_argument("--max-mem-mb", type=float, required=True)
    parser.add_argument("--max-cold-start-s", type=float, default=10.0)
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    failures = []

    if results["image_size_mb"] > args.max_size_mb:
        failures.append(
            f"Image size {results['image_size_mb']}MB exceeds budget of {args.max_size_mb}MB"
        )

    if results["idle_memory_mb"] > args.max_mem_mb:
        failures.append(
            f"Idle memory {results['idle_memory_mb']}MB exceeds budget of {args.max_mem_mb}MB"
        )

    if results["cold_start_seconds"] > args.max_cold_start_s:
        failures.append(
            f"Cold start {results['cold_start_seconds']}s exceeds budget of {args.max_cold_start_s}s"
        )

    print("Benchmark results:", json.dumps(results, indent=2))

    if failures:
        print("\nBUDGET CHECK FAILED:")
        for f_ in failures:
            print(f"  - {f_}")
        sys.exit(1)

    print("\nAll budgets passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
