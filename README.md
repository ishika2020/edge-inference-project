# Edge-Optimized ML Inference Container

A container footprint optimization project for deploying ML inference to resource-constrained
edge environments. Builds on production experience deploying CV models to low-connectivity,
resource-limited sites — this project formalizes that into a repeatable, CI/CD-enforced pattern.

Two versions of the same inference service (image classification via MobileNetV2) are built and
benchmarked:
- **Baseline**: naive Dockerfile, full PyTorch/torchvision stack
- **Optimized**: multi-stage build, slim base image, ONNX Runtime only (no training framework shipped)

A GitHub Actions pipeline builds the optimized image on every push and **fails the build** if it
exceeds a defined size/memory/cold-start budget.

## Setup (run locally first, before pushing)

```bash
# 1. Install export dependencies and export the model to ONNX
pip install torch torchvision onnx --break-system-packages
python export_model.py
# -> creates model/model.onnx

# 2. Build both images
docker build -f Dockerfile.baseline -t edge-inference:baseline .
docker build -f Dockerfile.optimized -t edge-inference:optimized .

# 3. Compare sizes directly
docker images | grep edge-inference

# 4. Run benchmarks against the optimized image
pip install requests --break-system-packages
python benchmarks/measure.py --image edge-inference:optimized --output results.json

# 5. Check it against budget (same check CI runs)
python benchmarks/check_budget.py --results results.json --max-size-mb 300 --max-mem-mb 150 --max-cold-start-s 10
```

## Results

| Metric | Baseline | Optimized | Improvement |
|---|---|---|---|
| Image size | 705.69  MB | 122.92 MB | ~82.6 % |
| Idle memory | 79.35 MB | 81.3 MB | no change % |
| Cold start | 1.51 s | 2.19 s | slightly slower % |

> Fill this table in with your real numbers after running the setup steps above.

## Why this matters for edge/IoT deployment

Edge and IoT deployment environments are constrained on bandwidth (pushing updates to devices),
storage, and RAM — unlike typical cloud deployment where over-provisioning is cheap and easy.
A smaller, leaner container means faster updates over constrained networks, lower resource
pressure on the device, and lower cost at fleet scale. This project demonstrates that
optimization as a measurable, CI-enforced engineering practice rather than a one-off manual step.

## Architecture

```
export_model.py          - exports pretrained MobileNetV2 to ONNX
app/main.py               - FastAPI inference service (ONNX Runtime)
Dockerfile.baseline        - naive full-framework container
Dockerfile.optimized       - multi-stage, slim, ONNX-only container
benchmarks/measure.py      - measures image size, memory, cold start
benchmarks/check_budget.py - fails CI if budgets are exceeded
.github/workflows/ci.yml   - builds + benchmarks + enforces budget on every push
```

## Tech Stack
Python, PyTorch (export-time only), ONNX Runtime, FastAPI, Docker (multi-stage builds),
GitHub Actions.
