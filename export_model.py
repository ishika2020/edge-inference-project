import torch
import torchvision.models as models
import os

os.makedirs("model", exist_ok=True)

print("Loading pretrained MobileNetV2...")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

print("Exporting to ONNX...")
torch.onnx.export(
    model,
    dummy_input,
    "model/model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=13,
    dynamo=False,  
)

print("Done. Saved to model/model.onnx")
print(f"ONNX model size: {os.path.getsize('model/model.onnx') / (1024*1024):.2f} MB")
