"""Xuat cac mo hinh SRVGG (Da dung, Anime) sang ONNX fp16 voi kich thuoc dong - dung cho TensorRT.
Chay 1 lan khi build:  .venv\\Scripts\\python.exe tools\\export_onnx.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import app_4k as A  # noqa: E402

out_dir = ROOT / "onnx"
out_dir.mkdir(exist_ok=True)
for name, (fname, build, scale) in A.MODELS.items():
    net = build()
    if not isinstance(net, A.SRVGGNetCompact):
        continue
    sd = torch.load(A.WEIGHTS / fname, map_location="cpu", weights_only=True)
    net.load_state_dict(sd.get("params_ema", sd.get("params", sd)))
    net = net.eval().cuda().half()
    x = torch.rand(1, 3, 270, 480, device="cuda").half()
    path = out_dir / (Path(fname).stem + ".onnx")
    with torch.no_grad():
        torch.onnx.export(net, (x,), str(path), input_names=["in"], output_names=["out"], opset_version=17,
                          dynamic_axes={"in": {2: "h", 3: "w"}, "out": {2: "H", 3: "W"}}, dynamo=False)
    print("ok", path, round(path.stat().st_size / 1e6, 1), "MB")
