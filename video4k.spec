# PyInstaller spec - dong goi "Video 4K Noel" thanh thu muc chay doc lap (onedir)
# Build:  .venv\Scripts\pyinstaller.exe video4k.spec --noconfirm --distpath release --workpath build_tmp
import shutil
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_data_files, copy_metadata

ROOT = Path(SPECPATH)
ffmpeg = Path(shutil.which("ffmpeg"))
ffprobe = Path(shutil.which("ffprobe"))

datas = [
    (str(ROOT / "app_assets"), "app_assets"),
    (str(ROOT / "onnx"), "onnx"),  # mo hinh cho TensorRT (tools/export_onnx.py)
    *[(str(p), "weights") for p in (ROOT / "weights").glob("*.pth")],
]
binaries = [(str(ffmpeg), "bin"), (str(ffprobe), "bin")]
hiddenimports = ["webview.platforms.edgechromium", "webview.platforms.winforms"]

for pkg in ("gradio", "gradio_client", "safehttpx", "groovy", "webview", "pythonnet", "clr_loader",
            "tensorrt", "tensorrt_bindings", "tensorrt_libs"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# TensorRT: bo DLL cho card may chu (A100 sm80, H100 sm90, B200 sm100) - giu RTX 20/30/40/50 + PTX (~1.3GB nhe hon)
_skip = ("resource_sm80_", "resource_sm90_", "resource_sm100_")
binaries = [b for b in binaries if not any(k in Path(b[0]).name for k in _skip)]
datas = [d for d in datas if not any(k in Path(d[0]).name for k in _skip)]
datas += copy_metadata("gradio", recursive=True)
datas += copy_metadata("gradio_client")

a = Analysis(
    ["app_4k.py"],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "torchvision", "torchaudio", "torch.utils.tensorboard", "IPython", "jupyter", "pytest"],
    # gradio doc ma nguon cua chinh no luc chay -> giu file .py thay vi .pyc
    module_collection_mode={"gradio": "py", "gradio_client": "py"},
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Video4KNoel",
    icon=str(ROOT / "app_assets" / "icon.ico"),
    console=False,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="Video4KNoel")

# Sau khi build: xoa DLL TensorRT cho card may chu (bo loc o tren khong bat duoc vi hook cua tensorrt tu them vao)
import os
for _p in (Path(DISTPATH) / "Video4KNoel").rglob("nvinfer_builder_resource_sm*_11.dll"):
    if any(k in _p.name for k in _skip):
        os.remove(_p)
        print("removed", _p.name)
