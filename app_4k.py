"""
Real-ESRGAN Video Upscaler - app giao dien web chay tren may.
Dua video vao -> upscale bang Real-ESRGAN tren GPU -> xuat video 4K (giu nguyen am thanh).

Khong phu thuoc basicsr (thu vien da ngung phat trien, khong cai duoc tren Python 3.13):
kien truc mo hinh duoc dinh nghia lai ben duoi, tuong thich 100% voi file trong so chinh thuc.
"""
import base64
from contextlib import nullcontext as _nullctx
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from fractions import Fraction
from pathlib import Path

FROZEN = getattr(sys, "frozen", False)
# Thu muc tai nguyen: canh file .py khi chay tu source, hoac thu muc _internal khi da dong goi
RES = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
WEIGHTS = RES / "weights"
ASSETS = RES / "app_assets"
OUT_DIR = Path.home() / "Videos" / "Video 4K"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Chi cho chay 1 ban app cung luc (ban dong goi): mo lan 2 -> bao va thoat
if FROZEN and os.name == "nt":
    import ctypes

    _mutex = ctypes.windll.kernel32.CreateMutexW(None, False, "Local\\Video4KNoel_SingleInstance")
    if ctypes.windll.kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
        ctypes.windll.user32.MessageBoxW(None, "Video 4K Noel đang chạy rồi.\nHãy xem cửa sổ app trên thanh Taskbar.",
                                         "Video 4K Noel", 0x40)
        sys.exit(0)


def _open_log():
    for name in ("app_log.txt", f"app_log_{os.getpid()}.txt"):
        try:
            return open(OUT_DIR / name, "w", encoding="utf-8", buffering=1)
        except OSError:
            continue
    return open(os.devnull, "w")


# Ban dong goi khong co cua so console -> stdout/stderr la None, ghi tam vao file log
if sys.stdout is None or sys.stderr is None:
    _log = _open_log()
    sys.stdout = sys.stdout or _log
    sys.stderr = sys.stderr or _log
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import cv2  # noqa: E402
import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def _tool(name):
    p = RES / "bin" / f"{name}.exe"
    return str(p) if p.exists() else name


FFMPEG, FFPROBE = _tool("ffmpeg"), _tool("ffprobe")
NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # khong bat cua so den khi goi ffmpeg
RUNNING = set()  # cac tien trinh ffmpeg dang chay, de dong khi thoat app
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HALF = DEVICE.type == "cuda"
torch.backends.cudnn.benchmark = True


# ----------------------------------------------------------------------------- kien truc mo hinh
class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.scale = scale
        in_ch = 3 * (4 if scale == 2 else 16 if scale == 1 else 1)
        self.conv_first = nn.Conv2d(in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            x = F.pixel_unshuffle(x, 2)
        elif self.scale == 1:
            x = F.pixel_unshuffle(x, 4)
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_feat=64, num_conv=16, upscale=4):
        super().__init__()
        self.upscale = upscale
        body = [nn.Conv2d(3, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        for _ in range(num_conv):
            body += [nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        body.append(nn.Conv2d(num_feat, 3 * upscale * upscale, 3, 1, 1))
        self.body = nn.ModuleList(body)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for m in self.body:
            out = m(out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


MODELS = {
    "⚡ Đa dụng – nhanh (khuyên dùng)": ("realesr-general-x4v3.pth", lambda: SRVGGNetCompact(num_conv=32), 4),
    "🎨 Anime / hoạt hình – rất nhanh": ("realesr-animevideov3.pth", lambda: SRVGGNetCompact(num_conv=16), 4),
    "💎 Chất lượng cao x4 – chậm": ("RealESRGAN_x4plus.pth", lambda: RRDBNet(scale=4), 4),
    "💎 Chất lượng cao x2 – chậm (hợp video 1080p)": ("RealESRGAN_x2plus.pth", lambda: RRDBNet(scale=2), 2),
}
_model_cache = {}


def get_model(name):
    if name not in _model_cache:
        fname, build, scale = MODELS[name]
        sd = torch.load(WEIGHTS / fname, map_location="cpu", weights_only=True)
        sd = sd.get("params_ema", sd.get("params", sd))
        net = build()
        net.load_state_dict(sd, strict=True)
        net.eval().to(DEVICE)
        if HALF:
            net.half()
        if DEVICE.type == "cuda":  # bo cuc bo nho NHWC: nhanh hon ~30% tren GPU RTX, ket qua giong het
            net.to(memory_format=torch.channels_last)
        _model_cache.clear()  # chi giu 1 mo hinh trong VRAM
        _trt_cache.clear()
        torch.cuda.empty_cache()
        _model_cache[name] = (net, scale)
    return _model_cache[name]


@torch.inference_mode()
def run_model(net, scale, x, tile):
    """x: 1x3xHxW (0..1). Tra ve 1x3x(H*s)x(W*s). tile>0 -> chia o de tiet kiem VRAM."""
    _, _, h, w = x.shape
    ph, pw = (h % 2, w % 2) if scale == 2 else (0, 0)
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    if DEVICE.type == "cuda":
        x = x.contiguous(memory_format=torch.channels_last)
    if tile <= 0:
        out = net(x)
    else:
        pad = 10
        H, W = x.shape[2:]
        out = x.new_zeros((1, 3, H * scale, W * scale))
        for y0 in range(0, H, tile):
            for x0 in range(0, W, tile):
                y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
                ya, xa = max(y0 - pad, 0), max(x0 - pad, 0)
                yb, xb = min(y1 + pad, H), min(x1 + pad, W)
                o = net(x[:, :, ya:yb, xa:xb])
                out[:, :, y0 * scale:y1 * scale, x0 * scale:x1 * scale] = o[
                    :, :, (y0 - ya) * scale:(y1 - ya) * scale, (x0 - xa) * scale:(x1 - xa) * scale]
    return out[:, :, :h * scale, :w * scale]


# ----------------------------------------------------------------------------- TensorRT (nhanh ~2.3 lan)
# Chi dung cho mo hinh SRVGG (Da dung, Anime): vung anh mo hinh "nhin" quanh moi diem chi ~34px, nen cat
# khung hinh thanh cac dai ngang co chong lan 40px roi ghep lai cho ket qua GIONG HET chay nguyen khung.
ONNX_DIR = RES / "onnx"
TRT_DIR = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Video4KNoel" / "trt_cache"
TRT_PAD = 40
TRT_MAX_PX = 2_100_000  # dien tich toi da moi lan chay (~1080p) - lon hon se thieu VRAM tren card 12GB
_trt_cache = {}


def trt_available(model_name):
    fname = MODELS[model_name][0]
    if DEVICE.type != "cuda" or not (ONNX_DIR / (Path(fname).stem + ".onnx")).exists():
        return False
    try:
        import tensorrt  # noqa: F401
        return True
    except Exception:
        return False


class TRTRunner:
    def __init__(self, model_name, h, w):
        import tensorrt as trt

        fname, _, self.scale = MODELS[model_name]
        n = max(1, -(-h * w // TRT_MAX_PX))  # so dai ngang
        self.h, self.w = h, w
        self.strip = -(-h // n)
        self.win = h if n == 1 else min(h, self.strip + 2 * TRT_PAD)
        gpu = "".join(c for c in torch.cuda.get_device_name(0) if c.isalnum())
        self.engine_path = TRT_DIR / f"{Path(fname).stem}_{self.win}x{w}_{gpu}_trt{trt.__version__}.engine"
        self.trt = trt
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.onnx_path = ONNX_DIR / (Path(fname).stem + ".onnx")

    @property
    def cached(self):
        return self.engine_path.exists()

    def load(self):
        trt = self.trt
        if not self.cached:
            TRT_DIR.mkdir(parents=True, exist_ok=True)
            torch.cuda.empty_cache()
            b = trt.Builder(self.logger)
            flag = getattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED", None)
            nw = b.create_network(1 << int(flag) if flag is not None else 0)
            parser = trt.OnnxParser(nw, self.logger)
            if not parser.parse_from_file(str(self.onnx_path)):
                raise RuntimeError(f"ONNX parse loi: {parser.get_error(0)}")
            cfg = b.create_builder_config()
            cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
            if flag is None and hasattr(trt.BuilderFlag, "FP16"):
                cfg.set_flag(trt.BuilderFlag.FP16)
            prof = b.create_optimization_profile()
            shape = (1, 3, self.win, self.w)
            prof.set_shape("in", shape, shape, shape)
            cfg.add_optimization_profile(prof)
            ser = b.build_serialized_network(nw, cfg)
            if ser is None:
                raise RuntimeError("TensorRT build that bai")
            tmp = self.engine_path.with_suffix(".tmp")
            tmp.write_bytes(bytes(ser))
            tmp.replace(self.engine_path)
        self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(self.engine_path.read_bytes())
        if self.engine is None:
            self.engine_path.unlink(missing_ok=True)
            raise RuntimeError("Khong nap duoc engine TensorRT")
        self.ctx = self.engine.create_execution_context()
        self.ctx.set_input_shape("in", (1, 3, self.win, self.w))
        s = self.scale
        self.buf_in = torch.empty((1, 3, self.win, self.w), device="cuda", dtype=torch.half)
        self.buf_out = torch.empty((1, 3, self.win * s, self.w * s), device="cuda", dtype=torch.half)
        self.ctx.set_tensor_address("in", self.buf_in.data_ptr())
        self.ctx.set_tensor_address("out", self.buf_out.data_ptr())
        return self

    def __call__(self, x):
        """x: 1x3xhxw half CUDA -> 1x3x(h*s)x(w*s) half. Chay tren stream CUDA hien tai."""
        s, h = self.scale, self.h
        out = torch.empty((1, 3, h * s, self.w * s), device="cuda", dtype=torch.half)
        stream = torch.cuda.current_stream().cuda_stream
        for y0 in range(0, h, self.strip):
            y1 = min(y0 + self.strip, h)
            a = min(max(y0 - TRT_PAD, 0), h - self.win)  # cua so luon nam tron trong anh
            self.buf_in.copy_(x[:, :, a:a + self.win])
            self.ctx.execute_async_v3(stream)
            out[:, :, y0 * s:y1 * s] = self.buf_out[:, :, (y0 - a) * s:(y1 - a) * s]
        return out


def get_trt(model_name, h, w):
    key = (model_name, h, w)
    if key not in _trt_cache:
        _trt_cache.clear()
        _trt_cache[key] = TRTRunner(model_name, h, w).load()
    return _trt_cache[key]


@torch.inference_mode()
def upscale_frame(net, scale, frame_rgb, tw, th, tile_box, pre=None, yuv=False, runner=None):
    """frame_rgb: HxWx3 uint8 -> thxtwx3 uint8 RGB, hoac bytes YUV420p BT.709 (yuv=True, doi mau ngay tren GPU)."""
    if not frame_rgb.flags.writeable:
        frame_rgb = frame_rgb.copy()
    x = torch.from_numpy(frame_rgb).to(DEVICE, non_blocking=True).permute(2, 0, 1).unsqueeze(0)
    x = (x.half() if HALF else x.float()) / 255.0
    if pre:  # che do tang toc: thu nho dau vao de mo hinh phong to vua dung kich thuoc dich
        x = F.interpolate(x.float(), size=(pre[1], pre[0]), mode="area").to(x.dtype)
    while True:
        try:
            y = runner(x.contiguous()) if runner is not None else run_model(net, scale, x, tile_box[0])
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if tile_box[0] == 0:
                tile_box[0] = 512
            elif tile_box[0] > 128:
                tile_box[0] //= 2
            else:
                raise
    if y.shape[2] != th or y.shape[3] != tw:
        y = F.interpolate(y.float(), size=(th, tw), mode="bicubic", antialias=True, align_corners=False)
    if yuv:
        return rgb_to_yuv420(y.float().clamp_(0, 1))
    y = (y.clamp_(0, 1) * 255.0).round_().to(torch.uint8)
    return y[0].permute(1, 2, 0).contiguous().cpu().numpy()


def rgb_to_yuv420(rgb):
    """1x3xHxW (0..1, GPU) -> bytes YUV420p, BT.709 limited range (giong ffmpeg out_color_matrix=bt709:out_range=tv)."""
    r, g, b = rgb[0, 0], rgb[0, 1], rgb[0, 2]
    y = 16.0 + 219.0 * (0.2126 * r + 0.7152 * g + 0.0722 * b)
    cb = 128.0 + 224.0 * (-0.114572 * r - 0.385428 * g + 0.5 * b)
    cr = 128.0 + 224.0 * (0.5 * r - 0.454153 * g - 0.045847 * b)
    c = F.avg_pool2d(torch.stack((cb, cr)).unsqueeze(0), 2)[0]  # lay mau mau 4:2:0
    out = torch.cat((y.flatten(), c[0].flatten(), c[1].flatten())).round_().clamp_(0, 255).to(torch.uint8)
    return out.cpu().numpy().tobytes()


# ----------------------------------------------------------------------------- video I/O
RES_PRESETS = {"4K": (3840, 2160), "2K": (2560, 1440), "Full HD": (1920, 1080)}
CODECS = {
    "H.265 / HEVC (file nhỏ, chất lượng tốt) ★": ["-c:v", "hevc_nvenc", "-tag:v", "hvc1"],
    "H.264 (tương thích mọi nơi)": ["-c:v", "h264_nvenc"],
    "AV1 (nhỏ nhất, cần máy/phần mềm mới)": ["-c:v", "av1_nvenc"],
}
COPY_AUDIO = {"aac", "mp3", "ac3", "eac3", "alac", "opus"}


def probe(src):
    r = subprocess.run([FFPROBE, "-v", "error", "-show_streams", "-show_format", "-of", "json", src],
                       capture_output=True, text=True, encoding="utf-8", creationflags=NOWIN)
    if r.returncode != 0:
        raise gr.Error(f"Không đọc được video: {r.stderr.strip()[:300]}")
    info = json.loads(r.stdout)
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if v is None:
        raise gr.Error("File không có luồng video.")
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    w, h = int(v["width"]), int(v["height"])
    rot = 0
    for sd in v.get("side_data_list", []):
        if "rotation" in sd:
            rot = int(sd["rotation"])
    rot = int(v.get("tags", {}).get("rotate", rot))
    if abs(rot) % 180 == 90:
        w, h = h, w
    fps = Fraction(v.get("avg_frame_rate") or "0/1")
    if fps == 0:
        fps = Fraction(v.get("r_frame_rate", "30/1"))
    dur = float(v.get("duration") or info["format"].get("duration") or 0)
    nb = int(v["nb_frames"]) if str(v.get("nb_frames", "")).isdigit() else int(dur * fps)
    cs = v.get("color_space", "")
    matrix = "bt709" if cs == "bt709" or (cs in ("", "unknown") and h >= 720) else ("bt2020" if "bt2020" in cs else "bt601")
    return dict(w=w, h=h, fps=fps, dur=dur, frames=nb, audio=a["codec_name"] if a else None, matrix=matrix,
                hdr=v.get("color_transfer") in ("smpte2084", "arib-std-b67"))


def target_size(w, h, preset):
    bw, bh = RES_PRESETS[preset]
    if h > w:
        bw, bh = bh, bw
    s = min(bw / w, bh / h)
    return int(round(w * s / 2)) * 2, int(round(h * s / 2)) * 2


def decoder_cmd(src, info, ss=None, t=None):
    cmd = [FFMPEG, "-v", "error"]
    if ss is not None:
        cmd += ["-ss", str(ss)]
    cmd += ["-i", src]
    if t is not None:
        cmd += ["-t", str(t)]
    cmd += ["-an", "-sn", "-dn", "-vf", f"scale=in_color_matrix={info['matrix']},format=rgb24",
            "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    return cmd


def resolve_src(path_text):
    src = (path_text or "").strip().strip('"')
    if not src:
        raise gr.Error("Hãy chọn video trước (nút 📂 Chọn video…, kéo thả vào cửa sổ, hoặc dán đường dẫn).")
    if not os.path.isfile(src):
        raise gr.Error(f"Không tìm thấy file: {src}")
    return src


def fmt_time(s):
    s = int(s)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


# ----------------------------------------------------------------------------- chuc nang
def fast_size(info, tw, th, scale, fast):
    """Che do tang toc: kich thuoc dau vao sao cho mo hinh phong to vua dung (tw, th). None = khong ap dung."""
    if not fast:
        return None
    pw, ph = int(round(tw / scale)), int(round(th / scale))
    return (pw, ph) if pw < info["w"] else None


_awake = {"n": 0, "lock": threading.Lock(), "thread": None}


def keep_awake(on):
    """Khong cho Windows tu ngu trong luc dang xu ly video dai (man hinh van tat duoc binh thuong)."""
    if os.name != "nt":
        return
    with _awake["lock"]:
        _awake["n"] = _awake["n"] + 1 if on else max(_awake["n"] - 1, 0)
        if on and not (_awake["thread"] and _awake["thread"].is_alive()):
            def loop():
                import ctypes

                while _awake["n"] > 0:
                    ctypes.windll.kernel32.SetThreadExecutionState(0x00000001)  # ES_SYSTEM_REQUIRED: reset bo dem ngu
                    time.sleep(30)

            _awake["thread"] = threading.Thread(target=loop, daemon=True)
            _awake["thread"].start()


def preview(path_text, model_name, res, tile, fast):
    src = resolve_src(path_text)
    info = probe(src)
    tw, th = target_size(info["w"], info["h"], res)
    ss = info["dur"] / 2 if info["dur"] else 0
    raw = subprocess.run(decoder_cmd(src, info, ss=ss) [:-1] + ["-frames:v", "1", "pipe:1"], capture_output=True, creationflags=NOWIN).stdout
    if len(raw) < info["w"] * info["h"] * 3:
        raw = subprocess.run(decoder_cmd(src, info)[:-1] + ["-frames:v", "1", "pipe:1"], capture_output=True, creationflags=NOWIN).stdout
    frame = np.frombuffer(raw[: info["w"] * info["h"] * 3], np.uint8).reshape(info["h"], info["w"], 3)
    net, scale = get_model(model_name)
    pre = fast_size(info, tw, th, scale, fast)
    upscale_frame(net, scale, frame, tw, th, [int(tile)], pre)  # lam nong GPU de uoc tinh chinh xac
    t0 = time.time()
    out = upscale_frame(net, scale, frame, tw, th, [int(tile)], pre)
    dt = time.time() - t0
    before = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_CUBIC)
    est = dt * info["frames"]
    msg = (f"**Video gốc:** {info['w']}×{info['h']} · {float(info['fps']):.3f} fps · {info['frames']} khung hình · "
           f"{fmt_time(info['dur'])}  \n**Đầu ra:** {tw}×{th}  \n"
           f"Mỗi khung hình ≈ {dt:.2f}s → ước tính toàn bộ video ≈ **{fmt_time(est)}**")
    if trt_available(model_name) and int(tile) == 0:
        msg += f"  \n🚀 Khi chạy thật app dùng TensorRT (cùng chất lượng) → thường chỉ còn ≈ **{fmt_time(est / 1.6)}**"
    return (before, out), msg


def process(path_text, model_name, res, codec, cq, test_seconds, tile, fast):
    src = resolve_src(path_text)
    info = probe(src)
    tw, th = target_size(info["w"], info["h"], res)
    net, scale = get_model(model_name)
    test = test_seconds and test_seconds > 0
    total = info["frames"] if not test else min(info["frames"], int(test_seconds * info["fps"]))
    stem = Path(src).stem
    tag = {3840: "4K", 2160: "4K", 2560: "2K", 1440: "2K"}.get(max(tw, th), f"{th}p")
    out_path = OUT_DIR / f"{stem}_{tag}{'_test' if test else ''}.mp4"
    n = 1
    while out_path.exists():
        out_path = OUT_DIR / f"{stem}_{tag}{'_test' if test else ''}_{n}.mp4"
        n += 1

    # TensorRT (nhanh ~2.3 lan, ket qua nhu PyTorch) - lan dau voi moi do phan giai can ~1 phut de toi uu
    pre = fast_size(info, tw, th, scale, fast)
    in_w, in_h = pre or (info["w"], info["h"])
    runner, engine_note = None, "PyTorch"
    if trt_available(model_name) and int(tile) == 0:
        try:
            if not TRTRunner(model_name, in_h, in_w).cached:
                yield ("⚙️ Đang tối ưu mô hình cho card đồ họa (TensorRT)… Chỉ làm **lần đầu** cho mỗi độ phân giải "
                       "video, mất khoảng 1 phút. Những lần sau sẽ bắt đầu ngay."), ""
            runner = get_trt(model_name, in_h, in_w)
            engine_note = "🚀 TensorRT"
        except Exception as e:  # loi bat ky -> quay ve PyTorch, van chay binh thuong
            print("TensorRT loi, dung PyTorch:", repr(e))
            runner = None

    enc = [FFMPEG, "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "yuv420p", "-color_range", "tv", "-colorspace", "bt709", "-s", f"{tw}x{th}", "-framerate", str(info["fps"]), "-i", "pipe:0"]
    if info["audio"]:
        enc += ["-i", src] + (["-t", str(test_seconds)] if test else [])
        enc += ["-map", "0:v:0", "-map", "1:a:0"]
        enc += ["-c:a", "copy"] if info["audio"] in COPY_AUDIO else ["-c:a", "aac", "-b:a", "320k"]
    enc += CODECS[codec] + ["-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", str(int(cq)), "-b:v", "0",
                            "-pix_fmt", "yuv420p", "-color_range", "tv",
                            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                            "-movflags", "+faststart", "-shortest", str(out_path)]

    log_path, log = None, None
    for name in ("ffmpeg_log.txt", f"ffmpeg_log_{os.getpid()}_{int(time.time())}.txt"):
        try:  # file log cu co the dang bi tien trinh khac giu -> dung ten khac, khong de loi
            log_path = OUT_DIR / name
            log = open(log_path, "w", encoding="utf-8")
            break
        except OSError:
            continue
    if log is None:
        log_path, log = os.devnull, open(os.devnull, "w")
    dec_p = subprocess.Popen(decoder_cmd(src, info, t=test_seconds if test else None),
                             stdout=subprocess.PIPE, stderr=log, bufsize=0, creationflags=NOWIN)
    enc_p = subprocess.Popen(enc, stdin=subprocess.PIPE, stderr=log, bufsize=0, creationflags=NOWIN)
    RUNNING.update((dec_p, enc_p))
    fsize = info["w"] * info["h"] * 3
    q_in, q_out = queue.Queue(maxsize=8), queue.Queue(maxsize=8)
    err = []

    def reader():
        try:
            while True:
                # doc thang vao vung nho cap san (noi bytes tung manh nho rat cham: ~1GB copy/khung hinh)
                buf = bytearray(fsize)
                mv, got = memoryview(buf), 0
                while got < fsize:
                    k = dec_p.stdout.readinto(mv[got:])
                    if not k:
                        break
                    got += k
                if got < fsize:
                    break
                q_in.put(buf)
        except Exception as e:
            err.append(e)
        q_in.put(None)

    def writer():
        try:
            while (item := q_out.get()) is not None:
                enc_p.stdin.write(item)
        except Exception as e:
            err.append(e)
            while q_out.get() is not None:  # xa hang doi de luong chinh khong bi treo
                pass

    threads = [threading.Thread(target=reader, daemon=True), threading.Thread(target=writer, daemon=True)]
    for t in threads:
        t.start()

    warn = "\n\n⚠️ Video gốc là HDR – kết quả sẽ là SDR, màu có thể hơi nhạt." if info["hdr"] else ""
    head = f"**{Path(src).name}** {info['w']}×{info['h']} → **{tw}×{th}** · {model_name}{" · ⚡ tăng tốc" if fast_size(info, tw, th, scale, fast) else ""}{warn}\n\n"
    tile_box = [int(tile)]
    keep_awake(True)
    done, t0, last = 0, time.time(), 0
    prev_buf, prev_out, dup = None, None, 0
    stream = torch.cuda.Stream() if DEVICE.type == "cuda" else None
    finished = False
    try:
        while (buf := q_in.get()) is not None:
            if prev_buf is not None and buf == prev_buf:  # khung giong het tung byte -> dung lai ket qua
                out, dup = prev_out, dup + 1
            else:
                frame = np.frombuffer(buf, np.uint8).reshape(info["h"], info["w"], 3)
                with torch.cuda.stream(stream) if stream is not None else _nullctx():
                    out = upscale_frame(net, scale, frame, tw, th, tile_box, pre, yuv=True, runner=runner)
                prev_buf, prev_out = buf, out
            q_out.put(out)
            done += 1
            now = time.time()
            if now - last > 1.0:
                last = now
                fps = done / (now - t0)
                eta = (total - done) / fps if fps > 0 else 0
                pct = 100 * done / max(total, 1)
                dup_note = f" · {dup} khung trùng bỏ qua" if dup else ""
                yield head + (f"⏳ {done}/{total} khung hình ({pct:.1f}%) · {fps:.2f} fps · còn lại ≈ {fmt_time(eta)}"
                              f" · {engine_note}{dup_note}"), ""
            if err:
                break
        q_out.put(None)
        threads[1].join()
        if err:
            dec_p.kill()
        try:
            enc_p.stdin.close()
        except OSError:
            pass
        enc_p.wait()
        dec_p.wait()
        finished = True
    finally:
        if not finished:  # bi huy giua chung
            for p in (dec_p, enc_p):
                p.kill()
            log.close()
            out_path.unlink(missing_ok=True)
        RUNNING.difference_update((dec_p, enc_p))
        keep_awake(False)
    log.close()
    if err or enc_p.returncode != 0:
        raise gr.Error(f"Lỗi khi xử lý: {err[0] if err else ''} – xem {log_path}")
    el = time.time() - t0
    mb = out_path.stat().st_size / 1e6
    dup_note = f" · {dup} khung trùng bỏ qua" if dup else ""
    yield head + (f"✅ **Xong!** {done} khung hình trong {fmt_time(el)} ({done / el:.2f} fps) · {mb:.0f} MB"
                  f" · {engine_note}{dup_note}"), str(out_path)


WINDOW = None  # cua so pywebview (None khi chay bang trinh duyet)
VIDEO_TYPES = ("Video (*.mp4;*.mkv;*.mov;*.avi;*.webm;*.m4v;*.wmv;*.flv;*.ts;*.mts;*.m2ts;*.3gp;*.mpg;*.mpeg)",
               "Tất cả file (*.*)")


def pick_file(current):
    """Mo hop thoai chon file cua Windows - chi lay DUONG DAN, khong nap file vao giao dien."""
    if WINDOW is None:
        gr.Info("Đang chạy bằng trình duyệt: hãy dán đường dẫn file vào ô bên dưới.")
        return current
    import webview

    res = WINDOW.create_file_dialog(webview.FileDialog.OPEN, allow_multiple=False, file_types=VIDEO_TYPES)
    return res[0] if res else current


def video_info(path_text):
    src = (path_text or "").strip().strip('"')
    if not src:
        return "🎅 Chọn video (nút **📂 Chọn video…** hoặc kéo thả vào cửa sổ) rồi bấm **Xem thử**."
    if not os.path.isfile(src):
        return f"⚠️ Không tìm thấy file: `{src}`"
    try:
        info = probe(src)
    except gr.Error as e:
        return f"⚠️ {e}"
    gb = os.path.getsize(src) / 1e9
    return (f"🎞️ **{Path(src).name}** · {info['w']}×{info['h']} · {float(info['fps']):.3f} fps · "
            f"{fmt_time(info['dur'])} · {info['frames']:,} khung hình · {gb:.2f} GB  \n"
            f"Bấm **🔍 Xem thử** để so sánh chất lượng và ước tính thời gian xử lý.")


def open_folder(path):
    p = Path(path) if path else OUT_DIR
    if p.is_file():
        subprocess.Popen(["explorer", "/select,", str(p)])
    else:
        os.startfile(OUT_DIR)


# ----------------------------------------------------------------------------- giao dien
APP_TITLE = "Video 4K Noel"


def _theme_files():
    bg = base64.b64encode((ASSETS / "background.jpg").read_bytes()).decode()
    css = (ASSETS / "theme.css").read_text(encoding="utf-8").replace("__BG__", f"data:image/jpeg;base64,{bg}")
    head = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
            '<link href="https://fonts.googleapis.com/css2?family=Mountains+of+Christmas:wght@700&display=swap" rel="stylesheet">'
            f"<script>{(ASSETS / 'effects.js').read_text(encoding='utf-8')}</script>")
    return css, head


with gr.Blocks(title=APP_TITLE) as demo:
    if DEVICE.type == "cuda":
        gpu_html = f"🎮 GPU: <b>{torch.cuda.get_device_name(0)}</b>"
    else:
        gpu_html = ('<span class="warn">⚠️ Không thấy GPU NVIDIA (hoặc driver cũ) – app sẽ chạy bằng CPU, rất chậm. '
                    'Hãy cập nhật driver NVIDIA mới nhất.</span>')
    gr.HTML(f'<h1><span class="bell">🔔</span> Video → 4K Noel <span class="bell r">🔔</span></h1>'
            f'<div class="sub">{gpu_html} · 📁 Kết quả lưu ở <b>{OUT_DIR}</b></div>', elem_id="xmas-header")
    with gr.Row(equal_height=True):
        # Hang 1: cot trai = dau vao & cai dat | cot phai = khung so sanh (cao bang cot trai)
        with gr.Column(scale=4, min_width=380, elem_classes="xmas-panel"):
            # Chi lay DUONG DAN file - khong tai video vao giao dien (video lon se lam sap cua so)
            gr.HTML('<div class="drop-row"><div class="drop-ico">🎁</div><div><b>Kéo thả video vào bất kỳ đâu trong cửa sổ</b>'
                    '<br><span>hoặc bấm <b>📂 Chọn video</b> · video dài/nặng bao nhiêu cũng được</span></div></div>',
                    elem_id="xmas-drop", padding=False)
            with gr.Row(equal_height=True):
                path_text = gr.Textbox(label="🎞️ Video nguồn", placeholder=r"D:\Videos\clip.mp4",
                                       max_lines=1, scale=5, elem_id="xmas-path")
                pick_btn = gr.Button("📂 Chọn video", scale=2, min_width=150, elem_id="xmas-pick")
            model = gr.Dropdown(list(MODELS), value=list(MODELS)[0], label="🤖 Mô hình",
                                info="Video quay thật → Đa dụng · Hoạt hình → Anime")
            res = gr.Radio(list(RES_PRESETS), value=list(RES_PRESETS)[0], label="📺 Độ phân giải đầu ra",
                           info="4K = 3840×2160 · 2K = 2560×1440 · Full HD = 1920×1080", elem_id="xmas-res")
            fast = gr.Checkbox(value=False, label="⚡ Tăng tốc cho video gốc ≥ 720p",
                               info="Nhanh hơn 3–4 lần · bấm Xem thử để so sánh chất lượng")
            with gr.Accordion("⚙️ Tùy chọn nâng cao", open=False):
                test_s = gr.Number(value=0, precision=0, label="Chạy thử N giây đầu (0 = cả video)")
                codec = gr.Radio(list(CODECS), value=list(CODECS)[0], label="Chuẩn nén")
                cq = gr.Slider(12, 32, value=19, step=1, label="Chất lượng nén (CQ) – số nhỏ = đẹp hơn, file lớn hơn")
                tile = gr.Slider(0, 1024, value=0, step=64,
                                 label="Tile (0 = tự động; đặt 256–512 nếu báo thiếu VRAM)")
        with gr.Column(scale=6, min_width=480, elem_classes="xmas-panel"):
            slider = gr.ImageSlider(label="Trước (phóng to thường) ↔ Sau (Real-ESRGAN)", type="numpy",
                                    height=310, elem_id="xmas-slider")
            status = gr.Markdown("🎅 Chọn video rồi bấm **Xem thử** để so sánh và ước tính thời gian.",
                                 elem_id="xmas-status", container=True)
            with gr.Row(equal_height=True):
                out_file = gr.Textbox(label="📦 File kết quả", interactive=False, max_lines=1, scale=3)
                gr.Button("📂 Mở thư mục", size="md", scale=1, elem_id="xmas-open").click(open_folder, out_file)
    # Hang 2: thanh nut bam ngang toan bo chieu rong
    with gr.Row(elem_classes="xmas-panel xmas-actions", equal_height=True):
        btn_prev = gr.Button("🔍 Xem thử 1 khung hình", scale=2, size="lg")
        btn_run = gr.Button("🚀 Bắt đầu upscale", variant="primary", scale=3, size="lg")
        btn_stop = gr.Button("⏹ Dừng", variant="stop", scale=1, size="lg")

    pick_btn.click(pick_file, path_text, path_text)
    path_text.change(video_info, path_text, status)
    btn_prev.click(preview, [path_text, model, res, tile, fast], [slider, status])
    job = btn_run.click(process, [path_text, model, res, codec, cq, test_s, tile, fast], [status, out_file])
    btn_stop.click(lambda: "⏹ Đã dừng.", None, status, cancels=[job])

def _shutdown():
    for p in list(RUNNING):
        try:
            p.kill()
        except Exception:
            pass
    os._exit(0)


def main():
    if not shutil.which(FFMPEG):
        raise SystemExit("Không tìm thấy ffmpeg.")
    use_browser = "--browser" in sys.argv
    css, head = _theme_files()
    demo.queue().launch(
        server_name="127.0.0.1", inbrowser=False, prevent_thread_lock=True, ssr_mode=False, quiet=FROZEN,
        css=css, head=head, theme=gr.themes.Soft(primary_hue="red", secondary_hue="green"),
        favicon_path=str(ASSETS / "icon.png"), footer_links=[],
    )
    url = demo.local_url.rstrip("/") + "/?__theme=light"
    if not use_browser:
        try:
            import webview  # cua so rieng (Edge WebView2) thay vi trinh duyet

            from webview.dom import DOMEventHandler

            global WINDOW
            WINDOW = webview.create_window(APP_TITLE, url, width=1440, height=920, min_size=(1000, 700), maximized=True,
                                           background_color="#a8000f")

            def on_drop(e):  # keo tha: pywebview cho biet duong dan that tren o dia
                files = e.get("dataTransfer", {}).get("files", [])
                path = next((f.get("pywebviewFullPath") for f in files if f.get("pywebviewFullPath")), None)
                if path:
                    WINDOW.evaluate_js(f"window.xmasSetPath && window.xmasSetPath({json.dumps(path)})")

            def on_loaded():
                WINDOW.dom.document.events.drop += DOMEventHandler(on_drop, True, True)

            WINDOW.events.loaded += on_loaded
            webview.start()
            _shutdown()  # dong cua so -> thoat app
        except Exception as e:  # may khong co WebView2 -> mo bang trinh duyet
            print("pywebview loi, chuyen sang trinh duyet:", e)
    import webbrowser

    webbrowser.open(url)
    try:
        demo.block_thread()
    finally:
        _shutdown()


if __name__ == "__main__":
    main()
