# /mnt/bigdisk/ultralytics/ultralytics/FLOPs.py
import gc
import sys
import time
from statistics import median

import torch
from thop import profile

from ultralytics import RTDETR

YAML = "/mnt/bigdisk/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-l.yaml"


def build_model():
    m = RTDETR(YAML)
    return m.model.eval().to("cuda")


def measure_params_flops():
    net = build_model()
    dummy = torch.randn(1, 3, 640, 640, device="cuda")
    with torch.no_grad():
        macs, params = profile(net, inputs=(dummy,))
    print(f"Params: {params / 1e6:.2f} M")
    print(f"FLOPs:  {macs / 1e9:.2f} G")
    del net, dummy
    torch.cuda.empty_cache()
    gc.collect()


def bench_latency(bsz=1, amp=False, iters=300, warmup=50):
    net = build_model()  # 全新实例，且在新进程里，不受 THOP hook 影响
    make_x = lambda b: torch.randn(b, 3, 640, 640, device="cuda")

    with torch.no_grad():
        for _ in range(warmup):
            x = make_x(bsz)
            with torch.cuda.amp.autocast(enabled=amp):
                _ = net(x)
    torch.cuda.synchronize()

    ts = []
    with torch.no_grad():
        for _ in range(iters):
            x = make_x(bsz)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.cuda.amp.autocast(enabled=amp):
                _ = net(x)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)

    ts.sort()
    avg = sum(ts) / len(ts)
    med = median(ts)
    p90 = ts[int(0.9 * len(ts))]
    fps = (1000.0 / avg) * bsz
    print(f"{'AMP' if amp else 'FP32'}  b={bsz}: Avg {avg:.2f}ms  Med {med:.2f}  P90 {p90:.2f}  FPS~{fps:.1f}")


if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench_latency(bsz=1, amp=False)
        bench_latency(bsz=8, amp=True)
    else:
        measure_params_flops()
