#!/usr/bin/env python3
"""OmniVoice 服务入口（跨平台版，供 cyber-xiaoya-live 桥调用）。

复用上游 build_demo 的「语音克隆 / 音色设计」两个 Tab，所以 gradio API
名 `/_clone_fn`、`/_design_fn` 与桥 src/tts/omnivoice.py 完全对齐。

与仓库原 demo_flashinfer.py 的区别：
- 原文件在 GitHub 上是「无换行符的单行文本」，无法被 Python 解析；这里重建为合法脚本。
- FlashInfer / CUDA Graph 改为可选（--no-flashinfer），aarch64 / DGX Spark
  没有对应 wheel 时自动回退到上游 eager 路径，服务照常可用。
"""
from __future__ import annotations

import argparse
import logging
import os
import pathlib
import threading
import time

import torch
from omnivoice import OmniVoice
from omnivoice.cli.demo import build_demo


def _start_cache_gc() -> None:
    """按 mtime 清理 gradio 落盘的合成 wav，避免长跑无限累积。"""
    every = int(os.environ.get("OMNIVOICE_GC_EVERY", "600"))
    age = int(os.environ.get("OMNIVOICE_GC_AGE", "900"))
    root = pathlib.Path(os.environ.get("GRADIO_TEMP_DIR") or "/tmp/gradio")
    print(f"[gc] 合成产物清理: 每 {every}s 扫 {root}, 删超过 {age}s 的", flush=True)

    def loop() -> None:
        while True:
            time.sleep(every)
            try:
                now = time.time()
                for f in root.rglob("*"):
                    try:
                        if f.is_file() and now - f.stat().st_mtime > age:
                            f.unlink()
                    except OSError:
                        pass
            except Exception:
                pass

    threading.Thread(target=loop, name="omnivoice-cache-gc", daemon=True).start()


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(prog="omnivoice-service")
    ap.add_argument("--model", default="k2-fsa/OmniVoice")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ip", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9877)
    ap.add_argument("--share", action="store_true", default=False)
    ap.add_argument("--no-asr", action="store_true", default=False)
    ap.add_argument("--no-flashinfer", action="store_true", default=False,
                    help="跳过 FlashInfer/CUDA Graph，直接用上游 eager 路径")
    ap.add_argument("--no-cuda-graph", action="store_true", default=False)
    ap.add_argument("--graph-buckets", default="5,10,15,20,30")
    args = ap.parse_args()

    logging.info("Loading model from %s, device=%s ...", args.model, args.device)
    model = OmniVoice.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=torch.float16,
        load_asr=not args.no_asr,
    )
    print("Model loaded.", flush=True)

    if not args.no_flashinfer:
        try:
            from omnivoice.models.omnivoice_flashinfer import apply_flashinfer
            buckets = [int(b) for b in args.graph_buckets.split(",") if b.strip()] \
                if args.graph_buckets.strip() else None
            apply_flashinfer(model,
                             enable_cuda_graph=not args.no_cuda_graph,
                             cuda_graph_buckets=buckets)
            print(f"FlashInfer applied. cuda_graph={not args.no_cuda_graph} "
                  f"| buckets={buckets}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] FlashInfer 不可用，回退 eager: {type(e).__name__}: {e}",
                  flush=True)

    demo = build_demo(model, args.model)
    _start_cache_gc()
    demo.queue().launch(server_name=args.ip, server_port=args.port,
                        share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
