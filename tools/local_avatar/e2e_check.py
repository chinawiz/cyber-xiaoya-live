"""端到端验证：通过桥协议推 PCM → 抓 MJPEG 帧 → 输出张嘴对比图。

    PORT=9880 OUT=/tmp/e2e python e2e_check.py [pcm]

pcm 需为 24kHz / s16le / mono（桥推给渲染服务的就是这个格式）。
"""
import asyncio
import json
import os
import threading
import time

import cv2
import numpy as np
import requests
import websockets

PORT = int(os.environ.get("PORT", "9880"))
OUT = os.environ.get("OUT", "/tmp/e2e")
PCM_PATH = os.environ.get("PCM", "/tmp/xiaoya24.pcm")
SECONDS = float(os.environ.get("SECONDS", "11"))
BASE = f"http://127.0.0.1:{PORT}"
WS = f"ws://127.0.0.1:{PORT}/live/v1/external-lives/e2e/stream"
os.makedirs(OUT, exist_ok=True)

frames: list[tuple[float, np.ndarray]] = []
stop = False


def reader() -> None:
    try:
        r = requests.get(BASE + "/stream", stream=True, timeout=120)
        buf = b""
        for chunk in r.iter_content(8192):
            if stop:
                break
            buf += chunk
            while True:
                a = buf.find(b"\xff\xd8")
                b = buf.find(b"\xff\xd9")
                if a < 0 or b < 0 or b < a:
                    break
                jpg = buf[a:b + 2]
                buf = buf[b + 2:]
                im = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if im is not None:
                    frames.append((time.time(), im))
    except Exception as e:  # noqa: BLE001
        print("reader:", type(e).__name__, e, flush=True)


async def main() -> None:
    global stop
    t = threading.Thread(target=reader, daemon=True)
    t.start()
    await asyncio.sleep(1.5)                       # 先拿到静音基线
    base_n = len(frames)

    pcm = open(PCM_PATH, "rb").read()
    step = 2400 * 2                                # 100ms @24k s16le
    n_bytes = int(SECONDS * 48000)
    t0_push = time.time()

    async with websockets.connect(WS, max_size=None) as ws:
        await ws.send(json.dumps({"type": 1, "payload": {}}))
        for i in range(0, min(len(pcm), n_bytes), step):
            await ws.send(pcm[i:i + step])
            await asyncio.sleep(0.1)
        pushed = min(len(pcm), n_bytes) / 48000
        print(f"pushed {pushed:.2f}s audio in {time.time() - t0_push:.2f}s", flush=True)
        # 留足时间：Ditto 是攒够一个 LMDM 窗口才吐一批帧，尾部还有一批在算
        await asyncio.sleep(float(os.environ.get("TAIL", "4.5")))
    stop = True
    await asyncio.sleep(0.3)

    talk = frames[base_n:]
    print(f"frames: baseline={base_n} talk={len(talk)} total={len(frames)}", flush=True)
    if not frames:
        return
    base = frames[0][1]
    if not talk:
        print("没有新增帧", flush=True)
        return

    # 帧间隔 → 实际出帧率
    ts = [t for t, _ in talk]
    if len(ts) > 2:
        span = ts[-1] - ts[0]
        print(f"出帧速率 {len(talk) / span:.1f} fps（含重复帧）", flush=True)

    scored = sorted(((np.abs(im.astype(np.float32) - base.astype(np.float32)).mean(), i, im)
                     for i, (_, im) in enumerate(talk)), key=lambda x: -x[0])
    print("top diffs:", [(round(s, 2), i) for s, i, _ in scored[:5]], flush=True)
    for rank, (d, _i, im) in enumerate(scored[:4]):
        cv2.imwrite(f"{OUT}/talk{rank}_d{d:.1f}.png", im)
    cv2.imwrite(f"{OUT}/baseline.png", base)

    tiles = []
    for j in list(range(0, len(talk), max(1, len(talk) // 8)))[:8]:
        im = talk[j][1]
        h, w = im.shape[:2]
        c = im[int(h * 0.30):int(h * 0.62), int(w * 0.20):int(w * 0.80)]
        c = cv2.resize(c, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
        tiles.append(c)
    if tiles:
        cv2.imwrite(f"{OUT}/seq.png", np.hstack(tiles))
    print("->", OUT, flush=True)


asyncio.run(main())
