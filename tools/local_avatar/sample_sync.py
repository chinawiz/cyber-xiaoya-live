"""推流同时采样 /health，看**说话过程中**的滞后分布。

   PORT=9880 SECONDS=12 python sample_sync.py

关心三个量：
  lag_frames  播放时钟要求第 k 帧、而最新生成帧还在 k-n —— 说话中持续不为 0 才是真落后
  reanchors   时钟重锚次数，说话中增长 = 画面在突然快进
  clock       播放时钟（秒），应稳定逼近 pcm_s - av_delay
"""
import asyncio
import json
import os
import time

import requests
import websockets

PORT = int(os.environ.get("PORT", "9880"))
SECONDS = float(os.environ.get("SECONDS", "12"))
PCM = os.environ.get("PCM", "/tmp/xiaoya24.pcm")
BASE = f"http://127.0.0.1:{PORT}"


def health() -> dict:
    try:
        return requests.get(BASE + "/health", timeout=2).json()
    except Exception:  # noqa: BLE001
        return {}


async def main() -> None:
    pcm = open(PCM, "rb").read()
    step = 2400 * 2
    n_bytes = int(SECONDS * 48000)
    rows = []
    t0 = time.time()
    stop = False

    async def sampler():
        while not stop:
            h = health()
            rows.append((round(time.time() - t0, 2), h.get("lag_frames"), h.get("reanchors"),
                         h.get("gen_frames"), h.get("clock"), h.get("pcm_s"),
                         h.get("queued")))
            await asyncio.sleep(0.25)

    task = asyncio.create_task(sampler())
    ws_url = f"ws://127.0.0.1:{PORT}/live/v1/external-lives/sync/stream"
    async with websockets.connect(ws_url, max_size=None) as ws:
        await ws.send(json.dumps({"type": 1, "payload": {}}))
        for i in range(0, min(len(pcm), n_bytes), step):
            await ws.send(pcm[i:i + step])
            await asyncio.sleep(0.1)
        # 说话结束后继续采 3s，区分「说话中落后」和「结尾静止」
        await asyncio.sleep(3.0)
    stop = True
    await task

    speak_end = SECONDS + 0.5
    talk = [r for r in rows if r[0] <= speak_end and r[1] is not None]
    tail = [r for r in rows if r[0] > speak_end]
    print(f"{'t':>5} {'lag':>4} {'re':>3} {'gen':>4} {'clock':>6} {'pcm':>5} {'q':>3}")
    for r in rows[::4]:
        print(f"{r[0]:>5} {str(r[1]):>4} {str(r[2]):>3} {str(r[3]):>4} "
              f"{str(r[4]):>6} {str(r[5]):>5} {str(r[6]):>3}")

    def stats(rs, name):
        if not rs:
            return
        lags = [r[1] or 0 for r in rs]
        res = [r[2] or 0 for r in rs]
        print(f"[{name}] 采样 {len(rs)} 次  lag max={max(lags)} 均值={sum(lags)/len(lags):.1f} "
              f"| reanchor 增长 {res[-1]-res[0]} | gen {rs[-1][3]-rs[0][3]} 帧")

    stats(talk, "说话中")
    stats(tail, "说话后")


asyncio.run(main())
