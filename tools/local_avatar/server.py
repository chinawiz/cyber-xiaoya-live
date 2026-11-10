#!/usr/bin/env python3
"""本地数字人渲染服务 —— Vidu 云端渲染的替换实现（跑在 GPU 机 :9880）。

对桥伪装成 Vidu 组件版：
- HTTP  POST /live/s_avatar/component          建会话（返回 live.id + managed_rtc）
- HTTP  GET  /live/v1/lives/{id} 等             状态/录播/积分/资产库（占位）
- WS    /live/v1/external-lives/{id}/stream    原样复用 Vidu 的 type 1/2/5/7/9/10/11 协议
                                               二进制帧 = 裸 PCM s16le/24k/mono
对舞台页提供本地流：
- GET  /stream   MJPEG 20fps（本地模型推理出的画面）
- GET  /audio    裸 PCM 24kHz 分块流（Web Audio 播放）

桥侧只需把 config.vidu.host 指到本服务、provider=local，其余 LLM/TTS/弹幕链路全不变。
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from renderer import FPS, build_renderer

HOST = os.environ.get("LOCAL_AVATAR_HOST", "0.0.0.0")
PORT = int(os.environ.get("LOCAL_AVATAR_PORT", "9880"))
PUBLIC = os.environ.get("LOCAL_AVATAR_PUBLIC", f"http://192.168.50.98:{PORT}")
AVATAR = os.environ.get("LOCAL_AVATAR_IMAGE", str(Path.home() / "local-avatar" / "avatar.jpg"))
CHECKPOINT = os.environ.get("LOCAL_AVATAR_CKPT",
                            str(Path.home() / "models" / "wav2lip" / "wav2lip_gan.pth"))
UPSTREAM = os.environ.get("LOCAL_AVATAR_UPSTREAM", str(Path.home() / "Wav2Lip"))

app = FastAPI(title="local-avatar (Vidu-compatible)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

renderer = build_renderer(AVATAR, CHECKPOINT, UPSTREAM)
# 帧率跟随后端：Ditto 的 25fps 是模型决定的，MuseTalk/Wav2Lip 才看 LOCAL_AVATAR_FPS
FPS = int(getattr(renderer, "fps", FPS))
# 音视频统一延迟：Ditto 首帧要等一个 LMDM 窗口，音频必须同步延后，
# 否则画面永远比声音慢一截。其它后端为 0。
AV_DELAY = float(getattr(renderer, "av_delay", 0.0))
sessions: dict[str, dict] = {}


class AudioHub:
    """把桥送来的 PCM 广播给所有音频订阅者，并喂给渲染器。

    AV_DELAY > 0 时（Ditto），每块音频带上「应当播出的 wall 时刻」，
    /audio 端点按这个时刻节流输出 —— 音频被整体推后，和视频共用一条时钟。
    """

    def __init__(self):
        self.subs: set[asyncio.Queue] = set()
        self.bytes_in = 0

    def publish(self, chunk: bytes) -> None:
        if not chunk:
            return
        self.bytes_in += len(chunk)
        renderer.push_audio(chunk)
        emit_at = time.monotonic() + AV_DELAY
        for q in list(self.subs):
            try:
                q.put_nowait((emit_at, chunk))
            except asyncio.QueueFull:
                pass

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self.subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self.subs.discard(q)


hub = AudioHub()


def _frame_loop() -> None:
    period = 1.0 / FPS
    while True:
        t0 = time.monotonic()
        try:
            renderer.tick()
        except Exception as e:  # noqa: BLE001
            print(f"[render] frame error: {type(e).__name__}: {e}", flush=True)
            time.sleep(0.2)
        dt = period - (time.monotonic() - t0)
        if dt > 0:
            time.sleep(dt)


@app.on_event("startup")
async def _startup() -> None:
    threading.Thread(target=_frame_loop, name="local-avatar-frames", daemon=True).start()
    print(f"[server] backend={renderer.backend} stream={PUBLIC}/stream audio={PUBLIC}/audio",
          flush=True)


# ─────────────────────────── Vidu 兼容 HTTP ───────────────────────────

def _rtc() -> dict:
    return {"provider": "local", "source": "managed",
            "channel_id": "local", "user_id": "viewer", "token": "",
            "stream_url": f"{PUBLIC}/stream", "audio_url": f"{PUBLIC}/audio"}


@app.post("/live/s_avatar/component")
async def create_component(body: dict):
    lid = f"local-{time.time_ns()}"
    sessions[lid] = {"id": lid, "created": time.time(), "status": "live",
                     "body": body, "billed_seconds": 0}
    print(f"[session] create {lid} model={body.get('model')} avatar={body.get('avatar_id')}",
          flush=True)
    return {"live": {"id": lid, "status": "live"}, "managed_rtc": _rtc()}


@app.get("/live/v1/lives/{live_id}")
async def get_live(live_id: str):
    s = sessions.get(live_id, {})
    return {"live": {"id": live_id, "status": s.get("status", "ended"),
                     "billed_seconds": s.get("billed_seconds", 0),
                     "credits_cost": 0, "end_time": int(time.time())}}


@app.get("/live/v1/lives/{live_id}/recording")
async def get_recording(live_id: str):
    return {"recording": {"enabled": False, "video_urls": []}}


@app.get("/live/v1/lives")
async def list_lives():
    return {"items": [], "total": 0}


@app.get("/ent/v2/credits")
async def credits():
    # 本地模型不计费；给积分面板一个恒定的 0 消耗
    return {"remains": [{"credit_remain": 0, "credit_type": "local"}]}


@app.get("/live/v1/avatars")
async def list_avatars():
    return {"items": [], "avatars": []}


@app.post("/live/v1/avatars")
async def create_avatar(body: dict):
    return {"avatar": {"id": "local", "image_uri": body.get("image_uri", "")}}


@app.delete("/live/v1/avatars/{avatar_id}")
async def delete_avatar(avatar_id: str):
    return {"ok": True}


@app.get("/live/v1/action-libraries")
async def list_libs():
    return {"items": []}


@app.post("/live/v1/action-libraries")
async def create_lib(body: dict):
    return {"id": "0"}


@app.delete("/live/v1/action-libraries/{lib_id}")
async def delete_lib(lib_id: str):
    return {"ok": True}


@app.get("/live/v1/action-libraries/{lib_id}/actions")
async def list_actions(lib_id: str):
    return {"items": []}


@app.post("/live/v1/action-libraries/{lib_id}/actions")
async def create_action(lib_id: str, body: dict):
    return {"id": "0"}


@app.delete("/live/v1/action-libraries/{lib_id}/actions/{action_id}")
async def delete_action(lib_id: str, action_id: str):
    return {"ok": True}


@app.post("/tools/v2/files/uploads")
async def upload_begin(body: dict):
    return {"id": "local", "put_url": f"{PUBLIC}/_upload/local"}


@app.put("/_upload/{fid}")
async def upload_put(fid: str, request: Request):
    await request.body()
    return JSONResponse({}, headers={"etag": '"local"'})


@app.put("/tools/v2/files/uploads/{fid}/finish")
async def upload_finish(fid: str, body: dict):
    return {"uri": "local://upload"}


@app.get("/health")
async def health():
    # cursor/skipped 是排查音画同步的关键：cursor 是模型最近的帧号，skipped 是
    # 因为播放时钟已经跑过而跳过的帧数。lag_frames 持续增长 = 生成速度跟不上
    # 25fps，reanchors 增长说明已经触发过时钟重锚（画面会有一点点快进）。
    info = {"ok": True, "backend": renderer.backend, "frames": renderer.frames,
            "fps_target": FPS, "av_delay": AV_DELAY,
            "audio_bytes": hub.bytes_in, "sessions": len(sessions),
            "compose": getattr(renderer, "compose", ""),
            # 权重版本：v15 的唇形与清晰度明显好于 v10，验收时要能一眼看到用哪个
            "version": getattr(renderer, "version", ""),
            "sharpen": getattr(renderer, "sharpen", 0),
            "cursor": getattr(renderer, "cursor", 0),
            "skipped": getattr(renderer, "skipped", 0),
            "pcm_s": round(getattr(getattr(renderer, "pcm", None), "seconds", 0.0), 2),
            "gen_ms": round(getattr(renderer, "gen_ms", 0.0), 1),
            "feat": getattr(renderer, "feat_refresh", 0)}
    extra = getattr(renderer, "extra_health", None)
    if callable(extra):
        info.update(extra())
    return info


# ─────────────────────────── Vidu 兼容 WS 信令 ───────────────────────────

@app.websocket("/live/v1/external-lives/{live_id}/stream")
async def stream_ws(ws: WebSocket, live_id: str):
    await ws.accept()
    conn_id = ws.query_params.get("conn_id", "")
    seq = 0
    print(f"[ws] connected live={live_id} conn={conn_id}", flush=True)

    async def reply(t: int, payload: dict) -> None:
        nonlocal seq
        seq += 1
        await ws.send_text(json.dumps({"type": t, "live_id": live_id,
                                       "conn_id": conn_id, "seq_id": seq,
                                       "payload": payload}, ensure_ascii=False))

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            raw = msg.get("bytes")
            if raw is not None:                       # 二进制帧 = 裸 PCM
                hub.publish(raw)
                continue
            text = msg.get("text")
            if not text:
                continue
            try:
                env = json.loads(text)
            except Exception:
                continue
            t = env.get("type")
            p = env.get("payload") or {}
            if t == 1:                                # conn_init → ack
                await reply(2, {"conn_init_ack": {"success": True, "error_code": ""}})
            elif t == 9:                             # input_transcription = 动作指令
                txt = ((p.get("text_msg") or {}).get("content") or "").strip()
                if txt:
                    renderer.set_action(txt)
            elif t == 7:                             # interrupt
                renderer.set_action("")
                _interrupt = getattr(renderer, "interrupt", None)
                if callable(_interrupt):
                    _interrupt()
            elif t == 10 or t == 11:                 # 字幕 / 参考图：本地画面不做换装
                pass
            elif t == 5:                             # hangup
                s = sessions.get(live_id)
                if s:
                    s["status"] = "ended"
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        print(f"[ws] closed: {type(e).__name__}: {e}", flush=True)
    finally:
        print(f"[ws] disconnected live={live_id}", flush=True)


# ─────────────────────────── 本地音视频流 ───────────────────────────

@app.get("/stream")
async def mjpeg():
    boundary = b"--frame\r\n"

    async def gen():
        while True:
            jpg = renderer.snapshot()
            if jpg:
                yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            await asyncio.sleep(1.0 / FPS)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store",
                                      "Access-Control-Allow-Origin": "*"})


@app.get("/audio")
async def pcm_stream():
    q = hub.subscribe()
    silence = np.zeros(2400, dtype="<i2").tobytes()   # 100ms 静音保活

    async def gen():
        try:
            while True:
                try:
                    emit_at, chunk = await asyncio.wait_for(q.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    # 空闲时也按 100ms 节奏补静音：音频时间轴必须和 wall clock 1:1，
                    # 否则静音"补"出来的时长会比真实时间短，画面会慢慢跑到声音前面
                    chunk, emit_at = silence, time.monotonic() + AV_DELAY
                wait = emit_at - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
                yield bytes(chunk)
        finally:
            hub.unsubscribe(q)

    return StreamingResponse(gen(), media_type="application/octet-stream",
                             headers={"Cache-Control": "no-store",
                                      "X-Audio-Format": "s16le/24000/1",
                                      "Access-Control-Allow-Origin": "*"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, log_level="info", ws_max_size=None)
