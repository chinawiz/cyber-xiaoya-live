#!/usr/bin/env python3
"""一键复核「Vidu 云端 → 本地 Wav2Lip」改造是否生效。

跑完后 verify-evidence/ 下会留下可核查的原始证据：
  00-run.log          本次复核的完整输出
  01-health-before    起会话前的渲染服务状态
  02-live-start       桥的 /api/live/start 返回（关键：provider=local + stream_url）
  03-say              一句话从 TTS 到本地渲染的推流增量
  04-live-status      会话中的运行状态
  05-live-stop        断开返回（关键：billed_seconds=0 / credits_cost=0）
  06-health-after     渲染服务累计帧数
  10-frame-silence.jpg / 11-frame-talking.jpg  真实出帧
  12-compare.jpg      静音 vs 说话 对照
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

BRIDGE = "http://127.0.0.1:8795"
AVATAR = "http://192.168.50.98:9880"
OUT = Path(__file__).parent / "verify-evidence"
LOG: list[str] = []


def log(*parts) -> None:
    line = " ".join(str(p) for p in parts)
    print(line, flush=True)
    LOG.append(line)


def dump(name: str, obj) -> None:
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def first_jpeg(buf: bytes) -> bytes:
    s = buf.find(b"\xff\xd8")
    if s < 0:
        return b""
    e = buf.find(b"\xff\xd9", s)
    return buf[s:e + 2] if e > s else b""


def grab_frame() -> bytes:
    """从 MJPEG 流里读一帧就断开（/stream 是无限流，不能直接 get）。"""
    buf = b""
    with httpx.Client(timeout=httpx.Timeout(8), trust_env=False) as c:
        with c.stream("GET", AVATAR + "/stream") as r:
            for chunk in r.iter_bytes():
                buf += chunk
                if b"\xff\xd9" in buf:
                    break
    return first_jpeg(buf)


def health() -> dict:
    with httpx.Client(timeout=6, trust_env=False) as c:
        return c.get(AVATAR + "/health").json()


def main() -> None:
    if not OUT.exists():
        OUT.mkdir(parents=True)
    with httpx.Client(timeout=90, trust_env=False) as c:
        log("[1] 渲染服务健康检查")
        hb = health()
        log("   ", json.dumps(hb, ensure_ascii=False))
        dump("01-health-before.json", hb)
        # 换模型后这里要跟着改：ditto(warp 形变) / musetalk(256² 重绘) / wav2lip(96²)
        assert hb.get("backend") in ("ditto", "musetalk", "wav2lip"), f"渲染后端异常: {hb}"
        if hb.get("backend") == "ditto":
            # Ditto 的帧率由模型决定（25fps），音视频统一延迟 av_delay 必须设置，
            # 否则首帧延迟会让画面永远追不上声音
            assert hb.get("fps_target") == 25, f"Ditto 应跑 25fps: {hb}"
            assert hb.get("av_delay", 0) >= 1.0, f"未设置音视频统一延迟: {hb}"
            assert hb.get("compose") == "ditto", f"合成路径异常: {hb}"
            if (hb.get("reanchors") or 0) > 0:
                log(f"    [注意] 已发生 {hb['reanchors']} 次时钟重锚（生成跟不上 25fps 时会触发）")
        elif hb.get("backend") == "musetalk":
            # blend = 语义蒙版 + 整块重建图 alpha 合成（当前方案，融合最自然）；
            # residual 系列已被实测否决（模型重建偏离原图，差值会在嘴角烧出亮带）。
            assert hb.get("compose") == "blend", f"MuseTalk 合成策略应为 blend: {hb}"
            assert hb.get("version") in ("v15", "v10"), f"未报告权重版本: {hb}"
            if hb.get("version") == "v10":
                log("    [警告] 当前用的是 v10 权重，v15 的唇形和清晰度明显更好")

        log("[2] POST /api/live/start")
        st = time.time()
        r = c.post(BRIDGE + "/api/live/start", json={})
        start = r.json()
        log(f"    http={r.status_code} 耗时={time.time() - st:.1f}s")
        log("   ", json.dumps(start, ensure_ascii=False))
        dump("02-live-start.json", start)
        rtc = start.get("rtc") or {}
        assert rtc.get("provider") == "local", f"没有切到本地 provider: {rtc}"
        assert rtc.get("stream_url", "").startswith(AVATAR), "stream_url 未指向本地渲染服务"
        assert not rtc.get("token"), "本地模式不应下发 RTC token"

        log("[3] 静音帧")
        f_idle = grab_frame()
        (OUT / "10-frame-silence.jpg").write_bytes(f_idle)
        log(f"    {len(f_idle)} bytes")

        log("[4] POST /api/say（TTS → 推流 → 本地渲染 全链路）")
        base = health()["audio_bytes"]
        t0 = time.time()
        rr = c.post(BRIDGE + "/api/say",
                    json={"text": "大家好，我是赛博小雅。我现在跑在本地显卡上，"
                                  "唇形同步模型已经换成本地 256 分辨率的了，云端 Vidu 不再参与。"})
        log(f"    http={rr.status_code} {rr.text[:200]}")
        delta, prev, stable = 0, -1, 0
        for _ in range(60):
            time.sleep(1.0)
            cur = health()["audio_bytes"]
            delta = cur - base
            stable = stable + 1 if cur == prev else 0
            prev = cur
            if delta > 96000 and stable >= 2:   # 推到 2s 音频且 2 秒内不再增长 = 本轮说完
                break
        f_talk = grab_frame()
        (OUT / "11-frame-talking.jpg").write_bytes(f_talk)
        say = {"ok": rr.json().get("ok"), "audio_bytes_base": base,
               "audio_bytes_delta": delta, "audio_seconds_est": round(delta / 48000, 2),
               "elapsed_s": round(time.time() - t0, 1), "frame_bytes": len(f_talk)}
        log("   ", json.dumps(say, ensure_ascii=False))
        dump("03-say.json", say)
        assert delta > 96000, "本地渲染服务没有收到音频，TTS→渲染链路断了"

        log("[5] GET /api/live/status")
        ls = c.get(BRIDGE + "/api/live/status").json()
        dump("04-live-status.json", ls)
        log("   ", json.dumps({k: ls.get(k) for k in ("active", "phase", "mode", "live_id")},
                              ensure_ascii=False))

        log("[6] POST /api/live/stop")
        stop = c.post(BRIDGE + "/api/live/stop").json()
        dump("05-live-stop.json", stop)
        log("   ", json.dumps(stop, ensure_ascii=False))
        ended = stop.get("ended") or {}
        assert ended.get("billed_seconds") == 0, "本地模式不应产生计费秒数"
        assert ended.get("credits_cost") == 0, "本地模式不应扣积分"

        log("[7] 渲染服务累计状态")
        ha = health()
        dump("06-health-after.json", ha)
        log("   ", json.dumps(ha, ensure_ascii=False))

    log(f"\n全部断言通过：桥已走本地 provider，TTS→本地模型({hb.get('backend')}) 有真实音频输入，断开零计费。")


if __name__ == "__main__":
    try:
        main()
    finally:
        (OUT / "00-run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
