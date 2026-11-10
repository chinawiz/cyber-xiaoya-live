# local-avatar —— 本地 GPU 数字人渲染服务（Vidu 云端渲染的替代实现）

跑在局域网 GPU 机器上，监听 `:9880`，**对桥伪装成 Vidu 组件版 API**，所以桥侧的
LLM / TTS / 弹幕 / 点播 / 动作链路一行都不用改 —— 只换 `config.vidu.host` 和 `provider`。

> 定位：**主用链路**。Vidu 云端保留为备用（画质更好、动作库更强，但按秒计费）。

---

## 1. 为什么要有它

| | Vidu 云端 | local-avatar |
|---|---|---|
| 成本 | 1.5 积分/秒 ≈ \$0.0075/秒 ≈ **\$0.45/分钟**（\$27/小时） | 电费 |
| 画面 | 720p，25–42 FPS，支持动作库 / 实时换装 / 换背景 | 256²（MuseTalk）/ 96²（Wav2Lip）贴回原图，无动作库 |
| 延迟 | 云端排队 + 网络 | 局域网，取决于本机 GPU |
| 合规 | 第三方人脸数据出本机 | 全本地 |

实测（DGX Spark aarch64 + MuseTalk 1.5，256²，`verify-evidence/01-health-before.json`）：

- 单帧推理 `gen_ms` ≈ **38.5 ms** → 理论上限约 26 FPS，而目标输出是 20 FPS
- 余量只有 ~20%：TTS 合成、音频重采样、MJPEG 编码一抢 CPU/显存，就会掉帧、口型追不上声音
- **结论：代码链路是通的、协议是对的，卡的是这台机器的算力。** 不是实现问题。

想真正流畅，三条路（见 §5）：换更高算力的 GPU / 换更轻的模型后端 / 把渲染搬到云端。

---

## 2. 它对外暴露什么

伪装成 Vidu 组件版（桥不用改）：

| 端点 | 说明 |
|---|---|
| `POST /live/s_avatar/component` | 建会话，返回 `live.id` + `managed_rtc`（带 `stream_url` / `audio_url`） |
| `GET  /live/v1/lives/{id}` | 状态；录播 / 积分 / 资产库接口为占位实现 |
| `WS   /live/v1/external-lives/{id}/stream` | 复用 Vidu 的 type 1/2/5/7/9/10/11 协议；二进制帧 = 裸 PCM s16le / 24 kHz / mono |

给舞台页用的本地流：

| 端点 | 说明 |
|---|---|
| `GET /stream` | MJPEG，默认 20 FPS，舞台页用 `<img>` 直接播 |
| `GET /audio` | 裸 PCM 24 kHz 分块流，舞台页用 Web Audio 按 100 ms 排程播放 |
| `GET /health` | 健康检查，`run-local.sh` 靠它判断是否需要拉起服务 |

---

## 3. 后端（`LOCAL_AVATAR_BACKEND`）

| 后端 | 分辨率 | 特点 | 显存 | 备注 |
|---|---|---|---|---|
| `ditto`（默认） | 原图 | warp 形变，**无贴片感**，模型硬编码 25 fps | 中 | 首帧要等一个 LMDM 窗口，音频按 `av_delay` 整体后移对齐 |
| `musetalk` | 256² | 腾讯 Lyra Lab，潜在空间单步修复，原生 30 FPS 设计 | ≥ 8–12 GB | 本项目实测主用；MIT 许可 |
| `wav2lip` | 96² | 经典 GAN 兜底，任意脸都能跑 | ≥ 8 GB | 嘴部糊，只适合验证链路 |
| `mock` | 占位帧 | 静态图，无需 GPU | 0 | 调 UI / 链路用 |

`LOCAL_AVATAR_COMPOSE` 只对 `musetalk` / `wav2lip` 生效，控制重建图贴回原图的合成方式：
`blend`（默认，语义蒙版 + alpha 合成）/ `residual-seg`（语义蒙版 + 差值叠加）/ `residual`（旧椭圆先验，仅作对照）。

---

## 4. 启动

### GPU 机（Linux / aarch64 亦可）

```bash
mkdir -p ~/local-avatar && cp -r <仓库>/tools/local_avatar/* ~/local-avatar/
cp 你的正脸照.jpg ~/local-avatar/avatar.jpg        # 不提供形状就用代码生成的占位形象
bash ~/local-avatar/start.sh
# 日志：tail -f ~/local-avatar/server.log
```

`start.sh` 会自动挑 Python 解释器（`ditto-venv` → `MuseTalk/venv` → 系统 python3），
并按后端设置 `PYTHONPATH`（TensorRT 在 aarch64 没有 PyPI wheel，需要从 deb 解包到 `~/trt`）。

### 桥侧（本机）

`config.json`：

```json
{
  "vidu": {
    "provider": "local",
    "model": "local-musetalk",
    "host": "http://<GPU机IP>:9880",
    "api_key": "",
    "avatar_id": ""
  },
  "avatar": { "avatar_id": "local" },
  "rtc": { "mode": "managed", "managed_provider": "local" }
}
```

macOS 一键起（自动拉起 GPU 侧服务 + 转发 TTS 9877 + 起桥）：

```bash
bash run-local.sh            # 前台运行，Ctrl+C 停
bash run-local.sh --no-browser
```

---

## 5. 性能不够时怎么选

| 方案 | 代价 | 适用 |
|---|---|---|
| 换更轻的后端（`wav2lip` / 降 `LOCAL_AVATAR_FPS` 到 15） | 画质下降 | 只想先能播 |
| 加显卡 / 换高算力机型（MuseTalk 官方在 V100 上 256² 可跑 30+ FPS） | 钱 | 想要 256² 且不掉帧 |
| 量化 / TensorRT / CUDA Graph（`ditto` 后端已用 TRT） | 工程量 | 榨干现有卡 |
| 回 Vidu 云端（`provider=cloud`） | \$0.45/分钟 | 要画质和动作库，能接受计费 |

---

## 6. 已知限制

- **无动作库**：`直发动作` 在本地模式只在画面下方提示，不会真的做动作（云端动作库是 Vidu 的资产）
- **无实时换装 / 换背景**：参考图功能在本地模式无效
- **无服务端录播**：`⏺ 自动录播` 走的是 Vidu 侧录播 API，本地模式需自己用 OBS/ffmpeg 录
- 舞台页的统计面板在本地模式只剩「记忆 / Token」一行（积分、钻石、约钻石是平台口径，已隐藏）
