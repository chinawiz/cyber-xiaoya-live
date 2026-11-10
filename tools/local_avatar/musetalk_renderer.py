#!/usr/bin/env python3
"""MuseTalk 256×256 唇形驱动 —— 替换 Wav2Lip 的 96×96 分辨率上限。

【为什么要换】
    Wav2Lip 的生成分辨率固定 96×96，而本项目肖像的脸框是 222×222，
    贴回时必须放大约 2.3 倍，唇线、牙齿、人中这些高频细节在插值里全部丢掉，
    所以画面"偏软"是模型上限决定的，靠后处理（锐化/蒙版）救不回来。
    MuseTalk 的生成分辨率是 256×256，唇部信息量是 Wav2Lip 的 7 倍。

【为什么不用官方的 scripts/inference.py】
    官方脚本是「整段音频 → 整段视频」的离线管线，并且依赖：
      - mmpose / mmcv (DWPose 关键点)  —— 官方 wheel 无 aarch64 版本，CUDA13 下源码编译必挂
    但本项目是**固定机位静态肖像**：
      - 脸框已经人工标定在 box.txt，不需要关键点检测
      - 融合蒙版是常量，只需初始化时算一次
    于是关键点这条依赖链可以整条去掉，剩下三个纯 diffusers 前向：
      whisper encoder → UNet 单步 → VAE decode

【融合：为什么必须走语义蒙版 + 整块 alpha 合成】
    早期版本用「生成图 - 静音帧」的**差值**贴回原图，再配一个椭圆高斯空间先验
    当蒙版。结果嘴部像一块浮在脸上的贴纸，怎么调参数都救不回来，原因有三：
      1. 差值在空间上是碎片化的（唇线处大、脸颊处 0），放大 2.3 倍后变成
         一片锯齿状色斑，边缘不可能连续；
      2. 椭圆先验和真实嘴部轮廓无关，羽化区压在法令纹/下巴上，过渡带里
         既不是原图也不是生成图，是两者按错误权重搅拌的产物；
      3. 模型对同一张脸在不同嘴型下会产生不同的 VAE 重建噪声，
         差值把这些噪声也一起放大，逐帧抖动 = "浮"。

    官方 MuseTalk 的做法（musetalk/utils/blending.py）完全不同：
      get_image_prepare_material  → 拿 crop 区域做 face-parsing 语义分割，
                                    只保留「脸 + 上唇 + 下唇 + 口腔」这几类像素，
                                    再切掉上半张脸，最后高斯羽化，得到 mask；
      get_image_blending          → 把**完整重建图**（不是差值）按 mask 与
                                    原图做 alpha 合成。
    这样贴回去的是模型重新画过的整块下半脸（含皮肤、阴影、下巴轮廓），
    光照和肤色本来就与上下文同源，羽化带里是两张同源图的插值，不会有接缝。

    我们照抄这条路径，只是把 mask 在初始化时算一次（固定机位，人物不动），
    每帧退化成一次 resize + 一次逐像素 lerp。

【流式怎么做的】
    官方推理用 timesteps=[0]，即单步去噪，没有迭代依赖，天生可以逐帧算。
    每帧只消费音频特征中的 [i*2-4, i*2+6) 这 10 个 20ms 帧（前后各 40ms 上下文），
    因此只要音频到达 (i/25 + 0.12) 秒就能算出第 i 帧 —— 画面只比音频晚 120ms。

【与音频的同步】
    视频帧号不是靠定时器自增，而是由「累计音频时长」反推：
        target = floor((audio_seconds - 0.20) * 25)
    这样 TTS 停顿时不推音频，画面就停在静音帧，不会空转跑飞；
    音频爆发推送时则按需跳帧追赶，保证嘴型始终贴着声音。
"""
from __future__ import annotations

import math
import os
import sys
import threading
import time

import cv2
import numpy as np

from renderer import FPS, SR_MEL, BaseRenderer, resample_24k_to_16k

# MuseTalk 的音频特征帧率固定 50Hz（whisper encoder 对 10ms mel 做 2 倍下采样）
AUDIO_FPS = 50
# 每个视频帧消费的 whisper 特征帧数 = 2*(left+right+1)，官方默认 left=right=2
CTX = 5
CLIP_LEN = 2 * CTX          # 10
# 模型总输出 256×256，输入同样 256（VAE 下采样 8 倍 → latent 32×32）
GEN = 256
# 面部裁剪框相对人脸框的放大系数。
# 官方 blending.get_crop_box 用 1.5，但那是给 DWPose 的检测框用的 ——
# 那种框里人脸只占中间一小块，扩 1.5 倍后脸仍然填满 256 输入。
# 本项目的 box.txt 是人工标的**紧脸框**（额头到下巴刚好占满），再扩 1.5 倍
# 等于让脸只占 256 里的 67%，而 MuseTalk 训练集（VHDTF）里脸基本都是填满的，
# 构图一旦偏离，生成质量立刻下降 —— 实测嘴唇发灰、唇形发钝。
# 1.12 时脸占 256 的 90%，既贴近训练分布，又留出足够的下巴/颈侧给蒙版羽化。
CROP_EXPAND = 1.12


class PCMBuffer:
    """带绝对时间轴的 16kHz 单声道缓冲。

    与 renderer.AudioRing 的区别：环形缓冲只保留最近 N 秒，拿不到"第 3.2 秒的音频"；
    这里维护绝对样本索引，可以按任意时间区间切片，这是逐帧对齐的前提。
    """

    def __init__(self, keep_sec: float = 60.0):
        self.keep = int(keep_sec * SR_MEL)
        self.buf = np.zeros(0, dtype=np.float32)
        self.start = 0                     # buf[0] 对应的绝对样本号
        self.lock = threading.Lock()

    def push_24k(self, pcm: bytes) -> None:
        x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        if not x.size:
            return
        y = resample_24k_to_16k(x)
        with self.lock:
            self.buf = np.concatenate((self.buf, y))
            if self.buf.size > self.keep:
                drop = self.buf.size - self.keep
                self.buf = self.buf[drop:]
                self.start += drop

    @property
    def seconds(self) -> float:
        """累计音频时长（不含停顿间隙）。"""
        with self.lock:
            return (self.start + self.buf.size) / SR_MEL

    def slice(self, t0: float, t1: float) -> np.ndarray:
        with self.lock:
            a = max(0, int(t0 * SR_MEL) - self.start)
            b = min(self.buf.size, int(t1 * SR_MEL) - self.start)
            if b <= a:
                return np.zeros(0, dtype=np.float32)
            return self.buf[a:b].copy()


class MuseTalkRenderer(BaseRenderer):
    """256×256 MuseTalk 唇形同步渲染器（流式逐帧）。"""

    def __init__(self, avatar_path: str = "", muse_dir: str = "",
                 models_dir: str = "", use_fp16: bool = True):
        super().__init__(avatar_path)
        import torch
        self.torch = torch
        self.backend = "musetalk"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pcm = PCMBuffer()
        self.cursor = 0                # 下一帧要渲染的帧号
        self.feat = None               # (1, T, 5, 384)
        self.feat_base = 0             # self.feat[0] 对应的绝对 whisper 帧号
        self.lock_render = threading.Lock()
        self.gen_ms = 0.0
        self.skipped = 0
        self._idle_jpg = b""
        self.feat_refresh = 0
        self.blend_frame = 0
        self.half = 0                  # 蒙版只覆盖 crop 下半，合成只算这一段

        # ── 合成方式 ───────────────────────────────────────────────
        # blend（默认）：官方路径，语义蒙版 + 整块重建图 alpha 合成
        # residual-seg：语义蒙版 + 差值叠加 —— 融合区由分割决定（解剖学正确），
        #               但像素仍以高清原图为底，只加模型算出来的变化量，
        #               因此不会带入 VAE 重建的糊和色偏
        # residual    ：旧路径，差值 + 椭圆先验，留作对照
        # direct      ：整块硬贴（无羽化），用来确认蒙版本身有没有问题
        self.compose = (os.environ.get("LOCAL_AVATAR_COMPOSE", "blend").strip().lower()
                        or "blend")
        if self.compose in ("blend", "mask", "seg", "semantic"):
            self.compose = "blend"
        # face-parsing 的蒙版范围，决定"融合发生在多大一块脸上"：
        #   lips（默认）：只取嘴唇本身，再按脸高比例膨胀一圈 → 覆盖完整开合范围，
        #                 但不碰两颊和下巴边缘。色偏/亮度漂移被限制在嘴周，
        #                 脸上不会出现圆形亮斑（整脸蒙版的主要毛病）。
        #   raw ：官方默认，脸皮 + 上下唇 + 口腔（不含脖子和鼻子）
        #   jaw ：1.5 新增，在 raw 之上把下颌线一并扩进来（更激进）
        self.seg_mode = os.environ.get("LOCAL_AVATAR_SEG_MODE", "lips").strip().lower() or "lips"
        # lips 模式的膨胀半径（相对脸框高度）。太小 → 嘴张大时下巴的形变被切掉；
        # 太大 → 退化成整脸蒙版，色斑又回来了。
        # 实测 0.20（crop 空间 44px）时融合区已经盖住嘴角外侧，模型把嘴角画成
        # 一条硬横线的错误被整块贴回来；0.10（≈22px）只覆盖嘴本身，干净得多。
        try:
            self.lip_dilate = max(0.02, float(
                os.environ.get("LOCAL_AVATAR_LIP_DILATE", "0.10")))
        except ValueError:
            self.lip_dilate = 0.10
        # 蒙版保留比例：只留下 crop 的下半张脸（上半不参与合成，和 split decode 对齐）
        try:
            self.upper_ratio = min(0.9, max(0.2,
                float(os.environ.get("LOCAL_AVATAR_UPPER_RATIO", "0.5"))))
        except ValueError:
            self.upper_ratio = 0.5
        # 羽化核相对 crop 宽度的比例（官方 0.1；调小接缝更硬，调大嘴部更"雾"）
        try:
            self.feather = max(0.01, float(os.environ.get("LOCAL_AVATAR_FEATHER", "0.06")))
        except ValueError:
            self.feather = 0.06
        # 色彩校准开关（默认**关闭**）。
        # 逐通道 gain/bias 是全局线性校正，当初是为了压住「整脸蒙版 + v10 权重」
        # 的系统性色偏（嘴唇发白、皮肤发灰）才引入的。
        # 现在蒙版收窄到嘴唇、权重换成 1.5，模型输出的色度本身就贴着原图 ——
        # 再用一组按静音帧拟合的 gain/bias 去乘，等于给"本来就对"的像素
        # 加了一层莫名的逐通道拉伸：张开的嘴缝是暗部，暗部被 gain 放大后
        # 通道比例失衡，于是画面上就出现一条橄榄绿/墨绿横线（实测消融确认）。
        # 需要旧行为时用 LOCAL_AVATAR_COLOR_FIX=1 打开。
        self.color_fix = os.environ.get("LOCAL_AVATAR_COLOR_FIX", "0").strip() not in (
            "0", "off", "false", "no")
        # 色度键控：模型的色度只在「亮度被明显改动」的地方放行（张嘴后的口腔、
        # 露出的牙齿），其余位置（唇面、皮肤）一律沿用原图色度。
        # 这是消灭黄斑/绿斑的最后一道闸 —— 逐通道 gain 校正是全局线性的，
        # 拗不过模型在嘴角画出的局部偏色。
        self.chroma_key = os.environ.get("LOCAL_AVATAR_CHROMA_KEY", "1").strip() not in (
            "0", "off", "false", "no")

        try:
            self.sharpen = float(os.environ.get("LOCAL_AVATAR_SHARPEN", "0.35"))
        except ValueError:
            self.sharpen = 0.35
        # 以下三项只对 residual 回退路径生效
        try:
            self.lowfreq = float(os.environ.get("LOCAL_AVATAR_LOWFREQ", "0"))
        except ValueError:
            self.lowfreq = 0.0
        try:
            self.mask_soft = max(0.12, float(os.environ.get("LOCAL_AVATAR_MASK_SOFT", "0.35")))
        except ValueError:
            self.mask_soft = 0.35
        try:
            self.gain = max(0.5, min(2.0, float(os.environ.get("LOCAL_AVATAR_GAIN", "1.25"))))
        except ValueError:
            self.gain = 1.25
        try:
            self.mask_prior = float(os.environ.get("LOCAL_AVATAR_MASK_PRIOR", "1"))
        except ValueError:
            self.mask_prior = 1.0

        if not (muse_dir and os.path.isdir(muse_dir)):
            raise RuntimeError(f"MuseTalk 代码目录不存在: {muse_dir}")
        if muse_dir not in sys.path:
            sys.path.insert(0, muse_dir)
        self.muse_dir = muse_dir

        from musetalk.models.unet import UNet, PositionalEncoding
        from musetalk.models.vae import VAE
        from musetalk.utils.audio_processor import AudioProcessor
        from transformers import WhisperModel

        mp = models_dir
        # 1.5 的 UNet 结构与 1.0 完全一致（musetalk.json 逐字节相同），
        # 差别只在权重：1.5 的训练加了感知损失 / GAN 损失 / 同步损失，
        # 唇部清晰度和身份一致性都明显更好。有就优先用。
        v15_cfg = os.path.join(mp, "musetalkV15", "musetalk.json")
        v15_ck = os.path.join(mp, "musetalkV15", "unet.pth")
        v10_cfg = os.path.join(mp, "musetalk", "musetalk.json")
        v10_ck = os.path.join(mp, "musetalk", "pytorch_model.bin")
        want = os.environ.get("LOCAL_AVATAR_MUSETALK_VERSION", "auto").strip().lower()
        if want in ("", "auto"):
            use_v15 = os.path.isfile(v15_cfg) and os.path.isfile(v15_ck)
        else:
            use_v15 = want in ("v15", "1.5")
        if use_v15 and os.path.isfile(v15_cfg) and os.path.isfile(v15_ck):
            unet_cfg, unet_ck, self.version = v15_cfg, v15_ck, "v15"
        else:
            unet_cfg, unet_ck, self.version = v10_cfg, v10_ck, "v10"
        vae_dir = os.path.join(mp, "sd-vae")
        whisper_dir = os.path.join(mp, "whisper")
        for p in (unet_cfg, unet_ck):
            if not os.path.isfile(p):
                raise RuntimeError(f"缺少 MuseTalk 权重: {p}")

        t_begin = time.time()
        self.vae = VAE(model_path=vae_dir)
        self.unet = UNet(unet_config=unet_cfg, model_path=unet_ck, device=self.device)
        self.pe = PositionalEncoding(d_model=384).to(self.device)

        # 精度策略：UNet 用 fp16 提速；VAE 保持 fp32，sd-vae 在 fp16 下 decode
        # 偶发 NaN（暗部会炸成色块），而 decode 只占单帧耗时的两成，不值得冒这个险。
        self.dtype = torch.float16 if (use_fp16 and self.device == "cuda") else torch.float32
        # 排查开关：某些显卡上 sd-vae 半精度会让 decode 输出崩成色块，
        # 用 LOCAL_AVATAR_FP32_VAE=1 退回 fp32 对比（慢一倍但不失真）
        fp32_vae = os.environ.get("LOCAL_AVATAR_FP32_VAE", "0").strip() == "1"
        if self.dtype == torch.float16:
            self.unet.model = self.unet.model.half()
            # decode 是单帧耗时的绝对大头（实测 UNet 12ms / VAE 44ms），
            # 半精度能砍掉一半以上；官方 inference 的 use_float16 也是这么做的。
            if not fp32_vae:
                self.vae.vae = self.vae.vae.half()
        self.unet.model = self.unet.model.to(self.device).eval()
        self.pe = self.pe.to(self.device)
        # PositionalEncoding 的正弦 buffer 默认是 fp32，forward 里只做了 .to(device)
        # 没转 dtype，于是 x(fp16) + pe(fp32) 会把 hidden_states 悄悄升回 fp32，
        # 到 UNet 的 cross-attn 就炸 "expected mat1 and mat2 to have the same dtype"。
        self.pe.pe = self.pe.pe.to(dtype=self.dtype)

        self.whisper = WhisperModel.from_pretrained(whisper_dir)
        self.whisper = self.whisper.to(device=self.device, dtype=self.dtype).eval()
        self.whisper.requires_grad_(False)
        self.fe = AudioProcessor(feature_extractor_path=whisper_dir).feature_extractor
        self.n_layers = int(self.whisper.config.encoder_layers) + 1   # 含 embedding

        self.timesteps = torch.tensor([0], device=self.device)

        # ── 静态肖像预处理：裁剪框、参考 latent、融合蒙版，全部只算一次 ──
        self.crop = self._crop_rect()
        x0, y0, x1, y1 = self.crop
        self.face_bgr = self.base[y0:y1, x0:x1]
        self.face_256 = cv2.resize(self.face_bgr, (GEN, GEN),
                                   interpolation=cv2.INTER_LANCZOS4)
        with torch.no_grad():
            self.latents = self.vae.get_latents_for_unet(self.face_256)
        # 只解码下半张脸：嘴部变化全落在这里，decode 面积减半（256² 26ms → 128×256 13.5ms）。
        # 切口在 256 坐标 y=128，实测该处伪影只有 0.5/255 且 8 行内衰减完。
        # LOCAL_AVATAR_SPLIT=0 关闭分区，退化为整图 decode（排查用）。
        try:
            self.split = max(0, int(os.environ.get("LOCAL_AVATAR_SPLIT", "128")))
        except ValueError:
            self.split = 128
        self.face_low = (self.face_256[self.split:] if self.split
                         else self.face_256).astype(np.float32)
        # 语义融合蒙版（固定机位 → 常量，只算一次）
        self.mask, self.core = self._build_blend_mask()
        # residual 回退路径要用的几何权重
        self.alpha = self._build_alpha(y1 - y0, x1 - x0)

        # 原图细节层：整块替换最大的代价是把高清纹理换成 256 解码出来的软像，
        # 脸上会突然"失焦"一块。这里提前把原图的高频（纹理 + 唇纹）抽出来，
        # 合成完再补回去。
        #
        # 但补的范围必须躲开嘴唇：原图是**闭嘴**的，唇线正好落在唇缘外侧，
        # 全量补回去会和模型画出来的张嘴唇线叠成"双唇线"重影（实测嘴角两端
        # 会出现两道横向暗痕）。所以把唇部再向外扩一圈作为禁区，
        # detail 只在更外面的皮肤上生效。
        try:
            self.detail_amt = float(os.environ.get("LOCAL_AVATAR_DETAIL", "1.0"))
        except ValueError:
            self.detail_amt = 1.0
        _roi = self.face_bgr.astype(np.float32)
        self.detail = _roi - cv2.GaussianBlur(_roi, (0, 0), 2.0)
        _r = max(3, int(0.12 * (y1 - y0)) // 2 * 2 + 1)
        _lipzone = cv2.dilate((self.core[..., 0] > 0.35).astype(np.uint8),
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_r, _r)))
        _lipzone = cv2.GaussianBlur(_lipzone.astype(np.float32), (0, 0), _r / 3.0)
        _lipzone = np.clip(_lipzone, 0.0, 1.0)[..., None]
        self.detail_w = (self.mask * (1.0 - _lipzone) * self.detail_amt).astype(np.float32)

        # 静音基准帧 —— 三个用途，缺一不可：
        #   1. residual / residual-seg 拿它当相减基准，抵消 VAE 的系统性重建误差；
        #   2. blend 拿它做**色彩校准**（见 _fit_color）；
        #   3. 排查时对比"闭嘴时模型会画成什么样"。
        # 代价是启动时多一次 forward（约 40ms），可忽略。
        self.idle_low = None
        self.color_gain = np.ones(3, np.float32)
        self.color_bias = np.zeros(3, np.float32)
        self.lip_gain = np.ones(3, np.float32)
        self.lip_bias = np.zeros(3, np.float32)
        self.gain_map = np.ones((y1 - y0, x1 - x0, 3), np.float32)
        self.bias_map = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
        try:
            with torch.no_grad():
                zeros = torch.zeros((1, CLIP_LEN, self.n_layers, 384),
                                    dtype=self.dtype, device=self.device)
                self.idle_low = self._infer(zeros).astype(np.float32)
            print(f"[musetalk] 静音基准帧已就绪 "
                  f"(mean={self.idle_low.mean():.1f} vs 原图 {self.face_low.mean():.1f})",
                  flush=True)
            self._fit_color()
        except Exception as e:  # noqa: BLE001
            print(f"[musetalk] 静音基准帧计算失败，回退原图基准: "
                  f"{type(e).__name__}: {e}", flush=True)

        self.ready = True
        self._prepare_lo()
        print(f"[renderer] MuseTalk {self.version} loaded on "
              f"{torch.cuda.get_device_name(0)} "
              f"dtype={str(self.dtype).split('.')[-1]} crop={self.crop} "
              f"compose={self.compose} load={time.time() - t_begin:.1f}s", flush=True)

    # ─────────────────────────── 静态预处理 ───────────────────────────

    def _crop_rect(self) -> tuple[int, int, int, int]:
        """以人脸框中心取一个正方形区域 —— MuseTalk 要求 1:1 输入。

        crop 边长 = 脸框长边 × CROP_EXPAND（1.12）。这个值和官方 1.5 不同，
        原因是两边脸框的定义不一样：官方用 DWPose 检测框（脸只占中间一块），
        我们用的是人工标的紧脸框，扩 1.5 会让脸在 256 输入里只占 67%，
        偏离训练分布、生成质量下降。
        可用 LOCAL_AVATAR_CROP_EXPAND / LOCAL_AVATAR_CROP_SIDE 覆盖。
        """
        y1, y2, x1, x2 = self.box
        h, w = self.base.shape[:2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        try:
            side = int(os.environ.get("LOCAL_AVATAR_CROP_SIDE", "0"))
        except ValueError:
            side = 0
        if side <= 0:
            try:
                expand = float(os.environ.get("LOCAL_AVATAR_CROP_EXPAND", str(CROP_EXPAND)))
            except ValueError:
                expand = CROP_EXPAND
            side = int(max(x2 - x1, y2 - y1) * expand)
        side = min(side, h, w)                     # 不能超过画幅
        half = side // 2
        # 以中心展开，越界就整体平移回画幅内（不要缩小，保住分辨率）
        nx0 = min(max(0, cx - half), w - side)
        ny0 = min(max(0, cy - half), h - side)
        return nx0, ny0, nx0 + side, ny0 + side

    def _build_blend_mask(self) -> np.ndarray:
        """融合蒙版，返回 (H,W,1) 的 0~1 权重。固定机位 → 只算一次。

        骨架照抄官方 get_image_prepare_material（分割 → 限定脸框 → 切上半 →
        高斯羽化），但蒙版范围做了收紧，这是本项目和官方最大的差别：
          官方 raw 模式取「全部脸部皮肤 + 上下唇 + 口腔」，蒙版铺满整张脸。
          那是给视频用的 —— 每帧都在动，观众不会去比对静态像素。
          而本项目是**一张静止的高清肖像**，脸上任何逐帧漂移都会被抓出来。
          UNet 的 VAE 重建对同一张脸在不同音频条件下会产生轻微的整体亮度与
          色调漂移，整脸蒙版会把这些漂移全放进来，在脸颊上留下圆形亮斑。

          所以默认改用 lips 模式：只取 BiSeNet 的嘴唇类别，按脸高比例膨胀一圈，
          刚好覆盖嘴张开的完整形变范围。色偏被限制在嘴周一小块，
          而嘴周本来就是模型真正在改的地方，看不出问题。

        face-parsing 缺失时退回「脸框下半矩形 + 羽化」，同样是 alpha 合成。
        """
        from PIL import Image

        x_s, y_s, x_e, y_e = self.crop
        W, H = x_e - x_s, y_e - y_s
        by1, by2, bx1, bx2 = self.box
        body = Image.fromarray(self.base[:, :, ::-1])      # BGR → RGB
        face_large = body.crop((x_s, y_s, x_e, y_e))

        parsing = None
        fp = self._load_face_parser()
        if fp is not None:
            try:
                parsing = self._seg_parsing(face_large)    # (ph,pw) 类别图，0~18
            except Exception as e:  # noqa: BLE001
                print(f"[musetalk] face-parsing 推理失败，改用几何蒙版: "
                      f"{type(e).__name__}: {e}", flush=True)
                parsing = None

        if parsing is not None:
            ph, pw = parsing.shape
            sy, sx = ph / H, pw / W
            # 脸框内的像素才可能被融合：crop 比脸框大，外面是头发/背景/衣领
            gate = np.zeros((ph, pw), np.uint8)
            gy1, gy2 = int(max(0, by1 - y_s) * sy), int(min(H, by2 - y_s) * sy)
            gx1, gx2 = int(max(0, bx1 - x_s) * sx), int(min(W, bx2 - x_s) * sx)
            gate[gy1:gy2, gx1:gx2] = 1

            if self.seg_mode == "lips":
                core = np.isin(parsing, [11, 12, 13]).astype(np.uint8)
            elif self.seg_mode == "jaw":
                core = np.isin(parsing, [1, 10, 11, 12, 13]).astype(np.uint8)
            elif self.seg_mode == "neck":
                core = np.isin(parsing, [1, 11, 12, 13, 14]).astype(np.uint8)
            else:                                          # raw
                core = np.isin(parsing, [1, 11, 12, 13]).astype(np.uint8)
            core &= gate
            core_raw = core.copy()        # 嘴唇本身（不膨胀），用于低频门控

            if self.seg_mode == "lips":
                # BiSeNet 的 lips 只标唇红，嘴张大时下巴的形变、人中、法令纹
                # 全在外面，必须膨胀一圈包住，否则张嘴会「露边」
                r = max(3, int(self.lip_dilate * (by2 - by1) * sx))
                r = r // 2 * 2 + 1
                core = cv2.dilate(core, cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (r, r)))
                core &= gate
                note = f"dilate={r}px"
            else:
                note = "官方范围"
            arr = cv2.resize(core * 255, (W, H), interpolation=cv2.INTER_NEAREST)
            core_arr = cv2.resize(core_raw * 255, (W, H),
                                  interpolation=cv2.INTER_NEAREST)
            print(f"[musetalk] 语义蒙版: mode={self.seg_mode} {note} "
                  f"覆盖={float((arr > 127).mean()) * 100:.1f}% "
                  f"核心={float((core_arr > 127).mean()) * 100:.1f}%", flush=True)
        else:
            arr = np.zeros((H, W), np.uint8)
            core_arr = np.zeros((H, W), np.uint8)
            ry1, ry2 = max(0, by1 - y_s), max(0, by2 - y_s)
            rx1, rx2 = max(0, bx1 - x_s), max(0, bx2 - x_s)
            arr[ry1:ry2, rx1:rx2] = 255
            core_arr[ry1:ry2, rx1:rx2] = 255
            print("[musetalk] 几何蒙版（未启用 face-parsing）", flush=True)

        # 砍掉上半张脸
        top = int(H * self.upper_ratio)
        keep = np.zeros_like(arr)
        keep[top:] = arr[top:]
        keep_c = np.zeros_like(core_arr)
        keep_c[top:] = core_arr[top:]

        # 羽化：官方 k = int(0.1*width // 2 * 2) + 1，必须奇数
        k = max(3, int(self.feather * W) // 2 * 2 + 1)
        keep = cv2.GaussianBlur(keep, (k, k), 0)
        # 核心区羽化得窄得多 —— 它只用来放行低频，边界不需要柔和
        kc = max(3, int(0.04 * W) // 2 * 2 + 1)
        keep_c = cv2.GaussianBlur(keep_c, (kc, kc), 0)

        def _cut(mat: np.ndarray, kern: int) -> np.ndarray:
            """羽化会把上半脸也染上权重，再乘一道干净的切口。"""
            o = mat.astype(np.float32) / 255.0
            if top <= 0:
                return o
            o[:top] = 0.0
            ramp = max(6, kern // 2)
            edge = min(H, top + ramp)
            if edge > top:
                o[top:edge] *= np.linspace(0.0, 1.0, edge - top,
                                           dtype=np.float32)[:, None]
            return o

        return _cut(keep, k)[..., None], _cut(keep_c, kc)[..., None]

    def _load_face_parser(self):
        """加载 BiSeNet 语义分割（79999_iter.pth，约 53MB）。

        官方 FaceParsing.__init__ 把权重写成了相对路径 './models/face-parse-bisent/*'，
        而服务是从 ~/local-avatar 起的，所以实例化期间临时切一下 cwd。
        只做一次，不影响运行时。
        """
        if getattr(self, "_fp", None) is not None:
            return self._fp
        self._fp = None
        try:
            from musetalk.utils.face_parsing import FaceParsing
        except Exception as e:  # noqa: BLE001
            print(f"[musetalk] 无法导入 face_parsing（缺 mmcv 之外的依赖？）: "
                  f"{type(e).__name__}: {e}", flush=True)
            return None
        # torch 2.6 起 torch.load 默认 weights_only=True，而 BiSeNet 的两份权重
        # （resnet18-5c106cde.pth / 79999_iter.pth）都是 legacy .tar 格式，
        # 直接 load 会抛 "Cannot use weights_only=True with files saved in the
        # legacy .tar format"。官方代码没适配，这里在初始化期间临时放开限定。
        import torch
        orig_load = torch.load

        def _load_trusted(*a, **kw):
            kw.setdefault("weights_only", False)
            return orig_load(*a, **kw)

        cwd = os.getcwd()
        try:
            os.chdir(self.muse_dir)
            torch.load = _load_trusted
            self._fp = FaceParsing()
        except Exception as e:  # noqa: BLE001
            print(f"[musetalk] face-parsing 权重加载失败: {type(e).__name__}: {e}",
                  flush=True)
            self._fp = None
        finally:
            torch.load = orig_load
            os.chdir(cwd)
        if self._fp is not None:
            print("[musetalk] face-parsing (BiSeNet) 已就绪", flush=True)
        return self._fp

    def _seg_parsing(self, pil_img):
        """跑 BiSeNet，返回原始类别图（0~18）。

        不走 FaceParsing.__call__ —— 那个函数只回一张"已经合并好类别"的二值图，
        我们要按需自己挑类别（嘴唇 vs 整张脸），所以直接调它的 net 和预处理。
        类别表（CelebAMask-HQ）：0 背景 1 皮肤 2/3 眉 4/5 眼 6 眼镜 7/8 耳
        10 鼻 11 口腔 12 上唇 13 下唇 14 颈 15 颈饰 16 衣 17 发 18 帽
        """
        from PIL import Image
        torch = self.torch
        fp = self._fp
        img = pil_img.convert("RGB").resize((512, 512), Image.BILINEAR)
        x = fp.preprocess(img).unsqueeze(0)
        x = x.cuda() if torch.cuda.is_available() else x
        with torch.no_grad():
            out = fp.net(x)[0]
        return out.squeeze(0).cpu().numpy().argmax(0)

    def _prepare_lo(self) -> None:
        """预切并预乘「下半区」常量，把每帧的合成运算量砍一半。

        蒙版上半恒为 0（上半脸不参与合成），所以整块 crop 的乘加有一半是白算的。
        这里把 crop 下半切出来、色彩校正图和细节层全部提前合并成常量，
        每帧只剩：resize → 1 次乘 → 1 次加 → 1 次 blend → 1 次加。
        实测合成开销从约 9ms 降到 3ms 以内。
        """
        if not self.split:
            return                       # 整图 decode 模式走全图路径，不预切
        h = self.crop[3] - self.crop[1]
        w = self.crop[2] - self.crop[0]
        self.half = h - int(round(self.split / GEN * h))
        if self.half <= 0:
            return
        self.mask_lo = np.ascontiguousarray(self.mask[self.half:], dtype=np.float32)
        self.gain_lo = np.ascontiguousarray(self.gain_map[self.half:], dtype=np.float32)
        self.bias_lo = np.ascontiguousarray(self.bias_map[self.half:], dtype=np.float32)
        # 细节层 × 权重提前乘好，每帧省一次 124×248×3 的乘法
        self.detail_pack_lo = np.ascontiguousarray(
            self.detail[self.half:] * self.detail_w[self.half:], dtype=np.float32)
        print(f"[musetalk] 合成下半区 {self.half}px/{h}px "
              f"(省去 {100 * (1 - self.half / h):.0f}% 的逐像素运算)", flush=True)

    def _fit_color(self) -> None:
        """用静音帧拟合「生成域 → 原图域」的逐通道线性校正（一次性）。

        整块替换方案的硬伤是色彩：MuseTalk 在 256 latent 里重画下半脸，
        输出相对原图有系统性色偏和对比度塌陷 —— 实测嘴唇从红润变成浅粉、
        皮肤整体发灰。但这是**固定偏差**（同一张脸、同一机位、同一个模型），
        所以可以一次性拟合、之后每帧直接套用，运行时开销只是三次乘加。

        用「均值 + 标准差」两参数对齐，而不是只平移均值：
            fix(x) = (x - mu_gen) * sd_orig/sd_gen + mu_orig
        标准差比把对比度也拉回来，嘴唇的饱和度和口腔阴影深度才恢复。
        统计必须用**静音帧**：说话帧本身的嘴部明暗变化也在统计里，
        会把校正量算偏；静音帧才是"同一张脸、同一姿态"的干净参照。

        而且必须**分区域统计**：融合区里嘴唇只占很小一块，皮肤占大头。
        只做一份参数的话，校正会被皮肤主导，嘴唇依旧发白 —— 这正是第一版
        整块替换看起来"嘴是浅粉色"的原因。所以唇部单独拟合一组，
        最后摊开成逐像素的 gain/bias 图，运行时只是一次乘加。
        """
        if self.idle_low is None or not self.color_fix:
            return
        H, W = self.mask.shape[:2]
        core_256 = cv2.resize(self.core[..., 0], (GEN, GEN),
                              interpolation=cv2.INTER_LINEAR)[self.split:] > 0.5
        mask_256 = cv2.resize(self.mask[..., 0], (GEN, GEN),
                              interpolation=cv2.INTER_LINEAR)[self.split:] > 0.5
        skin_256 = mask_256 & ~core_256
        if skin_256.sum() < 32:
            skin_256 = mask_256
        if core_256.sum() < 32:
            core_256 = mask_256

        def _fit(sel):
            gain = np.ones(3, np.float32)
            bias = np.zeros(3, np.float32)
            for c in range(3):
                g = self.idle_low[..., c][sel]
                b = self.face_low[..., c][sel]
                g_mu, g_sd = float(g.mean()), max(float(g.std()), 1e-3)
                b_mu, b_sd = float(b.mean()), float(b.std())
                gn = min(3.0, max(0.3, b_sd / max(g_sd, 1e-3)))
                gain[c] = gn
                bias[c] = b_mu - g_mu * gn
            return gain, bias

        self.color_gain, self.color_bias = _fit(skin_256)
        self.lip_gain, self.lip_bias = _fit(core_256)
        # 摊开成逐像素图：唇部核心用唇参数，其余用皮肤参数
        cw = self.core.astype(np.float32)
        self.gain_map = (self.color_gain[None, None, :] * (1.0 - cw)
                         + self.lip_gain[None, None, :] * cw).astype(np.float32)
        self.bias_map = (self.color_bias[None, None, :] * (1.0 - cw)
                         + self.lip_bias[None, None, :] * cw).astype(np.float32)
        print(f"[musetalk] 色彩校准 皮肤 gain={np.round(self.color_gain, 2).tolist()} "
              f"唇部 gain={np.round(self.lip_gain, 2).tolist()} "
              f"bias={np.round(self.lip_bias, 1).tolist()}", flush=True)

    @staticmethod
    def _build_alpha(h: int, w: int) -> np.ndarray:
        """residual 回退路径的几何权重：下半脸全用差值，四周羽化消接缝。"""
        a = np.ones((h, w), np.float32)
        top = int(h * 0.46)
        if top > 0:
            ramp = np.linspace(0.0, 1.0, top, dtype=np.float32) ** 1.6
            a[:top, :] *= ramp[:, None]
        fy = max(3, int(h * 0.09))
        fx = max(3, int(w * 0.09))
        ry = np.hanning(fy * 2)[:fy].astype(np.float32)
        rx = np.hanning(fx * 2)[:fx].astype(np.float32)
        ry /= max(ry.max(), 1e-6)
        rx /= max(rx.max(), 1e-6)
        a[:fy, :] *= ry[:, None]
        a[-fy:, :] *= ry[::-1][:, None]
        a[:, :fx] *= rx[None, :]
        a[:, -fx:] *= rx[None, ::-1]
        return a[..., None]

    # ─────────────────────────── 音频 → 特征 ───────────────────────────

    def push_audio(self, pcm: bytes) -> None:
        self.pcm.push_24k(pcm)

    def _encode(self, wav: np.ndarray):
        """一段 16k 音频 → whisper 全部层隐状态 (1, T, L, 384)，T 按 50Hz 截断。"""
        torch = self.torch
        inp = self.fe(wav, return_tensors="pt", sampling_rate=SR_MEL).input_features
        inp = inp.to(self.device, dtype=self.dtype)
        with torch.no_grad():
            hs = self.whisper.encoder(inp, output_hidden_states=True).hidden_states
        h = torch.stack(hs, dim=2)[0]              # (1500, L, 384)
        n = int(len(wav) / SR_MEL * AUDIO_FPS)     # 有效帧数
        return h[:max(0, n)].unsqueeze(0)

    def _ensure_feat(self, from_sec: float, to_sec: float) -> None:
        """保证特征覆盖 [from_sec, to_sec] 这段音频。

        注意必须校验**起点**：早期版本只检查"覆盖到 to_sec"，于是从 to-1.0 开始
        编码，窗口起点落在请求位置之后，_clip 取到负下标后静默补零 ——
        UNet 拿不到音频条件，只能靠 latent 自己重建，输出就是一张糊脸 + 嘴不动。
        这个 bug 很隐蔽，因为画面"看起来还有个人"。
        """
        if self.feat is not None:
            lo_ok = self.feat_base <= int(from_sec * AUDIO_FPS)
            hi_ok = self.feat_base + self.feat.shape[1] >= int(to_sec * AUDIO_FPS)
            # 也不能无限复用：feat 的右端只到编码那一刻的音频末尾，
            # 音频如果和画面同步走，hi_ok 会一直成立，特征就滞后了。
            stale = (from_sec - self.feat_base / AUDIO_FPS) > 1.5
            if lo_ok and hi_ok and not stale:
                return
        avail = self.pcm.seconds
        t0 = max(0.0, from_sec - 1.2)          # 回看一段，减少拼接处跳变
        t1 = min(avail, to_sec + 2.0)          # 能用多少用多少
        # 关键：只要有音频覆盖到请求位置就该编码。线上音频只领先画面 200ms，
        # 若坚持"前瞻 2 秒、窗口至少 1 秒"，这个条件永远不成立 —— 实测
        # feat_refresh=0，特征一次都没算过，_clip 全程补零，画面恒等于静音帧。
        if t1 <= from_sec + 0.02 or t1 - t0 < 0.35:
            return                             # 音频还没到，保持静音帧
        wav = self.pcm.slice(t0, t1)
        if wav.size < SR_MEL // 20:
            return
        self.feat = self._encode(wav)
        self.feat_base = int(t0 * AUDIO_FPS)
        self.feat_refresh += 1

    def _clip(self, idx: int):
        """取第 idx 帧所需的 10 个 whisper 帧，越界补零。"""
        torch = self.torch
        L = self.n_layers
        # 一帧画面覆盖 1/FPS 秒音频，换算成 whisper 的 50Hz 帧轴。
        # 必须用浮点步长：FPS=20 时一帧是 2.5 个音频帧，整除会悄悄丢掉半帧，
        # 时间轴越跑越偏（20 帧后就是 10 帧的偏差 = 200ms 的口型错位）。
        step = float(AUDIO_FPS) / FPS
        f0 = int(round((idx - CTX // 2) * step))
        self._ensure_feat(max(0.0, f0 / AUDIO_FPS), (f0 + CLIP_LEN) / AUDIO_FPS)
        if self.feat is None:
            return torch.zeros((1, CLIP_LEN, L, 384), dtype=self.dtype, device=self.device)
        a = f0 - self.feat_base
        out = torch.zeros((1, CLIP_LEN, L, 384), dtype=self.feat.dtype, device=self.device)
        src_lo = max(0, a)
        src_hi = min(self.feat.shape[1], a + CLIP_LEN)
        if src_hi > src_lo:
            dst_lo = src_lo - a
            out[:, dst_lo:dst_lo + (src_hi - src_lo)] = self.feat[:, src_lo:src_hi]
        return out

    # ─────────────────────────── 推理 ───────────────────────────

    def _infer(self, clip) -> np.ndarray:
        """单步 UNet + VAE 解码 → 256×(256-split) BGR uint8（下半张脸）。"""
        torch = self.torch
        from einops import rearrange
        if not torch.is_tensor(clip):
            clip = torch.as_tensor(clip, device=self.device, dtype=self.dtype)
        x = rearrange(clip, "b c h w -> b (c h) w").to(self.device, dtype=self.dtype)
        x = self.pe(x).to(self.dtype)
        # UNet 必须吃完整的 32×32 latent（它是在全图 latent 上训练的，
        # 喂半张会直接产生严重失真）；省算力只能在 decode 之后裁 ——
        # 只把预测的下半部分交给 VAE，decode 面积减半（256² → 128×256）。
        lat = self.latents.to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            pred = self.unet.model(lat, self.timesteps, encoder_hidden_states=x).sample
        pred_low = (pred[:, :, self.split // 8:, :].contiguous() if self.split else pred)
        img = self.vae.decode_latents(pred_low.to(self.vae.vae.dtype))
        if isinstance(img, np.ndarray) and img.ndim == 4:
            img = img[0]
        return img

    @staticmethod
    def _key_chroma(lo: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """按亮度差异键控色度：亮度没怎么变的地方，色度保持原图。

        模型的色度误差是局部的、非线性的（嘴角发黄、唇缝发绿），
        逐通道 gain/bias 这种全局线性校正压不住它。但换个角度想：
        **说话时真正需要改颜色的地方只有口腔和牙齿** —— 它们伴随剧烈的
        亮度下降/上升。所以用 |ΔY| 当门控：
            |ΔY| 小  → 皮肤、唇面没有实质变化 → 色度 100% 用原图，绝不偏色
            |ΔY| 大  → 张嘴后的暗腔、露出的牙齿 → 放行模型色度，才有立体感
        权重再做一次小半径高斯，避免门控边界出现色块。
        """
        a = cv2.cvtColor(np.clip(lo, 0, 255).astype(np.float32), cv2.COLOR_BGR2YCrCb)
        b = cv2.cvtColor(np.clip(ref, 0, 255).astype(np.float32), cv2.COLOR_BGR2YCrCb)
        dy = np.abs(a[..., 0] - b[..., 0])
        w = np.clip((dy - 6.0) / 18.0, 0.0, 1.0)
        w = cv2.GaussianBlur(w, (0, 0), 2.0)[..., None]
        cc = b[..., 1:] * (1.0 - w) + a[..., 1:] * w
        return cv2.cvtColor(np.concatenate([a[..., :1], cc], axis=2), cv2.COLOR_YCrCb2BGR)

    def _delta_mask(self, delta: np.ndarray) -> np.ndarray:
        """residual 回退路径的自适应蒙版（旧方案，保留仅作对照）。

        按「模型到底改了哪里」生成 0~1 蒙版。缺点是它在空间上跟着差值走，
        和真实嘴部轮廓无关，放大后边缘不可能连续 —— 这正是"贴片感"的来源。
        """
        h, w = delta.shape[:2]
        mag = np.abs(delta).mean(axis=2)
        mag = cv2.GaussianBlur(mag, (0, 0), 3.2)
        peak = float(np.percentile(mag, 99.5))
        if peak < 1.2:                       # 静音/几乎没动嘴 → 整帧保持原图
            return np.zeros((h, w), np.float32)
        m = np.clip((mag - 0.05 * peak) / (self.mask_soft * peak), 0.0, 1.0)
        m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 2.4)
        if self.mask_prior > 0:
            yy = np.arange(h, dtype=np.float32)[:, None]
            xx = np.arange(w, dtype=np.float32)[None, :]
            cy, cx = h * 0.44, w * 0.50
            ry, rx = h * 0.33, w * 0.36
            g = np.exp(-((yy - cy) ** 2 / (ry * ry) + (xx - cx) ** 2 / (rx * rx)) * 1.6)
            m = m * g.astype(np.float32)
        k = min(10, h)
        if k > 1:
            m[:k] *= np.linspace(0.0, 1.0, k, dtype=np.float32)[:, None]
        return m

    def _paste(self, img_low: np.ndarray) -> bytes:
        out = self.base.copy()
        x0, y0, x1, y1 = self.crop
        w, h = x1 - x0, y1 - y0
        roi = out[y0:y1, x0:x1].astype(np.float32)

        if self.compose == "blend" and self.half > 0:
            # 快速路径：蒙版只覆盖下半，直接只算下半 —— 上半一个像素都不碰。
            half = self.half
            sub = img_low
            if sub.shape[0] != GEN - self.split:
                sub = cv2.resize(sub, (GEN, GEN - self.split),
                                 interpolation=cv2.INTER_LANCZOS4)
            gen_lo = cv2.resize(sub.astype(np.float32), (w, half),
                                interpolation=cv2.INTER_LANCZOS4)
            if self.color_fix:
                # 色彩校准：皮肤区和唇部各用各的 gain/bias（预摊成逐像素图）
                gen_lo *= self.gain_lo
                gen_lo += self.bias_lo
            m = self.mask_lo
            lo = gen_lo * m
            lo += roi[half:] * (1.0 - m)
            if self.detail_amt > 0:
                lo += self.detail_pack_lo      # 补回 256 解码丢掉的纹理
            if self.chroma_key:
                lo = self._key_chroma(lo, roi[half:])
            merged = roi.copy()
            merged[half:] = lo
            self.blend_frame += 1

        elif self.compose == "blend":
            # 慢速路径（LOCAL_AVATAR_SPLIT=0 整图 decode 时才会走到）
            full = self.face_256.astype(np.float32)
            if img_low.shape[0] != GEN:
                img_low = cv2.resize(img_low, (GEN, GEN),
                                     interpolation=cv2.INTER_LANCZOS4)
            full = img_low.astype(np.float32)
            gen_big = cv2.resize(full, (w, h), interpolation=cv2.INTER_LANCZOS4)
            if self.color_fix:
                gen_big = gen_big * self.gain_map + self.bias_map
            merged = gen_big * self.mask + roi * (1.0 - self.mask)
            if self.detail_amt > 0:
                merged = merged + self.detail * self.detail_w
            if self.chroma_key:
                merged = self._key_chroma(merged, roi)
            self.blend_frame += 1

        elif self.compose == "direct":
            sub_h = int(round((GEN - self.split) / GEN * h)) if self.split else h
            face = cv2.resize(img_low, (w, sub_h), interpolation=cv2.INTER_LANCZOS4)
            a = self.alpha[h - sub_h:, :, :]
            merged = roi.copy()
            merged[h - sub_h:] = face.astype(np.float32) * a + roi[h - sub_h:] * (1.0 - a)

        else:
            # residual / residual-seg：基准 = 静音帧（没有就退回原图）
            base_low = self.idle_low if self.idle_low is not None else self.face_low
            delta = img_low.astype(np.float32) - base_low
            if self.lowfreq > 0:
                delta = delta - cv2.GaussianBlur(delta, (0, 0), self.lowfreq)
            sub_h = int(round(delta.shape[0] / GEN * h))
            delta_big = cv2.resize(delta, (w, sub_h), interpolation=cv2.INTER_LANCZOS4)
            if self.compose == "residual-seg":
                # 权重取自语义分割（与 blend 同一份 mask，解剖学正确），
                # 像素底仍是高清原图 —— 只把"模型算出来的变化量"按嘴部轮廓叠上去。
                #
                # 但差值里混着两类东西，必须分开处理：
                #   · 高频：唇线位移、口腔边缘、牙齿轮廓 —— 这才是口型，全要
                #   · 低频：模型对两颊/下巴阴影的整体改判，是一大片平滑的偏色，
                #           直接叠上去就是用户看到的"黄斑/贴片感"
                # 所以在这里做一次分频：高通无条件保留，低通只在「嘴唇核心区」
                # 放行（嘴张开时口腔整体变暗就在低频里，不能一起削掉）。
                sig = self.lowfreq if self.lowfreq > 0 else 12.0
                low = cv2.GaussianBlur(delta_big, (0, 0), sig)
                gate_core = cv2.resize(self.core[..., 0], (w, sub_h),
                                       interpolation=cv2.INTER_LINEAR)[..., None]
                d = (delta_big - low) + low * gate_core
                a = self.mask[h - sub_h:, :, :]
                merged = roi.copy()
                merged[h - sub_h:] = roi[h - sub_h:] + d * a
            else:
                mask = cv2.resize(self._delta_mask(delta), (w, sub_h),
                                  interpolation=cv2.INTER_LINEAR)[..., None]
                a = mask * self.alpha[h - sub_h:, :, :]
                merged = roi.copy()
                merged[h - sub_h:] = roi[h - sub_h:] + delta_big * (a * self.gain)

        # 轻微 USM：VAE 解码偏平滑，补一点唇部锐度（LOCAL_AVATAR_SHARPEN 可调）。
        # blend 模式已经用原图细节层补过高频，再做 USM 会过锐出白边，跳过。
        if self.sharpen > 0 and self.compose != "blend":
            blur = cv2.GaussianBlur(merged, (0, 0), 1.2)
            merged = merged + self.sharpen * 0.5 * (merged - blur)
        out[y0:y1, x0:x1] = np.clip(merged, 0, 255).astype(np.uint8)
        if self.watermark:
            cv2.putText(out, f"LOCAL MUSETALK {self.version.upper()}", (20, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (90, 220, 255), 2, cv2.LINE_AA)
        return self._overlay(out)

    def render_frame(self) -> bytes:
        lead = 0.50                              # 安全余量：音频领先画面 500ms
        target = int((self.pcm.seconds - lead) * FPS)
        if target < 0:
            target = 0
        # 落后太多（TTS 爆发式吐字）就直接跳到最新帧，宁可跳帧也不能音画错位
        if target - self.cursor > 6:
            self.skipped += target - self.cursor - 1
            self.cursor = target - 1
        if self.cursor > target:
            self.cursor = target                 # 没有新音频 → 原地不动
            return self._idle_frame()

        t0 = time.monotonic()
        with self.lock_render:
            img = self._infer(self._clip(self.cursor))
        self.gen_ms = 0.9 * self.gen_ms + 0.1 * (time.monotonic() - t0) * 1000.0
        self.cursor += 1
        return self._paste(img)

    def _idle_frame(self) -> bytes:
        """没有新音频时直接出原图 —— 嘴是闭合的，也避开 VAE 重建的色差。"""
        return self._overlay(self.base)

    def stats(self) -> dict:
        return {"gen_ms": round(self.gen_ms, 1), "cursor": self.cursor,
                "audio_s": round(self.pcm.seconds, 2), "skipped": self.skipped,
                "feat_refresh": self.feat_refresh, "compose": self.compose,
                "version": self.version}
