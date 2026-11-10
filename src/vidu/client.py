"""Vidu 组件版(s_avatar/component) HTTP 客户端 —— 全异步(httpx.AsyncClient)。

端点(来自官方文档《详细参数》+ 实时版项目实测的端点表):
- POST /live/s_avatar/component              创建组件版会话(核心入口)
- GET  /live/v1/lives/{live_id}              查询直播状态
- GET  /live/v1/lives/{live_id}/recording    查询录播(managed_rtc.enable_recording=true 时)
- POST /tools/v2/files/uploads               三步式上传第 1 步(拿 put_url)
- PUT  {put_url}                             三步式上传第 2 步(直传对象存储, 不带鉴权头)
- PUT  /tools/v2/files/uploads/{id}/finish   三步式上传第 3 步(etag → ssupload 短链)
- /live/v1/action-libraries(+/{id}/actions)  动作库管理

鉴权: Authorization: Token vda_xxx(除第 2 步 PUT 外都要)。
"""
from __future__ import annotations

import base64
import pathlib

import httpx

try:
    import certifi
    _VERIFY = certifi.where()
except Exception:
    _VERIFY = True

CONF: dict = {}
_client: httpx.AsyncClient | None = None


def configure(vidu_conf: dict) -> None:
    global CONF, _client
    CONF = vidu_conf or {}
    _client = None


def _http() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(verify=_VERIFY, timeout=httpx.Timeout(30.0))
    return _client


def _headers() -> dict:
    # 本地渲染 / 未填 key 时不要发 "Token " —— httpx 会判定为非法 header 值
    h = {"Content-Type": "application/json"}
    key = str(CONF.get("api_key") or "").strip()
    if key and str(CONF.get("provider") or "").lower() != "local":
        h["Authorization"] = f"Token {key}"
    return h


def _url(path: str) -> str:
    """host 带 scheme 时走本地渲染服务(如 http://<GPU机IP>:9880)，否则走 Vidu 云。

    这样云端 / 本地两套后端共用同一套调用代码，engine 层无需分支。
    """
    host = str(CONF.get("host", "api.vidu.cn")).rstrip("/")
    if "://" in host:
        return f"{host}{path}"
    return f"https://{host}{path}"


async def close() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


async def _check(r: httpx.Response) -> dict:
    if r.status_code >= 400:
        # 保留响应体: 400 的 {"code":..., "message":...} 才看得懂被谁拒了
        raise RuntimeError(f"Vidu HTTP {r.status_code}: {r.text[:500]}")
    return r.json()


async def create_component(body: dict, timeout: float = 120) -> dict:
    r = await _http().post(_url(CONF.get("create_path", "/live/s_avatar/component")),
                           headers=_headers(), json=body, timeout=timeout)
    return await _check(r)


async def get_live(live_id: str, timeout: float = 8) -> dict:
    r = await _http().get(_url(f"/live/v1/lives/{live_id}"), headers=_headers(), timeout=timeout)
    return await _check(r)


async def get_recording(live_id: str, timeout: float = 15) -> dict:
    r = await _http().get(_url(f"/live/v1/lives/{live_id}/recording"),
                          headers=_headers(), timeout=timeout)
    return await _check(r)


async def list_lives(page: int = 0, size: int = 20, timeout: float = 8) -> dict:
    """账号下的历史会话清单(含组件版场次): live_id/status/时间等。"""
    r = await _http().get(_url("/live/v1/lives"), headers=_headers(),
                          params={"pager.page": page, "pager.pagesz": size},
                          timeout=timeout)
    return await _check(r)


async def credits(timeout: float = 8) -> dict:
    r = await _http().get(_url("/ent/v2/credits"), headers=_headers(),
                          params={"show_detail": "false"}, timeout=timeout)
    return await _check(r)


# ---------- 三步式上传(图片 → ssupload 短链, 建会话 image_uri 用) ----------

async def upload_bytes(content: bytes, content_type: str) -> str:
    sess = await _check(await _http().post(_url("/tools/v2/files/uploads"),
                                           headers=_headers(),
                                           json={"scene": "vidu"}, timeout=60))
    fid, put_url = sess["id"], sess["put_url"]
    r = await _http().put(put_url, content=content, headers={"Content-Type": content_type},
                          timeout=300)
    r.raise_for_status()
    etag = r.headers.get("etag", "").strip('"')
    if not etag:
        raise RuntimeError("PUT 成功但响应无 etag")
    fin = await _http().put(_url(f"/tools/v2/files/uploads/{fid}/finish"),
                            headers=_headers(), json={"etag": etag}, timeout=60)
    if fin.status_code >= 400:
        raise RuntimeError(f"finish {fin.status_code}: {fin.text[:300]}")
    return fin.json().get("uri") or f"ssupload:?id={fid}"


async def upload_file(path: str, content_type: str = "") -> str:
    p = pathlib.Path(path)
    ctype = content_type or {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(p.suffix.lower(), "application/octet-stream")
    return await upload_bytes(p.read_bytes(), ctype)


def b64_data_uri(data: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode()


# ---------- 形象资产库(实时版同款端点 /live/v1/avatars) ----------

async def list_avatars(page: int = 0, size: int = 100, timeout: float = 8) -> dict:
    r = await _http().get(_url("/live/v1/avatars"), headers=_headers(),
                          params={"pager.page": page, "pager.pagesz": size},
                          timeout=timeout)
    return await _check(r)


async def create_avatar(image_uri: str, name: str = "", timeout: float = 300) -> dict:
    """图片(ssupload 短链) → 数字人资产。S2: 任意单人图即可; 图片建的资产平台 90 天后自动删。"""
    body: dict = {"image_uri": image_uri}
    if name:
        body["name"] = name
    r = await _http().post(_url("/live/v1/avatars"), headers=_headers(),
                           json=body, timeout=timeout)
    return await _check(r)


async def delete_avatar(avatar_id: str, timeout: float = 30) -> dict:
    r = await _http().delete(_url(f"/live/v1/avatars/{avatar_id}"),
                             headers=_headers(), timeout=timeout)
    return await _check(r)


# ---------- 动作库(建会话 action.library_id 引用) ----------

async def list_action_libraries(page: int = 0, size: int = 100) -> dict:
    r = await _http().get(_url("/live/v1/action-libraries"), headers=_headers(),
                          params={"pager.page": page, "pager.pagesz": size}, timeout=8)
    return await _check(r)


async def create_action_library(name: str, description: str = "") -> dict:
    body = {"name": name}
    if description:
        body["description"] = description
    r = await _http().post(_url("/live/v1/action-libraries"), headers=_headers(),
                           json=body, timeout=30)
    return await _check(r)


async def delete_action_library(lib_id: str) -> dict:
    r = await _http().delete(_url(f"/live/v1/action-libraries/{lib_id}"),
                             headers=_headers(), timeout=30)
    return await _check(r)


async def list_actions(lib_id: str, page: int = 0, size: int = 100) -> dict:
    r = await _http().get(_url(f"/live/v1/action-libraries/{lib_id}/actions"),
                          headers=_headers(),
                          params={"pager.page": page, "pager.pagesz": size}, timeout=8)
    return await _check(r)


async def create_action(lib_id: str, name: str, description: str) -> dict:
    """加动作: name 1..20 字, description 5..500 字(服务端会先改写再入库, 数秒耗时)。"""
    r = await _http().post(_url(f"/live/v1/action-libraries/{lib_id}/actions"),
                           headers=_headers(),
                           json={"name": name, "description": description}, timeout=120)
    return await _check(r)


async def delete_action(lib_id: str, action_id: str) -> dict:
    r = await _http().delete(_url(f"/live/v1/action-libraries/{lib_id}/actions/{action_id}"),
                             headers=_headers(), timeout=30)
    return await _check(r)
