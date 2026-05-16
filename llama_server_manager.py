"""
llama_server_manager.py
管理 llama-server 子进程的启动与关闭，支持多模型切换与 mmproj（视觉模型投影器）。
"""
import platform
import sys
import signal
import subprocess
import time
import atexit
import requests
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

# ──────────────────────────────────────────────
# 全局状态
# ──────────────────────────────────────────────
_process: Optional[subprocess.Popen] = None
_current_config: Optional["ModelConfig"] = None

LLAMA_SERVER_PORT = 1241
LLAMA_SERVER_URL = f"http://127.0.0.1:{LLAMA_SERVER_PORT}"


# ──────────────────────────────────────────────
# 模型配置数据类
# ──────────────────────────────────────────────
@dataclass
class ModelConfig:
    """
    描述一次 llama-server 启动所需的全部参数。

    Attributes:
        model_path:    主 GGUF 模型文件路径（必填）
        mmproj_path:   视觉投影器 GGUF 路径（LLaVA / MiniCPM-V 等需要，可选）
        n_gpu_layers:  上 GPU 的层数，-1 表示全部
        ctx_size:      上下文长度（token 数）
        flash_attn:    Flash Attention 模式（"auto" / "on" / "off"）
        cache_type_k:  KV cache K 量化类型
        cache_type_v:  KV cache V 量化类型
        ubatch_size:   micro-batch 大小
        threads:       CPU 线程数
        parallel:      并发槽位数
        reasoning:     是否开启推理链（0=关闭）
        extra_args:    追加到命令行末尾的额外参数列表
    """
    model_path: str
    mmproj_path: Optional[str] = None
    n_gpu_layers: int = -1
    ctx_size: int = 1024
    flash_attn: str = "auto"
    cache_type_k: str = "q8_0"
    cache_type_v: str = "q8_0"
    ubatch_size: int = 128
    threads: int = 4
    parallel: int = 1
    reasoning: int = 0
    extra_args: list = field(default_factory=list)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModelConfig):
            return False
        return (
            Path(self.model_path).resolve() == Path(other.model_path).resolve()
            and (
                (self.mmproj_path is None and other.mmproj_path is None)
                or (
                    self.mmproj_path is not None
                    and other.mmproj_path is not None
                    and Path(self.mmproj_path).resolve()
                    == Path(other.mmproj_path).resolve()
                )
            )
            and self.n_gpu_layers == other.n_gpu_layers
            and self.ctx_size == other.ctx_size
            and self.flash_attn == other.flash_attn
            and self.cache_type_k == other.cache_type_k
            and self.cache_type_v == other.cache_type_v
            and self.ubatch_size == other.ubatch_size
            and self.threads == other.threads
            and self.parallel == other.parallel
            and self.reasoning == other.reasoning
            and self.extra_args == other.extra_args
        )


# ──────────────────────────────────────────────
# 内部工具
# ──────────────────────────────────────────────
def _get_server_exe() -> str:
    """获取 llama-server 可执行文件路径"""
    if getattr(sys, "frozen", False):
        base = Path(sys._MEIPASS)
    else:
        base = Path(__file__).parent
    if platform.system() == "Windows":
        exe = base / "tools" / "llama-cpp-win" / "llama-server.exe"
    elif platform.system() == "Darwin":
        exe = base / "tools" / "llama-cpp-mac" / "llama-server"
    else:
        exe = base / "tools" / "llama-cpp-linux" / "llama-server"
    if not exe.exists():
        raise FileNotFoundError(f"llama-server 未找到: {exe}")
    return str(exe)


def _build_cmd(exe: str, cfg: ModelConfig) -> list:
    """根据 ModelConfig 构建完整的命令行参数列表"""
    cmd = [
        exe,
        "--model", cfg.model_path,
        "--port", str(LLAMA_SERVER_PORT),
        "--host", "127.0.0.1",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "--ctx-size", "4096",
        "--reasoning", "0",
    ]
    # 视觉模型：附加 mmproj 投影器路径
    if cfg.mmproj_path:
        cmd += ["--mmproj", cfg.mmproj_path]
    # 用户自定义额外参数
    cmd += cfg.extra_args
    return cmd


def _wait_until_ready(timeout: int = 90) -> bool:
    """轮询 /health 接口直到服务就绪或超时"""
    global _process
    for _ in range(timeout):
        if _process is None or _process.poll() is not None:
            print("[llama_server] 进程意外退出")
            return False
        try:
            r = requests.get(f"{LLAMA_SERVER_URL}/health", timeout=1)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


# ──────────────────────────────────────────────
# 公开 API
# ──────────────────────────────────────────────
def stop_server() -> None:
    """强制关闭 llama-server 子进程及其进程树"""
    global _process, _current_config
    if _process is None:
        return

    pid = _process.pid
    print("[llama_server] 正在关闭...")

    if platform.system() == "Windows":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        try:
            _process.terminate()
            _process.wait(timeout=5)
        except Exception:
            _process.kill()

    _process = None
    _current_config = None
    print("[llama_server] 已关闭")


def start_server(
    model_path: str,
    mmproj_path: Optional[str] = None,
    *,
    reasoning: int = 0,
    extra_args: Optional[list] = None,
) -> bool:
    """
    启动（或切换）llama-server 子进程。

    若请求的模型配置与当前运行的完全一致，则直接复用，不重启。
    若不一致（包括切换了模型或 mmproj），先关闭旧进程再启动新进程。

    :param model_path:   主 GGUF 模型文件路径
    :param mmproj_path:  视觉投影器路径（LLaVA/MiniCPM-V 等视觉模型需要）
    :param n_gpu_layers: 上 GPU 的层数，-1 = 全部
    :param ctx_size:     上下文长度
    :param flash_attn:   Flash Attention 模式
    :param cache_type_k: KV-cache K 量化类型
    :param cache_type_v: KV-cache V 量化类型
    :param ubatch_size:  micro-batch 大小
    :param threads:      CPU 线程数
    :param parallel:     并发槽位数
    :param reasoning:    推理链开关（0=关闭）
    :param extra_args:   追加到命令行的额外参数列表
    :return: 启动成功返回 True，否则 False
    """
    global _process, _current_config

    new_cfg = ModelConfig(
        model_path=str(Path(model_path).resolve()),
        mmproj_path=str(Path(mmproj_path).resolve()) if mmproj_path else None,
        reasoning=reasoning,
        extra_args=extra_args or [],
    )

    # ── 验证文件存在 ──────────────────────────
    if not Path(new_cfg.model_path).is_file():
        print(f"[llama_server] 错误: 模型文件不存在: {new_cfg.model_path}")
        return False
    if new_cfg.mmproj_path and not Path(new_cfg.mmproj_path).is_file():
        print(f"[llama_server] 错误: mmproj 文件不存在: {new_cfg.mmproj_path}")
        return False

    # ── 判断是否可以复用 ──────────────────────
    if (
        _current_config is not None
        and new_cfg == _current_config
        and _process is not None
        and _process.poll() is None
    ):
        print("[llama_server] 模型配置未变，复用现有进程")
        return True

    # ── 切换模型：先停止旧进程 ────────────────
    if _process is not None and _process.poll() is None:
        print(f"[llama_server] 切换模型 → {Path(new_cfg.model_path).name}")
        stop_server()

    # ── 获取可执行文件 ────────────────────────
    try:
        exe = _get_server_exe()
    except FileNotFoundError as e:
        print(f"[llama_server] 错误: {e}")
        return False

    cmd = _build_cmd(exe, new_cfg)

    mmproj_info = f", mmproj: {Path(new_cfg.mmproj_path).name}" if new_cfg.mmproj_path else ""
    print(f"[llama_server] 启动中... 模型: {Path(new_cfg.model_path).name}{mmproj_info}")

    # ── 启动子进程 ────────────────────────────
    popen_kwargs = dict(
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if platform.system() == "Windows":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    try:
        _process = subprocess.Popen(cmd, **popen_kwargs)
        _current_config = new_cfg
    except Exception as e:
        print(f"[llama_server] 启动失败: {e}")
        return False

    # ── 等待服务就绪 ──────────────────────────
    if _wait_until_ready(timeout=90):
        print(f"[llama_server] 就绪，端口 {LLAMA_SERVER_PORT}")
        return True
    else:
        print("[llama_server] 启动超时")
        stop_server()
        return False


def switch_model(
    model_path: str,
    mmproj_path: Optional[str] = None,
    **kwargs,
) -> bool:
    """
    便捷函数：切换到另一个模型（内部调用 start_server）。
    相同配置时不会重启，不同配置时先停止再启动。

    :param model_path:  新的 GGUF 主模型路径
    :param mmproj_path: 新的 mmproj 投影器路径（可选）
    :param kwargs:      其余参数透传给 start_server
    :return: 切换成功返回 True
    """
    return start_server(model_path, mmproj_path, **kwargs)


def get_current_model() -> Optional[str]:
    """返回当前已加载的主模型路径，未运行时返回 None"""
    if _current_config is not None and is_running():
        return _current_config.model_path
    return None


def get_current_mmproj() -> Optional[str]:
    """返回当前已加载的 mmproj 路径，未使用时返回 None"""
    if _current_config is not None and is_running():
        return _current_config.mmproj_path
    return None


def is_running() -> bool:
    """检查 llama-server 是否正在运行"""
    return _process is not None and _process.poll() is None


# ──────────────────────────────────────────────
# 进程退出钩子
# ──────────────────────────────────────────────
def _on_exit(sig, frame):
    """Ctrl+C / SIGTERM 信号处理"""
    stop_server()
    sys.exit(0)


atexit.register(stop_server)
signal.signal(signal.SIGINT, _on_exit)
signal.signal(signal.SIGTERM, _on_exit)
