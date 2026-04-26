"""Application configuration for Milvus + LangChain terminal demo."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


SYSTEM_PROMPT = """你是「小咪」，一个活泼可爱、反应超快的智能助手！
说话简洁不废话，直接给出最有用的答案。
回答控制在3-5句话以内，除非用户明确要求详细说明。
语气亲切自然，偶尔用一个emoji点缀，但不过分。
如果检索到的数据里有相关信息，直接引用；没有就坦率说不知道。"""


@dataclass(frozen=True)
class AppConfig:
    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_backend: str = "auto"
    milvus_uri: str = "./milvus_data/demo.db"
    collection_name: str = "people_profiles"
    top_k: int = 5
    data_count: int = 500
    dataset_json_path: str = "./dataset/profiles_500.json"
    session_history_dir: str = "./chat_history"
    batch_size: int = 100
    max_memory_turns: int = 10
    request_timeout: int = 60
    max_retries: int = 3
    hf_endpoint: str = "https://hf-mirror.com"
    hf_download_timeout: int = 180

    @property
    def milvus_path(self) -> Path:
        return Path(self.milvus_uri).resolve()

    @property
    def history_path(self) -> Path:
        return Path(self.session_history_dir).resolve()


def load_config() -> AppConfig:
    api_key = (os.getenv("DEEPSEEK_API_KEY", "") or "").strip()
    if not api_key:
        raise ValueError("缺少 DEEPSEEK_API_KEY，请在 .env 或环境变量中配置。")
    embedding_backend = (os.getenv("EMBEDDING_BACKEND", "auto") or "auto").strip().lower()
    if embedding_backend not in {"auto", "hf", "local_hash"}:
        raise ValueError("EMBEDDING_BACKEND 仅支持 auto / hf / local_hash。")
    return AppConfig(
        deepseek_api_key=api_key,
        embedding_backend=embedding_backend,
        hf_endpoint=os.getenv("HF_ENDPOINT", "https://hf-mirror.com"),
        hf_download_timeout=int(os.getenv("HF_HUB_DOWNLOAD_TIMEOUT", "180")),
    )
