from __future__ import annotations

import os
from dataclasses import dataclass

from pathlib import Path
from dotenv import load_dotenv


def _load_env() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    config_env = base_dir / "config" / ".env"
    root_env = base_dir / ".env"
    if config_env.exists():
        load_dotenv(dotenv_path=config_env)
    if root_env.exists():
        load_dotenv(dotenv_path=root_env)


_load_env()


@dataclass(frozen=True)
class RedisConfig:
    redis_url: str
    application_id: str
    agent_id: str
    user_id: str
    thread_id: str


@dataclass(frozen=True)
class LlmConfig:
    provider: str
    foundry_endpoint: str
    foundry_model: str
    azure_openai_endpoint: str
    azure_openai_deployment: str
    azure_openai_api_key: str


@dataclass(frozen=True)
class BingConfig:
    api_key: str
    endpoint: str


@dataclass(frozen=True)
class AzureSearchConfig:
    endpoint: str
    api_key: str
    index_name: str
    api_version: str


@dataclass(frozen=True)
class RagConfig:
    source: str


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str
    model: str
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_api_version: str


def _build_azure_redis_url(host: str, access_key: str, port: int) -> str:
    return f"rediss://:{access_key}@{host}:{port}"


def load_redis_config() -> RedisConfig:
    redis_url = os.getenv("REDIS_URL")
    azure_host = os.getenv("AZURE_REDIS_HOST")
    azure_key = os.getenv("AZURE_REDIS_ACCESS_KEY")
    azure_port = int(os.getenv("AZURE_REDIS_PORT", "10000"))

    if not redis_url and azure_host and azure_key:
        redis_url = _build_azure_redis_url(azure_host, azure_key, azure_port)

    if not redis_url:
        redis_url = "redis://localhost:6379"

    return RedisConfig(
        redis_url=redis_url,
        application_id=os.getenv("APPLICATION_ID", "reflective_memory_diffing"),
        agent_id=os.getenv("AGENT_ID", "memory_diff_agent"),
        user_id=os.getenv("USER_ID", "local_user"),
        thread_id=os.getenv("THREAD_ID", "memory_diff_thread"),
    )


def load_llm_config() -> LlmConfig:
    foundry_endpoint = os.getenv("FOUNDRY_PROJECT_ENDPOINT", "").strip()
    foundry_model = os.getenv("FOUNDRY_MODEL_DEPLOYMENT", "gpt-4o-mini")

    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()

    if azure_openai_endpoint:
        provider = "azure_openai"
    else:
        provider = "foundry"

    return LlmConfig(
        provider=provider,
        foundry_endpoint=foundry_endpoint,
        foundry_model=foundry_model,
        azure_openai_endpoint=azure_openai_endpoint,
        azure_openai_deployment=azure_openai_deployment,
        azure_openai_api_key=azure_openai_api_key,
    )


def load_bing_config() -> BingConfig:
    api_key = os.getenv("BING_SEARCH_API_KEY", "").strip()
    endpoint = os.getenv("BING_SEARCH_ENDPOINT", "https://api.bing.microsoft.com/v7.0/search").strip()
    return BingConfig(api_key=api_key, endpoint=endpoint)


def load_azure_search_config() -> AzureSearchConfig:
    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT", "").strip()
    api_key = os.getenv("AZURE_SEARCH_API_KEY", "").strip()
    index_name = os.getenv("AZURE_SEARCH_INDEX", "").strip()
    api_version = os.getenv("AZURE_SEARCH_API_VERSION", "2024-07-01").strip()
    return AzureSearchConfig(
        endpoint=endpoint,
        api_key=api_key,
        index_name=index_name,
        api_version=api_version,
    )


def load_rag_config() -> RagConfig:
    source = os.getenv("RAG_SOURCE", "bing").strip().lower()
    return RagConfig(source=source)


def load_embedding_config() -> EmbeddingConfig:
    provider = os.getenv("EMBEDDING_PROVIDER", "none").strip().lower()
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip()
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21").strip()
    return EmbeddingConfig(
        provider=provider,
        model=model,
        azure_openai_endpoint=azure_openai_endpoint,
        azure_openai_api_key=azure_openai_api_key,
        azure_openai_api_version=azure_openai_api_version,
    )
