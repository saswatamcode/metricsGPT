import argparse
import re
import urllib.parse
import sys
import yaml
import os
import json
import hashlib
import logging
import traceback
import uvicorn
import asyncio
import aioconsole

from colorama import Fore, init
import prometheus_api_client

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.llms import ChatMessage
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.gemini import Gemini
from llama_index.llms.ibm.base import WatsonxLLM
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.embeddings.ibm import WatsonxEmbeddings


# TODO(@saswatamcode): Separate prompts based on the LLM?
PROMPT_TEMPLATE_CHAT = """First, explain what the query does and how it helps answer the question. Think of yourself as a PromQL Expert SRE.
Then, on a new line, provide just the PromQL query between <PROMQL> and </PROMQL> tags.

Ensure that,
- The PromQL query is valid PromQL and will not cause errors and can actually run.
- The PromQL query is URL encodable.
- The PromQL query takes into account the upstream and open source best practices and norms for Prometheus.
- The PromQL query make reasonable assumptions from the query and the metrics provided as well as their nomenclature.
- Ensure that your final PromQL query has balanced brackets and balanced double quotes(when dealing with label selectors)

Format your response like this:
Your explanation of what the query does and how it helps...

<PROMQL>your_query_here</PROMQL>


Here is some more information below,
        
Assume that the following is a list of metrics that are available to query within the TSDB (but there can be more). Take this into context when designing the query:
{data}

Below is an excerpt of the recent conversation. Understand it, and if the user is asking follow-up questions,
edit your response accordinly, but do not go beyond the format:
{chat_history}

And finally here is the user's actual question: {prompt}
"""

PROMPT_TEMPLATE_API = """First, explain what the query does and how it helps answer the question. Think of yourself as a PromQL Expert SRE.
Then, on a new line, provide just the PromQL query between <PROMQL> and </PROMQL> tags.

Ensure that,
- The PromQL query is valid PromQL and will not cause errors and can actually run.
- The PromQL query is URL encodable.
- The PromQL query takes into account the upstream and open source best practices and norms for Prometheus.
- The PromQL query make reasonable assumptions from the query and the metrics provided as well as their nomenclature.
- Ensure that your final PromQL query has balanced brackets and balanced double quotes(when dealing with label selectors)

Format your response like this:
Your explanation of what the query does and how it helps...

<PROMQL>your_query_here</PROMQL>

Here is some more information below,
        
Assume that the following is a list of metrics that are available to query within the TSDB (but there can be more). Take this into context when designing the query:
{data}

And finally here is the user's actual question: {prompt}
"""

class LogfmtFormatter(logging.Formatter):
    """
    A formatter to emit log messages in logfmt format, including labels and stack traces.
    """

    DEFAULT_FORMAT = (
        "level=%(levelname)s time=%(asctime)s component=%(name)s message=%(message)s"
    )

    def __init__(self, *args, **kwargs):
        """
        Initialize the formatter with an optional default component label.
        """
        super().__init__(*args, **kwargs)

    def format(self, record):
        log_data = {
            "level": record.levelname,
            "time": self.formatTime(record, self.datefmt),
            "component": record.name,
            "message": record.getMessage(),
        }

        if hasattr(record, "extra"):
            log_data.update(record.extra)

        if record.exc_info:
            stack_trace = traceback.format_exception(*record.exc_info)
            log_data["stack_trace"] = "".join(stack_trace).strip()

        return " ".join(
            f"{key}={json.dumps(value)}"
            for key, value in log_data.items()
            if value is not None
        )

    @staticmethod
    def setup_logging(level=logging.INFO):
        """Configure basic logging with LogfmtFormatter."""
        logging.basicConfig(
            level=level,
            handlers=[logging.StreamHandler()],
            format=LogfmtFormatter.DEFAULT_FORMAT,
        )


def create_logger(component_name: str):
    logger = logging.getLogger(component_name)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    formatter = LogfmtFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


class MetricsCache:
    """MetricsCache is a manager for the prometheus metrics cache file.
    You can load it as a cache handler for PrometheusClient."""

    def __init__(self, cache_file: str, logger: logging.Logger):
        self.cache_file = cache_file
        self.logger = logger

    def load(self) -> list:
        """Load cached time series from file."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "r") as f:
                    return json.load(f)
        except Exception as err:
            self.logger.error(
                f"Error loading cache", extra={"error": err}, exc_info=True
            )
        return []

    def save(self, timeseries: list) -> None:
        """Save time series to cache file."""
        try:
            with open(self.cache_file, "w") as f:
                json.dump(timeseries, f)
        except Exception as err:
            self.logger.error(
                f"Error saving cache", extra={"error": err}, exc_info=True
            )

    def append(self, series: dict) -> None:
        """Append a series to the cache if not already present."""
        cache = self.load()
        if series not in cache:
            cache.append(series)
            self.save(cache)


class PrometheusClient:
    """PrometheusClient is a wrapper around the Prometheus API client with some custom methods not available in upstream client."""

    def __init__(self, url: str, logger: logging.Logger, auth_config: dict = None):
        client_kwargs = {
            'url': url,
            'disable_ssl': True
        }
        
        # Add authentication configuration if provided
        if auth_config:
            if auth_config.get('basic_auth'):
                client_kwargs['auth'] = (
                    auth_config['basic_auth'].get('username'),
                    auth_config['basic_auth'].get('password')
                )
            if auth_config.get('bearer_token'):
                client_kwargs['headers'] = {
                    'Authorization': f"Bearer {auth_config['bearer_token']}"
                }
            if auth_config.get('custom_headers'):
                client_kwargs['headers'] = auth_config['custom_headers']
            if auth_config.get('tls'):
                client_kwargs.update({
                    'disable_ssl': False,
                    'verify': not auth_config['tls'].get('skip_verify', False),
                    'cert': (
                        auth_config['tls'].get('cert_file'),
                        auth_config['tls'].get('key_file')
                    ) if auth_config['tls'].get('cert_file') else None
                })

        self.client = prometheus_api_client.PrometheusConnect(**client_kwargs)
        self.logger = logger

    async def get_series(self, params: dict = None) -> list:
        """get_series calls /api/v1/series on a Prometheus-compatible API server."""
        params = params or {}
        response = self.client._session.get(
            f"{self.client.url}/api/v1/series",
            verify=self.client._session.verify,
            headers=self.client.headers,
            params=params,
            auth=self.client.auth,
            cert=self.client._session.cert,
        )
        if response.status_code == 200:
            return response.json()["data"]
        raise prometheus_api_client.PrometheusApiClientException(
            f"/api/v1/series PrometheusApiClientException: HTTP Status Code {response.status_code} ({response.content})"
        )

    async def get_all_series(self, cache_handler: MetricsCache) -> None:
        """Fetches all metrics and their series from Prometheus."""
        try:
            self.logger.info("Fetching all metrics")
            for metric in self.client.all_metrics():
                series_list = await self.get_series(params={"match[]": metric})
                for series in series_list:
                    cache_handler.append(series)
        except prometheus_api_client.PrometheusApiClientException as err:
            self.logger.error(
                f"Error getting series", extra={"error": err}, exc_info=True
            )


class VectorStoreManager:
    """VectorStoreManager is a manager for the vector store."""

    def __init__(self, vectordb_path: str, embed_model, logger: logging.Logger):
        self.vectordb_path = vectordb_path
        self.embed_model = embed_model
        self.vector_store = None
        self.index = None
        self.lock = asyncio.Lock()
        self.logger = logger

    async def initialize(self, cache: list) -> None:
        """Initialize or load the vector store."""
        self.logger.info("Initializing vector store")
        if not os.path.exists(self.vectordb_path):
            await self._create_new_index(cache)
        else:
            await self._load_existing_index()

    async def _create_new_index(self, cache: list) -> None:
        self.logger.info("Creating new index")
        self.vector_store = MilvusVectorStore(
            uri=self.vectordb_path, dim=768, overwrite=True, collection_name="metrics"
        )
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        documents = self._create_documents(cache)
        async with self.lock:
            self.index = VectorStoreIndex.from_documents(
                documents, storage_context=storage_context, embed_model=self.embed_model
            )
            self.logger.info("New index created successfully")

    async def refresh_embeddings(self, cache: list) -> None:
        """Refresh the vector store with updated metrics."""
        self.logger.info("Refreshing vector store embeddings...")
        await self._create_new_index(cache)

    async def _load_existing_index(self) -> None:
        self.logger.info("Loading existing index from disk")
        self.vector_store = MilvusVectorStore(
            uri=self.vectordb_path, dim=768, overwrite=False, collection_name="metrics"
        )
        async with self.lock:
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store, embed_model=self.embed_model
            )
            self.logger.info("Existing index loaded successfully")

    @staticmethod
    def _create_documents(cache: list) -> list:
        documents = []
        for metric in cache:
            metric_str = f"{metric['__name__']}{{{', '.join(f'{k}={v}' for k, v in metric.items() if k != '__name__')}}}"
            documents.append(Document(text=metric_str, doc_id=hash_metric(metric)))
        return documents


def hash_metric(d: dict) -> str:
    return hashlib.sha256(str(json.dumps(d, sort_keys=True)).encode()).hexdigest()


def extract_promql(text):
    """extract_promql extracts all PROMQL code blocks from the LLM response."""
    pattern = re.compile(r"<PROMQL>(.*?)</PROMQL>", re.DOTALL)
    matches = pattern.findall(text)

    return matches


class MetricsGPTServer:
    """MetricsGPTServer is the main server class for metricsGPT."""

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        prometheus: PrometheusClient,
        metrics_cache: MetricsCache,
        llm,
        embed_model,
        prometheus_url,
        prom_external_url,
        query_lookback_hours,
        refresh_interval: int,
        logger: logging.Logger,
    ):
        self.vector_store_manager = vector_store_manager
        self.prometheus = prometheus
        self.metrics_cache = metrics_cache
        self.llm = llm
        self.embed_model = embed_model
        self.prometheus_url = prometheus_url
        self.prom_external_url = prom_external_url
        self.query_lookback_hours = query_lookback_hours
        self.logger = logger
        self.fastapi_app = FastAPI(lifespan=self.lifespan)
        self.fastapi_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        if getattr(sys, "frozen", False):
            self.build_dir = os.path.join(sys._MEIPASS, "ui/build")
        else:
            self.build_dir = os.path.join(os.path.dirname(__file__), "ui", "build")

        self.setup_routes()
        self.shutdown_event = asyncio.Event()
        self.refresh_interval = refresh_interval

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        refresh_task = asyncio.create_task(self.refresh_data_server())
        try:
            yield
        finally:
            self.shutdown_event.set()
            refresh_task.cancel()
            try:
                await refresh_task
            except asyncio.CancelledError:
                pass

    async def initialize(self):
        cache = self.metrics_cache.load()
        if len(cache) == 0:
            self.logger.info("No cache found, fetching all series from Prometheus...")
            await self.prometheus.get_all_series(self.metrics_cache)
            cache = self.metrics_cache.load()
        await self.vector_store_manager.initialize(cache)

    def setup_routes(self):
        if os.path.exists(self.build_dir):
            self.fastapi_app.mount(
                "/static",
                StaticFiles(directory=os.path.join(self.build_dir, "static")),
                name="static",
            )
        
        @self.fastapi_app.get("/")
        async def serve_spa():
            return FileResponse(os.path.join(self.build_dir, "index.html"))

        @self.fastapi_app.get("/{catch_all:path}")
        async def serve_spa_catch_all(catch_all: str):
            filepath = os.path.join(self.build_dir, catch_all)
            if os.path.isfile(filepath):
                return FileResponse(filepath)
            else:
                return FileResponse(os.path.join(self.build_dir, "index.html"))

        self.fastapi_app.post("/chat")(self.chat_endpoint)

    async def refresh_data(self):
        while True:
            try:
                await asyncio.sleep(self.refresh_interval)
                self.logger.info("Refreshing metrics cache and embeddings...")
                await self.prometheus.get_all_series(self.metrics_cache)
                updated_cache = self.metrics_cache.load()
                await self.vector_store_manager.refresh_embeddings(updated_cache)
            except Exception as e:
                self.logger.error(
                    "Error in refresh_data", extra={"error": str(e)}, exc_info=True
                )
                await asyncio.sleep(5)  # Wait a bit before retrying if there's an error

    async def refresh_data_server(self):
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(self.refresh_interval)
                self.logger.info("Refreshing metrics cache and embeddings...")
                await self.prometheus.get_all_series(self.metrics_cache)
                updated_cache = self.metrics_cache.load()
                await self.vector_store_manager.refresh_embeddings(updated_cache)
            except Exception as e:
                if not self.shutdown_event.is_set():
                    self.logger.error(
                        "Error in refresh_data", extra={"error": str(e)}, exc_info=True
                    )
                    await asyncio.sleep(5)

    async def chat_endpoint(self, request: Request):
        try:
            data = await request.json()
            prompt = data.get("message", "")

            if not prompt:
                return {"error": "Message is required"}, 400

            async def generate():
                data = ""
                response = self.embed_model.get_text_embedding(prompt)
                async with self.vector_store_manager.lock:
                    results = self.vector_store_manager.index.vector_store.query(
                        VectorStoreQuery(query_embedding=response, similarity_top_k=5)
                    )

                    for result in results.nodes:
                        data += result.text + "\n"

                chat_messages = [
                    ChatMessage(
                        role="user",
                        # TODO(@saswatamcode): Add chat history
                        content=PROMPT_TEMPLATE_API.format(
                            data=data,
                            prompt=prompt,
                        ),
                    )
                ]

                full_response = ""
                for chunk in self.llm.stream_chat(messages=chat_messages):
                    full_response += chunk.delta
                    yield json.dumps({"type": "content", "data": chunk.delta}) + "\n"

                queries = extract_promql(full_response)
                if queries:
                    prometheus_links = []
                    for query in queries:
                        base_url = self.prom_external_url or self.prometheus_url
                        url = f"{base_url}/graph?g0.expr={urllib.parse.quote(query)}&g0.range_input={self.query_lookback_hours}h&g0.tab=0"
                        prometheus_links.append(url)

                    yield json.dumps(
                        {"type": "prometheus_links", "data": prometheus_links}
                    ) + "\n"

            return StreamingResponse(generate(), media_type="application/x-ndjson")

        except Exception as e:
            self.logger.error(
                "Chat endpoint error", extra={"error": str(e)}, exc_info=True
            )
            return {"error": str(e)}, 500

    async def chat(self):
        messages = []
        while True:
            try:
                prompt = await aioconsole.ainput(
                    f"{Fore.GREEN}Ask about your metrics (or 'exit' to quit): {Fore.RESET}"
                )
                if prompt.lower() == "exit":
                    break

                data = ""
                response = self.embed_model.get_text_embedding(prompt)
                async with self.vector_store_manager.lock:
                    results = self.vector_store_manager.index.vector_store.query(
                        VectorStoreQuery(query_embedding=response, similarity_top_k=5)
                    )

                    for result in results.nodes:
                        data += result.text + "\n"

                chat_messages = []
                chat_history = ""
                if messages:
                    recent_messages = messages[-4:]
                    for msg in recent_messages:
                        role = "User" if msg["role"] == "user" else "Assistant"
                        chat_history += f"{role}: {msg['content']}\n\n"

                chat_messages.append(
                    ChatMessage(
                        role="user",
                        content=PROMPT_TEMPLATE_CHAT.format(
                            data=data, chat_history=chat_history, prompt=prompt
                        ),
                    )
                )

                response = ""
                for chunk in self.llm.stream_chat(messages=chat_messages):
                    response += chunk.delta
                await aioconsole.aprint(f"\n{Fore.BLUE}Assistant:{Fore.RESET}")
                await aioconsole.aprint(response)

                # Print Prometheus links
                queries = extract_promql(response)
                if queries:
                    await aioconsole.aprint("\nView these queries in Prometheus:")
                    for i, query in enumerate(queries, 1):
                        base_url = self.prom_external_url or self.prometheus_url
                        url = f"{base_url}/graph?g0.expr={urllib.parse.quote(query)}&g0.range_input={self.query_lookback_hours}h&g0.tab=0"
                        await aioconsole.aprint(f"{i}. {url}")

                # Store message history
                messages.append({"role": "user", "content": prompt})
                messages.append({"role": "assistant", "content": response})

            except Exception as e:
                self.logger.error(
                    f"An error occurred", extra={"error": e}, exc_info=True
                )


def get_llm_from_config(config: dict):
    """Initialize LLM based on configuration."""
    llm_config = config.get("llm", {})
    provider = llm_config.get("provider", "ollama")

    if provider == "ollama":
        return Ollama(
            model=llm_config.get("model", "metricsGPT"),
            request_timeout=llm_config.get("timeout", 120.0),
        )
    elif provider == "openai":
        return OpenAI(
            model=llm_config.get("model", "gpt-4"),
            api_key=llm_config.get("api_key"),
            temperature=llm_config.get("temperature", 0.7),
        )
    elif provider == "azure":
        return AzureOpenAI(
            model=llm_config.get("model"),
            deployment_name=llm_config.get("deployment_name"),
            api_key=llm_config.get("api_key"),
            azure_endpoint=llm_config.get("endpoint"),
        )
    elif provider == "gemini":
        return Gemini(
            api_key=llm_config.get("api_key"),
            model=llm_config.get("model", "gemini-pro"),
        )
    elif provider == "watsonx":
        return WatsonxLLM(
            api_key=llm_config.get("api_key"),
            project_id=llm_config.get("project_id"),
            model_id=llm_config.get("model_id"),
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def get_embedding_model_from_config(config: dict):
    """Initialize embedding model based on configuration."""
    embed_config = config.get("embedding", {})
    provider = embed_config.get("provider", "ollama")

    if provider == "ollama":
        return OllamaEmbedding(model_name=embed_config.get("model", "nomic-embed-text"))
    elif provider == "openai":
        return OpenAIEmbedding(
            api_key=embed_config.get("api_key"),
            model=embed_config.get("model", "text-embedding-3-small"),
        )
    elif provider == "azure":
        return AzureOpenAIEmbedding(
            model=embed_config.get("model"),
            deployment_name=embed_config.get("deployment_name"),
            api_key=embed_config.get("api_key"),
            azure_endpoint=embed_config.get("endpoint"),
            api_version=embed_config.get("api_version", "2023-05-15"),
        )
    elif provider == "watsonx":
        return WatsonxEmbeddings(
            api_key=embed_config.get("api_key"),
            project_id=embed_config.get("project_id"),
            model_id=embed_config.get("model_id", "google/flan-ul2"),
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")


async def main():
    parser = argparse.ArgumentParser(description="metricsGPT - Chat with Your Metrics!")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Run in server mode instead of CLI chat mode",
    )
    args = parser.parse_args()
    logger = create_logger("metrics_gpt_server")

    # Load YAML configuration
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning(f"Configuration file {args.config} not found. Using defaults.")
        config = {}

    # Set defaults and override with YAML values
    settings = {
        "prometheus_url": config.get("prometheus_url", "http://localhost:9090"),
        "prom_external_url": config.get("prom_external_url", None),
        "prometheus_auth": config.get("prometheus_auth", None),
        "vectordb_path": config.get("vectordb_path", "./data.db"),
        "query_lookback_hours": float(config.get("query_lookback_hours", 1.0)),
        "series_cache_file": config.get("series_cache_file", "./series_cache.json"),
        "refresh_interval": int(config.get("refresh_interval", 300)),
        "server_host": config.get("server_host", "0.0.0.0"),
        "server_port": int(config.get("server_port", 8081)),
    }

    # Initialize LLM and embedding model from config
    Settings.llm = get_llm_from_config(config)
    Settings.embed_model = get_embedding_model_from_config(config)

    prometheus = PrometheusClient(
        settings["prometheus_url"],
        create_logger("prometheus"),
        settings["prometheus_auth"]
    )
    metrics_cache = MetricsCache(
        settings["series_cache_file"], create_logger("metrics_cache")
    )
    vector_store_manager = VectorStoreManager(
        settings["vectordb_path"],
        Settings.embed_model,
        create_logger("vector_store_manager"),
    )

    metrics_gpt_server = MetricsGPTServer(
        vector_store_manager,
        prometheus,
        metrics_cache,
        Settings.llm,
        Settings.embed_model,
        settings["prometheus_url"],
        settings["prom_external_url"],
        settings["query_lookback_hours"],
        settings["refresh_interval"],
        logger,
    )
    await metrics_gpt_server.initialize()

    if args.server:
        logger.info(
            f"Starting server on http://{settings['server_host']}:{settings['server_port']}"
        )
        config = uvicorn.Config(
            metrics_gpt_server.fastapi_app,
            host=settings["server_host"],
            port=settings["server_port"],
            loop="asyncio",
        )
        server = uvicorn.Server(config)
        try:
            await server.serve()
        except KeyboardInterrupt:
            logger.info("Shutting down server gracefully...")
            metrics_gpt_server.shutdown_event.set()
            await server.shutdown()
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise
        finally:
            metrics_gpt_server.shutdown_event.set()
            await asyncio.sleep(1)  # Give tasks time to clean up
    else:
        logger.info("Starting chat mode...")
        refresh_task = asyncio.create_task(metrics_gpt_server.refresh_data())
        chat_task = asyncio.create_task(metrics_gpt_server.chat())

        try:
            await chat_task
        finally:
            refresh_task.cancel()
            try:
                await refresh_task
            except asyncio.CancelledError:
                pass


if __name__ == "__main__":
    init()  # Initialize colorama
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested... exiting gracefully")
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise
