import prometheus_api_client
import argparse
import re
import urllib.parse
from colorama import Fore, init
import os
import json
import hashlib
from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.llms import ChatMessage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
import logging
import traceback
import uvicorn
import asyncio
import aioconsole
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import sys


PROMPT_TEMPLATE = """First, explain what the query does and how it helps answer the question. Think of yourself as a PromQL Expert SRE.
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
        # Start with basic log information
        log_data = {
            "level": record.levelname,
            "time": self.formatTime(record, self.datefmt),
            "component": record.name,
            "message": record.getMessage(),
        }

        # Include any extra fields
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        # Include stack trace if an exception is present
        if record.exc_info:
            stack_trace = traceback.format_exception(*record.exc_info)
            log_data["stack_trace"] = "".join(stack_trace).strip()

        # Build logfmt key-value string
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

    def __init__(self, url: str, logger: logging.Logger):
        self.client = prometheus_api_client.PrometheusConnect(url=url, disable_ssl=True)
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
        # Determine the base path for static files
        if getattr(sys, 'frozen', False):
            # Running in PyInstaller bundle
            self.build_dir = os.path.join(sys._MEIPASS, "ui/build")
        else:
            # Running in normal Python environment
            self.build_dir = os.path.join(os.path.dirname(__file__), "ui", "build")
            
        if os.path.exists(self.build_dir):
            self.fastapi_app.mount("/static", StaticFiles(directory=os.path.join(self.build_dir, "static")), name="static")
            
        self.setup_routes()
        self.shutdown_event = asyncio.Event()
        self.refresh_interval = refresh_interval

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        # Start the refresh task
        refresh_task = asyncio.create_task(self.refresh_data_server())
        try:
            yield
        finally:
            # Cleanup: Cancel the refresh task when shutting down
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
                        content=PROMPT_TEMPLATE.format(
                            data=data,
                            chat_history="",  # For now, not maintaining chat history in API mode
                            prompt=prompt,
                        ),
                    )
                ]

                # Stream the main response
                full_response = ""
                for chunk in self.llm.stream_chat(messages=chat_messages):
                    full_response += chunk.delta
                    yield json.dumps({"type": "content", "data": chunk.delta}) + "\n"

                # Extract and send Prometheus links
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
                        content=PROMPT_TEMPLATE.format(
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


async def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="metricsGPT - Chat with Your Metrics!")
    parser.add_argument(
        "--prometheus-url",
        default="http://localhost:9090",
        help="URL of the Prometheus-API compatible server to query.",
    )
    parser.add_argument(
        "--prom-external-url",
        help="External URL of the Prometheus-compatible instance, to provide URL links.",
    )
    parser.add_argument(
        "--embedding-model",
        default="nomic-embed-text",
        help="Model to use for RAG embeddings for your metrics.",
    )
    parser.add_argument(
        "--query-model",
        default="metricsGPT",
        help="Model to use for processing your prompts.",
    )
    parser.add_argument(
        "--vectordb-path",
        default="./data.db",
        help="Path to persist Milvus storage to.",
    )
    parser.add_argument(
        "--modelfile-path",
        default="./Modelfile",
        help="Path to Ollama Modelfile for metricsGPT model.",
    )
    parser.add_argument(
        "--query-lookback-hours",
        type=float,
        default=1.0,
        help="Hours to lookback when executing PromQL queries.",
    )
    parser.add_argument(
        "--query-step", default="14s", help="PromQL range query step parameter."
    )
    parser.add_argument(
        "--series-cache-file",
        default="./series_cache.json",
        help="Path to the series cache file.",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Run in server mode instead of CLI chat mode",
    )
    parser.add_argument(
        "--refresh-interval",
        type=int,
        default=300,
        help="Interval in seconds between metric cache refreshes.",
    )

    args = parser.parse_args()
    logger = create_logger("metrics_gpt_server")

    Settings.llm = Ollama(model=args.query_model, request_timeout=120.0)
    Settings.embed_model = OllamaEmbedding(
        model_name=args.embedding_model,
    )

    prometheus = PrometheusClient(args.prometheus_url, create_logger("prometheus"))
    metrics_cache = MetricsCache(args.series_cache_file, create_logger("metrics_cache"))
    vector_store_manager = VectorStoreManager(
        args.vectordb_path, Settings.embed_model, create_logger("vector_store_manager")
    )

    llm = Ollama(model=args.query_model, request_timeout=120.0)
    embed_model = OllamaEmbedding(model_name=args.embedding_model)

    metrics_gpt_server = MetricsGPTServer(
        vector_store_manager,
        prometheus,
        metrics_cache,
        llm,
        embed_model,
        args.prometheus_url,
        args.prom_external_url,
        args.query_lookback_hours,
        args.refresh_interval,
        logger,
    )
    await metrics_gpt_server.initialize()

    if args.server:
        # Run in server mode
        logger.info("Starting server on http://localhost:8081")
        config = uvicorn.Config(
            metrics_gpt_server.fastapi_app,
            host="0.0.0.0",
            port=8081,
            loop="asyncio",
        )
        server = uvicorn.Server(config)
        try:
            await server.serve()
        except KeyboardInterrupt:
            logger.info("Shutting down server gracefully...")
            metrics_gpt_server.shutdown_event.set()  # Signal shutdown
            await server.shutdown()  # Gracefully shutdown uvicorn
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise
        finally:
            # Ensure cleanup happens
            metrics_gpt_server.shutdown_event.set()
            await asyncio.sleep(1)  # Give tasks time to clean up
    else:
        # Run in CLI chat mode
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
