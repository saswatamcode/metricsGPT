import ollama
import prometheus_api_client
import argparse
import re
import urllib.parse
from yaspin import yaspin
from yaspin.spinners import Spinners
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
import threading
import time


def get_series(
    prom_client: prometheus_api_client.PrometheusConnect, params: dict = None
):
    """get_series calls /api/v1/series on a Prometheus-compatible API server."""

    params = params or {}
    response = prom_client._session.get(
        "{0}/api/v1/series".format(prom_client.url),
        verify=prom_client._session.verify,
        headers=prom_client.headers,
        params=params,
        auth=prom_client.auth,
        cert=prom_client._session.cert,
    )
    if response.status_code == 200:
        series = response.json()["data"]
    else:
        raise prometheus_api_client.PrometheusApiClientException(
            f"/api/v1/series PrometheusApiClientException: HTTP Status Code {response.status_code} ({response.content})"
        )
    return series


def get_all_series(
    cache_file: str, prom_client: prometheus_api_client.PrometheusConnect
):
    """get_all_series fetches all __name__ labels and then fetches all series for each metric
    from a Prometheus-compatible API server."""

    try:
        for metric in prom_client.all_metrics():
            series_list = get_series(prom_client, params={"match[]": metric})
            for series in series_list:
                append_to_cache(cache_file, series)

    except prometheus_api_client.PrometheusApiClientException as err:
        print(
            f"/api/v1/labels PrometheusApiClientException occurred while getting series: {err}"
        )


def hash_metric(d: dict) -> str:
    return hashlib.sha256(str(json.dumps(d, sort_keys=True)).encode()).hexdigest()


def load_cache(cache_file: str) -> list:
    """Load cached time series from file as a dictionary."""
    try:
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                return json.load(f)
    except Exception as err:
        print(f"Error loading cache: {err}")
    return []


def save_cache(cache_file: str, timeseries: list):
    """Save time series to cache file."""
    try:
        with open(cache_file, "w") as f:
            json.dump(timeseries, f)
    except Exception as err:
        print(f"Error saving cache: {err}")


def append_to_cache(cache_file: str, series: dict):
    """Append a series to the cache."""

    cache = load_cache(cache_file)
    to_append = {**series}

    for metric in cache:
        if metric == to_append:
            return
    cache.append(to_append)
    save_cache(cache_file, cache)


def extract_promql(text):
    """extract_promql extracts all PROMQL code blocks from the LLM response."""
    pattern = re.compile(r"<PROMQL>(.*?)</PROMQL>", re.DOTALL)
    matches = pattern.findall(text)

    return matches


@yaspin(spinner=Spinners.moon, text="Initializing metricsGPT model from Modelfile...")
def initialize_ollama_model(modelfile_path: str):
    """initialize_ollama_model creates a model from the provided Modelfile."""
    try:
        ollama.create(model="metricsGPT", path=modelfile_path, stream=True)
    except ollama.ResponseError as err:
        print(f"Response error occurred while creating model from Modelfile: {err}")


def refresh_cache_periodically(cache_file: str, prom_client: prometheus_api_client.PrometheusConnect):
    """Refresh the series cache from Prometheus every 30 minutes."""
    while True:
        # Clear the cache by saving an empty list
        save_cache(cache_file, [])
        get_all_series(cache_file, prom_client)
        time.sleep(15)  # Sleep for 30 minutes


def main():
    parser = argparse.ArgumentParser(
        description="Talk to your metrics with metricsGPT!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--prometheus-url",
        type=str,
        default="http://localhost:9090",
        help="URL of the Prometheus-API compatible server to query.",
    )
    parser.add_argument(
        "--prom-external-url",
        type=str,
        help="External URL of the Prometheus-compatible instance, to provide URL links.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="nomic-embed-text",
        help="Model to use for RAG embeddings for your metrics.",
    )
    parser.add_argument(
        "--query-model",
        type=str,
        default="metricsGPT",
        help="Model to use for processing your prompts.",
    )
    parser.add_argument(
        "--vectordb-path",
        type=str,
        default="./data.db",
        help="Path to persist Milvus storage to.",
    )
    parser.add_argument(
        "--modelfile-path",
        type=str,
        default="./Modelfile",
        help="Path to Ollama Modelfile for metricsGPT model.",
    )
    parser.add_argument(
        "--query-lookback-hours",
        type=float,
        default=1,
        help="Hours to lookback when executing PromQL queries.",
    )
    parser.add_argument(
        "--query_step",
        type=str,
        default="14s",
        help="PromQL range query step parameter.",
    )
    parser.add_argument(
        "--series-cache-file",
        type=str,
        default="./series_cache.json",
        help="Path to the series cache file.",
    )
    
    args = parser.parse_args()
    init(autoreset=True)

    ascii_art = r"""
                _        _          _____ ______ _____
               | |      (_)        |  __ \| ___ \_   _|
 _ __ ___   ___| |_ _ __ _  ___ ___| |  \/| |_/ / | |
| '_ ` _ \ / _ \ __| '__| |/ __/ __| | __ |  __/  | |
| | | | | |  __/ |_| |  | | (__\__ \ |_\ \| |     | |
|_| |_| |_|\___|\__|_|  |_|\___|___/\____/\_|     \_/


"""
    print(Fore.BLUE + ascii_art)
    initialize_ollama_model(args.modelfile_path)

    prom_client = prometheus_api_client.PrometheusConnect(
        url=args.prometheus_url, disable_ssl=True
    )

    Settings.llm = Ollama(model=args.query_model, request_timeout=120.0)
    Settings.embed_model = OllamaEmbedding(
        model_name=args.embedding_model,
    )

    cache = load_cache(args.series_cache_file)
    if len(cache) == 0:
        print("Loading metrics from Prometheus...")
        get_all_series(args.series_cache_file, prom_client)

    # Start cache refresh thread
    refresh_thread = threading.Thread(
        target=refresh_cache_periodically,
        args=(args.series_cache_file, prom_client),
        daemon=True
    )
    refresh_thread.start()

    cache = load_cache(args.series_cache_file)

    if not os.path.exists(args.vectordb_path):
        print("Creating metricsGPT index...")
        vector_store = MilvusVectorStore(
            uri=args.vectordb_path, dim=768, overwrite=True, collection_name="metrics"
        )
        embed_model = OllamaEmbedding(model_name=args.embedding_model)

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Convert metrics to documents and add to index
        documents = []
        for metric in cache:
            metric_str = f"{metric['__name__']}{{{', '.join(f'{k}={v}' for k, v in metric.items() if k != '__name__')}}}"
            documents.append(Document(text=metric_str, doc_id=hash_metric(metric)))

        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, embed_model=embed_model
        )

        index.set_index_id("metricsGPT")
    else:
        print("Loading metricsGPT index from disk...")
        vector_store = MilvusVectorStore(
            uri=args.vectordb_path, dim=768, overwrite=False, collection_name="metrics"
        )
        embed_model = OllamaEmbedding(model_name=args.embedding_model)

        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, embed_model=embed_model
        )

    llm = Ollama(model=args.query_model, request_timeout=120.0)
    while True:
        prompt = input(">>> ")
        if prompt == "/exit":
            break
        elif len(prompt) > 0:
            try:
                data = ""

                response = embed_model.get_text_embedding(prompt)
                results = index.vector_store.query(
                    VectorStoreQuery(query_embedding=response, similarity_top_k=5)
                )

                for result in results.nodes:
                    data += result.text + "\n"

                print(data)

                messages = []

                messages.append(
                    ChatMessage(
                        role="user",
                        content=f"""Assume that the following is a list of metrics that are available to query:
                {data}
                To respond to this prompt: {prompt}
                Make sure to provide the PromQL query between <PROMQL> and </PROMQL> tags.
                Do not add any new line or space and put both tags on the same line, with the query in between.
                Make sure there are no characters in the query which can cause errors when URL encoded.
                """,
                    )
                )

                stream = llm.stream_chat(
                    messages=messages,
                )

                
                response = ""
                for chunk in stream:
                    print(Fore.BLUE + chunk.delta, end="", flush=True)
                    response = response + chunk.delta

                print("")
                messages.append(
                    ChatMessage(
                        role="assistant",
                        content=response,
                    )
                )

                queries = extract_promql(response)
                urlencoded_queries = [urllib.parse.quote(query) for query in queries]

                print("\nYou can view these queries here:")
                for query in urlencoded_queries:
                    if args.prom_external_url is not None:
                        print(
                            f"{args.prom_external_url}/graph?g0.expr={query}&g0.range_input={args.query_lookback_hours}h&g0.tab=0\n"
                        )
                    else:
                        print(
                            f"{args.prometheus_url}/graph?g0.expr={query}&g0.range_input={args.query_lookback_hours}h&g0.tab=0\n"
                        )

            except ollama.ResponseError as err:
                print(
                    f"Response occurred while embedding metrics from Prometheus or generating response: {err}"
                )
            except ValueError as err:
                print(f"Response occurred while querying Milvus: {err}")


if __name__ == "__main__":
    main()
