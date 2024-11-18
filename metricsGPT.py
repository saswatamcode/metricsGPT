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
import streamlit as st
from streamlit_chat import message
# import threading
# import time


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


# def refresh_cache_periodically(cache_file: str, prom_client: prometheus_api_client.PrometheusConnect):
#     """Refresh the series cache from Prometheus every 30 minutes."""
#     while True:
#         # Clear the cache by saving an empty list
#         save_cache(cache_file, [])
#         get_all_series(cache_file, prom_client)
#         time.sleep(15)  # Sleep for 30 minutes


def initialize_page():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def display_chat_history():
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            message(msg["content"], is_user=True, key=f"msg_{i}")
        else:
            message(msg["content"], is_user=False, key=f"msg_{i}")

def main():
    # Must be the first Streamlit command
    st.set_page_config(page_title="metricsGPT 📊", page_icon="📈")
    
    # Then sidebar and other UI elements
    st.title("metricsGPT 📊 - Chat with Your Metrics!")
    st.sidebar.title("Configuration")
    args = type('Args', (), {})()
    
    args.prometheus_url = st.sidebar.text_input(
        "Prometheus URL",
        value="http://localhost:9090",
        help="URL of the Prometheus-API compatible server to query."
    )
    args.prom_external_url = st.sidebar.text_input(
        "External Prometheus URL",
        help="External URL of the Prometheus-compatible instance, to provide URL links."
    )
    args.embedding_model = st.sidebar.text_input(
        "Embedding Model",
        value="nomic-embed-text",
        help="Model to use for RAG embeddings for your metrics."
    )
    args.query_model = st.sidebar.text_input(
        "Query Model",
        value="metricsGPT",
        help="Model to use for processing your prompts."
    )
    args.vectordb_path = st.sidebar.text_input(
        "VectorDB Path",
        value="./data.db",
        help="Path to persist Milvus storage to."
    )
    args.modelfile_path = st.sidebar.text_input(
        "Modelfile Path",
        value="./Modelfile",
        help="Path to Ollama Modelfile for metricsGPT model."
    )
    args.query_lookback_hours = st.sidebar.number_input(
        "Query Lookback Hours",
        value=1.0,
        help="Hours to lookback when executing PromQL queries."
    )
    args.query_step = st.sidebar.text_input(
        "Query Step",
        value="14s",
        help="PromQL range query step parameter."
    )
    args.series_cache_file = st.sidebar.text_input(
        "Series Cache File",
        value="./series_cache.json",
        help="Path to the series cache file."
    )
    
    initialize_page()
    
    # Initialize components
    prom_client = prometheus_api_client.PrometheusConnect(
        url=args.prometheus_url, disable_ssl=True
    )
    
    Settings.llm = Ollama(model=args.query_model, request_timeout=120.0)
    Settings.embed_model = OllamaEmbedding(
        model_name=args.embedding_model,
    )

    # Initialize cache and vector store
    cache = load_cache(args.series_cache_file)
    if len(cache) == 0:
        with st.spinner("Loading metrics from Prometheus..."):
            get_all_series(args.series_cache_file, prom_client)
            cache = load_cache(args.series_cache_file)

    if not os.path.exists(args.vectordb_path):
        with st.spinner("Creating metricsGPT index..."):
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
        with st.spinner("Loading metricsGPT index from disk..."):
            vector_store = MilvusVectorStore(
                uri=args.vectordb_path, dim=768, overwrite=False, collection_name="metrics"
            )
            embed_model = OllamaEmbedding(model_name=args.embedding_model)

            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store, embed_model=embed_model
            )

    llm = Ollama(model=args.query_model, request_timeout=120.0)

    # Create a container for messages
    messages = st.container(height=500)
    
    # Display existing messages
    for message in st.session_state.messages:
        with messages.chat_message(message["role"]):
            st.write(message["content"])
            
            # If it's an assistant message with queries, show the Prometheus links
            if message["role"] == "assistant":
                queries = extract_promql(message["content"])
                urlencoded_queries = [urllib.parse.quote(query) for query in queries]
                
                if queries:
                    st.markdown("### View these queries in Prometheus:")
                    cols = st.columns(len(queries))  # Create columns for buttons
                    for idx, (query, col) in enumerate(zip(urlencoded_queries, cols)):
                        base_url = args.prom_external_url or args.prometheus_url
                        url = f"{base_url}/graph?g0.expr={query}&g0.range_input={args.query_lookback_hours}h&g0.tab=0"
                        with col:
                            st.link_button("View Query 📊", url, type="primary", use_container_width=True)
    
    # Chat input
    if prompt := st.chat_input("Ask about your metrics..."):
        # Display user message
        with messages.chat_message("user"):
            st.write(prompt)
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display assistant response
        with messages.chat_message("assistant"):
            try:
                data = ""
                response = embed_model.get_text_embedding(prompt)
                results = index.vector_store.query(
                    VectorStoreQuery(query_embedding=response, similarity_top_k=5)
                )

                for result in results.nodes:
                    data += result.text + "\n"

                messages = []
                # Get last few messages for context (e.g., last 2 exchanges)
                chat_history = ""
                if len(st.session_state.messages) > 0:
                    recent_messages = st.session_state.messages[-4:]  # Last 2 exchanges (2 user, 2 assistant)
                    for msg in recent_messages:
                        role = "User" if msg["role"] == "user" else "Assistant"
                        chat_history += f"{role}: {msg['content']}\n\n"

                messages.append(
                    ChatMessage(
                        role="user",
                        content=f"""First, explain what the query does and how it helps answer the question. Think of yourself as a PromQL Expert SRE.
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
                """,
                    )
                )

                response = ""
                message_placeholder = st.empty()
                with st.spinner("Thinking..."):
                    for chunk in llm.stream_chat(messages=messages):
                        response += chunk.delta
                        message_placeholder.write(response)
                
                # Clear the streaming placeholder
                message_placeholder.empty()
                
                # First display the explanation part
                explanation = response.split('<PROMQL>')[0].strip()
                st.write(explanation)
                
                # Then display queries in a box
                queries = extract_promql(response)
                if queries:
                    with st.container(border=True):
                        for query in queries:
                            st.code(query.strip(), language="promql")
                            base_url = args.prom_external_url or args.prometheus_url
                            url = f"{base_url}/graph?g0.expr={urllib.parse.quote(query)}&g0.range_input={args.query_lookback_hours}h&g0.tab=0"
                            st.link_button("View Query 📊", url, type="primary", use_container_width=True)

                # Save the sanitized response for history
                sanitized_response = explanation
                if queries:
                    sanitized_response += "\n\n" + "\n".join(queries)
                st.session_state.messages.append({"role": "assistant", "content": sanitized_response})

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
