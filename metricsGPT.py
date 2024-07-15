import ollama
import prometheus_api_client
import chromadb
import argparse


def initialize_ollama_model(modelfile_path: str):
    try:
        for response in ollama.create(
            model="metricsGPT", path=modelfile_path, stream=True
        ):
            print(response["status"])
    except ollama.ResponseError as err:
        print(
            f"Response error occurred while creating model from Modelfile: {err}")


def get_series(
        prom: prometheus_api_client.PrometheusConnect,
        params: dict = None):
    params = params or {}
    response = prom._session.get(
        "{0}/api/v1/series".format(prom.url),
        verify=prom._session.verify,
        headers=prom.headers,
        params=params,
        auth=prom.auth,
        cert=prom._session.cert,
    )
    if response.status_code == 200:
        series = response.json()["data"]
    else:
        raise prometheus_api_client.PrometheusApiClientException(
            "HTTP Status Code {} ({!r})".format(
                response.status_code, response.content))
    return series


def get_prometheus_series(prom: prometheus_api_client.PrometheusConnect):
    try:
        all_series = []
        for metric in prom.all_metrics():
            series = get_series(prom, params={"match[]": metric})
            all_series.append(series)

        return all_series
    except prometheus_api_client.PrometheusApiClientException as err:
        print(
            f"PrometheusApiClientException occurred while getting series: {err}")


def generate_embeddings_for_metrics(
        collection: chromadb.Collection,
        prom: prometheus_api_client.PrometheusConnect,
        embedding_model: str):
    all_series = get_prometheus_series(prom)
    metric_doc = []
    for series_list in all_series:
        for series in series_list:
            labelstr = ""
            for label in series:
                if label != "__name__":
                    labelstr += f"{label}={series[label]}, "
            metric_doc.append(
                f"This Prometheus-compatible TSDB has a metric named {
                    series["__name__"]} with the following label-value pairs: {labelstr}\n")

    for i, metric in enumerate(metric_doc):
        response = ollama.embeddings(prompt=metric, model=embedding_model)
        embedding = response["embedding"]
        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[metric],
        )


def query_with_rag(
        prompt: str,
        collection: chromadb.Collection,
        embedding_model: str,
        query_model: str):
    response = ollama.embeddings(
        prompt=prompt,
        model=embedding_model
    )
    results = collection.query(
        query_embeddings=[response["embedding"]],
        n_results=200
    )
    data = results['documents'][0][0]
    output = ollama.generate(
        model=query_model,
        prompt=f"""Using the following facts:
      {data}
      Respond to this prompt: {prompt}
      """
    )

    print(output['response'])


def main():
    parser = argparse.ArgumentParser(
        description='Talk to your metrics with metricsGPT!',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--prometheus-url',
        type=str,
        default="http://localhost:9090",
        help='URL of the Prometheus-API compatible server.')
    parser.add_argument(
        '--prom-external-url',
        type=str,
        help='URL of the Prometheus instance.')
    parser.add_argument(
        '--embedding-model',
        type=str,
        default="nomic-embed-text",
        help='Model to use for RAG embeddings for your metrics.')
    parser.add_argument(
        '--query-model',
        type=str,
        default="metricsGPT",
        help='Model to use for processing your prompts.')
    parser.add_argument(
        '--vectordb-path',
        type=str,
        default="./data",
        help='Path to persist chromadb storage.')
    parser.add_argument(
        '--modelfile-path',
        type=str,
        default="./Modelfile",
        help='Path to Ollama Modelfile for metricGPT model.')

    args = parser.parse_args()

    initialize_ollama_model(args.modelfile_path)

    prom_client = prometheus_api_client.PrometheusConnect(
        url=args.prometheus_url, disable_ssl=True)
    chromadb_client = chromadb.PersistentClient(path=args.vectordb_path)
    collection = chromadb_client.get_or_create_collection(name="metrics")

    generate_embeddings_for_metrics(
        collection, prom_client, args.embedding_model)

    prompt = input("Enter your prompt: ")
    query_with_rag(prompt, collection, args.embedding_model, args.query_model)


if __name__ == "__main__":
    main()
