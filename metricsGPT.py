import ollama
import prometheus_api_client
import chromadb
import argparse
import re
import urllib.parse


class MetricsGPT:
    prom_client: prometheus_api_client.PrometheusConnect
    chromadb_collection: chromadb.Collection

    def __init__(
            self,
            prom_client: prometheus_api_client.PrometheusConnect,
            chromadb_collection: chromadb.Collection):
        self.prom_client = prom_client
        self.chromadb_collection = chromadb_collection

    def get_series(
            self,
            params: dict = None):
        '''get_series calls /api/v1/series on a Prometheus-compatible API server.'''

        params = params or {}
        response = self.prom_client._session.get(
            "{0}/api/v1/series".format(self.prom_client.url),
            verify=self.prom_client._session.verify,
            headers=self.prom_client.headers,
            params=params,
            auth=self.prom_client.auth,
            cert=self.prom_client._session.cert,
        )
        if response.status_code == 200:
            series = response.json()["data"]
        else:
            raise prometheus_api_client.PrometheusApiClientException(
                "HTTP Status Code {} ({!r})".format(
                    response.status_code, response.content))
        return series

    def get_all_series(self):
        '''get_all_series fetches all __name__ labels and then fetches all series for each metric
        from a Prometheus-compatible API server.'''

        try:
            all_series = []
            for metric in self.prom_client.all_metrics():
                series = self.get_series(params={"match[]": metric})
                all_series.append(series)

            return all_series
        except prometheus_api_client.PrometheusApiClientException as err:
            print(
                f"PrometheusApiClientException occurred while getting series: {err}")

    def generate_embeddings_for_metrics(
            self,
            embedding_model: str):
        '''generate_embeddings_for_metrics generates embeddings for all metrics in the Prometheus-compatible TSDB
        using provided embedding model.'''

        all_series = self.get_all_series()
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
            self.chromadb_collection.add(
                ids=[str(i)],
                embeddings=[embedding],
                documents=[metric],
            )

    def query_with_rag(
            self,
            prompt: str,
            embedding_model: str,
            query_model: str) -> str:
        '''query_with_rag generates embedding from the provided prompt and then queries chromadb for
        using the provided embedding model'''

        response = ollama.embeddings(
            prompt=prompt,
            model=embedding_model
        )
        results = self.chromadb_collection.query(
            query_embeddings=[response["embedding"]],
            n_results=200
        )
        data = results['documents'][0][0]
        output = ollama.generate(
            model=query_model,
            prompt=f"""Using the following facts:
        {data}
        To respond to this prompt: {prompt}
        """
        )

        print(output['response'])
        return output['response']


def extract_and_urlencode_promql(text):
    '''extract_and_urlencode_promql extracts all PROMQL code blocks from the LLM response and then URL encodes them.'''
    pattern = re.compile(r'<PROMQL>(.*?)</PROMQL>', re.DOTALL)
    matches = pattern.findall(text)

    urlencoded_matches = [urllib.parse.quote(match) for match in matches]

    return urlencoded_matches


def initialize_ollama_model(modelfile_path: str):
    '''initialize_ollama_model creates a model from the provided Modelfile.'''
    try:
        for response in ollama.create(
                model="metricsGPT", path=modelfile_path, stream=True
        ):
            print(response["status"])
    except ollama.ResponseError as err:
        print(
            f"Response error occurred while creating model from Modelfile: {err}")


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

    mgpt = MetricsGPT(prom_client, collection)
    mgpt.generate_embeddings_for_metrics(args.embedding_model)

    prompt = input("Enter your prompt: ")
    response = mgpt.query_with_rag(
        prompt, args.embedding_model, args.query_model)
    queries = extract_and_urlencode_promql(response)

    print("\nYou can view these queries here:")
    for query in queries:
        if args.prom_external_url is not None:
            print(f"{args.prom_external_url}/graph?g0.expr={query}\n")
        else:
            print(f"{args.prometheus_url}/graph?g0.expr={query}\n")


if __name__ == "__main__":
    main()
