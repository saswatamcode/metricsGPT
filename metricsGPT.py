import ollama
import prometheus_api_client
import chromadb
import argparse
import re
import urllib.parse
from datetime import datetime, timedelta
from yaspin import yaspin
from yaspin.spinners import Spinners


class MetricsGPT:
    prom_client: prometheus_api_client.PrometheusConnect
    chromadb_collection: chromadb.Collection
    messages: list

    def __init__(
            self,
            prom_client: prometheus_api_client.PrometheusConnect,
            chromadb_collection: chromadb.Collection):
        self.prom_client = prom_client
        self.chromadb_collection = chromadb_collection
        self.messages = []

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

    @yaspin(spinner=Spinners.moon,
            text="Creating vectorDB from all exposed metrics...")
    def generate_embeddings_for_metrics(self, embedding_model: str):
        '''generate_embeddings_for_metrics generates embeddings for all metrics in the Prometheus-compatible TSDB using the provided embedding model.'''

        try:
            all_series = self.get_all_series()
            metric_docs = [
                f"This Prometheus-compatible TSDB has a metric named {series['__name__']} with the following label-value pairs: "
                + ", ".join(f"{label}={series[label]}" for label in series if label != "__name__")
                for series_list in all_series for series in series_list
            ]

            for i, metric in enumerate(metric_docs):
                response = ollama.embeddings(
                    prompt=metric, model=embedding_model)
                embedding = response["embedding"]
                self.chromadb_collection.add(
                    ids=[str(i)],
                    embeddings=[embedding],
                    documents=[metric],
                )
        except ollama.ResponseError as err:
            print(
                f"Response occurred while embedding metrics from Prometheus: {err}")

    def query_with_rag(
            self,
            prompt: str,
            embedding_model: str,
            query_model: str) -> str:
        '''query_with_rag generates embedding from the provided prompt and then queries chromadb for
        using the provided embedding model. Also provides it with chat history.'''

        try:
            response = ollama.embeddings(
                prompt=prompt,
                model=embedding_model
            )
            results = self.chromadb_collection.query(
                query_embeddings=[response["embedding"]],
                n_results=200
            )
            data = results['documents'][0][0]

            self.messages.append(
                {
                    'role': 'user',
                    'content': f"""Using the following facts:
            {data}
            To respond to this prompt: {prompt}
            Make sure to provide the PromQL query between <PROMQL> and </PROMQL> tags.
            Do not add any new line or space and put both tags on the same line, with the query in between.
            Make sure there are no characters in the query which can cause errors when URL encoded.
            """,
                }
            )

            stream = ollama.chat(
                model=query_model,
                messages=self.messages,
                stream=True,
            )

            response = ""
            for chunk in stream:
                part = chunk['message']['content']
                print(part, end='', flush=True)
                response = response + part

            print("")
            self.messages.append(
                {
                    'role': 'assistant',
                    'content': response,
                }
            )

            return response

        except ollama.ResponseError as err:
            print(
                f"Response occurred while embedding metrics from Prometheus or generating response: {err}")
        except ValueError as err:
            print(
                f"Response occurred while querying Chromadb: {err}")

    def query_with_results(
            self,
            query: str,
            data: str,
            prompt: str,
            query_model: str) -> str:
        '''query_with_results asks LLM to analyze the results of a PromQL query.'''
        try:
            self.messages.append(
                {'role': 'user', 'content': f"""We have queried prometheus with this PromQL query:
            {query}

            And have received the following response in json format:
            {data}

            Using the PromQL query and the JSON response, reason about what the query result means.
            Think in SRE-like terms, and try to imagine what the result will look like when graphed.
            Then based on SRE terminology, and how you imagine this data to look like, respond accurately to this prompt:

            {prompt}

            Do not repeat these instructions, and only respond to the above prompt concisely, and answer exactly what was asked.
            """, })

            stream = ollama.chat(
                model=query_model,
                messages=self.messages,
                stream=True,
            )

            response = ""
            for chunk in stream:
                part = chunk['message']['content']
                print(part, end='', flush=True)
                response = response + part

            print("")
            self.messages.append(
                {
                    'role': 'assistant',
                    'content': response,
                }
            )

            return response

        except ollama.ResponseError as err:
            print(
                f"Response occurred while generating response for Prometheus query result analysis: {err}")

    def query_prom(self, query: str, query_lookback_hours: float, step: str):
        '''query_prom queries the Prometheus-compatible API server with the provided query.'''
        try:
            response = self.prom_client.custom_query_range(
                query,
                start_time=(
                    datetime.now() -
                    timedelta(
                        hours=query_lookback_hours)),
                end_time=datetime.now(),
                step=step)
            return response
        except prometheus_api_client.PrometheusApiClientException as err:
            print(
                f"PrometheusApiClientException occurred while querying Prometheus: {err}")


def extract_promql(text):
    '''extract_promql extracts all PROMQL code blocks from the LLM response.'''
    pattern = re.compile(r'<PROMQL>(.*?)</PROMQL>', re.DOTALL)
    matches = pattern.findall(text)

    return matches


@yaspin(spinner=Spinners.moon,
        text="Initializing metricsGPT model from Modelfile...")
def initialize_ollama_model(modelfile_path: str):
    '''initialize_ollama_model creates a model from the provided Modelfile.'''
    try:
        ollama.create(
            model="metricsGPT", path=modelfile_path, stream=True
        )
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
    parser.add_argument(
        '--query-lookback-hours',
        type=float,
        default=1,
        help='Hours to lookback when executing PromQL queries.')
    parser.add_argument(
        '--query_step',
        type=str,
        default="14s",
        help='PromQL range query step.')

    args = parser.parse_args()

    initialize_ollama_model(args.modelfile_path)

    prom_client = prometheus_api_client.PrometheusConnect(
        url=args.prometheus_url, disable_ssl=True)
    chromadb_client = chromadb.PersistentClient(path=args.vectordb_path)
    collection = chromadb_client.get_or_create_collection(name="metrics")

    mgpt = MetricsGPT(prom_client, collection)
    mgpt.generate_embeddings_for_metrics(args.embedding_model)

    while True:
        prompt = input(">>> ")
        if prompt == "/exit":
            break
        elif len(prompt) > 0:
            response = mgpt.query_with_rag(
                prompt, args.embedding_model, args.query_model)
            queries = extract_promql(response)

            urlencoded_queries = [
                urllib.parse.quote(query) for query in queries]

            print("\nYou can view these queries here:")
            for query in urlencoded_queries:
                if args.prom_external_url is not None:
                    print(
                        f"{args.prom_external_url}/graph?g0.expr={query}&g0.range_input={args.query_lookback_hours}h&g0.tab=0\n")
                else:
                    print(
                        f"{args.prometheus_url}/graph?g0.expr={query}&g0.range_input={args.query_lookback_hours}h&g0.tab=0\n")

            for query in queries:
                response = mgpt.query_prom(query, args.query_lookback_hours,
                                           args.query_step)
                print(response)
                response_df = prometheus_api_client.MetricRangeDataFrame(
                    response)
                resp_json = response_df.to_json()

                while True:
                    # Create new chat context within one, to focus on query
                    # results
                    result_prompt = input("result>>> ")
                    if result_prompt == "/done":
                        break
                    elif len(result_prompt) > 0:
                        mgpt.query_with_results(
                            query, resp_json, result_prompt, args.query_model)


if __name__ == "__main__":
    main()
