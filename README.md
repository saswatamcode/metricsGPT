# metricsGPT

Talk to your metrics.

```bash
usage: metricsGPT [-h] [--prometheus-url PROMETHEUS_URL] [--prom-external-url PROM_EXTERNAL_URL] [--embedding-model EMBEDDING_MODEL]
                  [--query-model QUERY_MODEL] [--vectordb-path VECTORDB_PATH] [--modelfile-path MODELFILE_PATH]
                  [--query-lookback-hours QUERY_LOOKBACK_HOURS] [--query_step QUERY_STEP]

Talk to your metrics with metricsGPT!

options:
  -h, --help            show this help message and exit
  --prometheus-url PROMETHEUS_URL
                        URL of the Prometheus-API compatible server. (default: http://localhost:9090)
  --prom-external-url PROM_EXTERNAL_URL
                        URL of the Prometheus instance. (default: None)
  --embedding-model EMBEDDING_MODEL
                        Model to use for RAG embeddings for your metrics. (default: nomic-embed-text)
  --query-model QUERY_MODEL
                        Model to use for processing your prompts. (default: metricsGPT)
  --vectordb-path VECTORDB_PATH
                        Path to persist chromadb storage. (default: ./data)
  --modelfile-path MODELFILE_PATH
                        Path to Ollama Modelfile for metricGPT model. (default: ./Modelfile)
  --query-lookback-hours QUERY_LOOKBACK_HOURS
                        Hours to lookback when executing PromQL queries. (default: 1)
  --query_step QUERY_STEP
                        PromQL range query step. (default: 14s)
```


## Installation

Make sure you have [ollama](https://ollama.com/) installed locally with at least one embedding model and chat-able model pulled, and Python 3.12+.

By default this tool uses [`llama3`](https://ollama.com/library/llama3) and [`nomic-embed-text`](https://ollama.com/library/nomic-embed-text).

```bash
ollama pull llama3
ollama pull nomic-embed-text
```

After cloning run,
```bash
make venv
source venv/bin/activate
make build
```

Have some local/remote prometheus up and running. You can use `make run-prom` to get one running in docker that scrapes itself.

Finally run,
```bash
metricsGPT
```
and ask it to come up with PromQL queries.

## TODOs:
- Other models, OpenAI etc
- Use other Prom HTTP APIs for more context
- Range queries
- Visualize
- Embed query results for better analysis
- Run chromadb separately