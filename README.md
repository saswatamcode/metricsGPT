# metricsGPT

Talk to your metrics.

![Demo](./demo.png)

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
pip3 install -r requirements.txt
```

Have some local/remote prometheus up and running. You can use `make run-prom` to get one running in docker that scrapes itself.

Finally run,
```bash
streamlit run metricsGPT.py
```
and ask it to come up with PromQL queries.

## TODOs:
- Other models, OpenAI etc
- Use other Prom HTTP APIs for more context
- Range queries
- Visualize
- Embed query results for better analysis
- Run chromadb separately
- Refresh chromadb based on last mod