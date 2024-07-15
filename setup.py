import setuptools

setuptools.setup(
    name="metricsGPT",
    version="0.0.1",
    author="saswatamcode",
    py_modules=['metricsGPT'],
    description="Talk to your metrics, with metricsGPT.",
    entry_points={
        'console_scripts': [
            'metricsGPT = metricsGPT:main',
        ],
    },
    install_requires=["ollama", "chromadb", "prometheus-api-client"],
)
