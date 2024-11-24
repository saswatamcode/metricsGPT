from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read()


setup(
    name="metricsGPT",
    version="0.1.0",
    author="saswatamcode",
    description="Chat with Your Metrics using LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/saswatamcode/metricsGPT",
    packages=find_packages(),
    py_modules=["metricsGPT"],
    license="Apache License 2.0",
    python_requires=">=3.12",
    install_requires = [requirements],
    entry_points={
        "console_scripts": [
            "metricsgpt=metricsGPT:runner",
        ],
    },
    include_package_data=True,
    package_data={
        "metricsGPT": ["ui/build/**/*"],
    },
)