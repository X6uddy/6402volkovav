from setuptools import setup, find_packages

setup(
    name="trends_analyzer",
    version="0.1.0",
    packages=find_packages(include=["trends_analyzer", "trends_analyzer.*"]),
    install_requires=[
        'pytrends',
        'pandas',
        'numpy',
        'statsmodels',
    ],
    description="Анализ трендов",
    author="sasha volkov",
    author_email="sashavolkov2211@gmail.com",
    long_description_content_type="text/markdown",
)
