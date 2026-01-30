"""
Content generation module for RTB simulation article series.

Extracts insights from simulation results and generates article-ready content.
"""

from .article_generator import ArticleGenerator, ArticleSeries, Article
from .sample_data import generate_sample_simulation
from .findings import FindingExtractor, Finding

__all__ = [
    "ArticleGenerator",
    "ArticleSeries",
    "Article",
    "generate_sample_simulation",
    "FindingExtractor",
    "Finding",
]
