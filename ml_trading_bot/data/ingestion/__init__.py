"""
Data ingestion module for stocks and cryptocurrencies
"""
from .stock_fetcher import StockDataFetcher
from .crypto_fetcher import CryptoDataFetcher

__all__ = ['StockDataFetcher', 'CryptoDataFetcher']
