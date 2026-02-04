
from typing import List, Optional
from duckduckgo_search import DDGS
from agents.deals import Deal, DealSelection, ScrapedDeal
from agents.agent import Agent
from openai import OpenAI
import re

class SearchAgent(Agent):
    """
    SearchAgent replaces the RSS-based ScannerAgent for active user queries.
    It uses DuckDuckGo to find real-time products.
    """
    name = "Search Agent"
    color = Agent.CYAN
    MODEL = "gpt-4o-mini"
    
    SYSTEM_PROMPT = """You are a shopping assistant. From the provided list of search results, select the 5 most relevant product listings that appear to be valid e-commerce pages with clear pricing.
    Extract the price and description from the snippet if possible.
    Respond strictly in JSON."""

    def __init__(self):
        self.log("Search Agent is initializing with DuckDuckGo")
        self.openai = OpenAI()
        self.ddgs = DDGS()

    def search(self, query: str) -> List[Deal]:
        """
        Active search for specific products
        """
        self.log(f"Search Agent executing active search for: '{query}'")
        
        # 1. Surgical Strike: Get search results (Fast & Free)
        # Try specific e-commerce sites first
        search_query = f"{query} price site:amazon.com OR site:bestbuy.com"
        results = list(self.ddgs.text(search_query, max_results=10))
        
        # Fallback: Broad search if strict search fails
        if not results:
            self.log("Strict search returned 0 results. Retrying with broad search...")
            results = list(self.ddgs.text(f"{query} price", max_results=10))
        
        self.log(f"Search Agent found {len(results)} raw results from DuckDuckGo")
        
        # 2. Heuristic Filter & Deal Construction
        found_deals = []
        for r in results:
            # Simple heuristic: try to find a price in the snippet
            price = self._extract_price(r.get('body', ''))
            if price > 0:
                found_deals.append(Deal(
                    product_description=r.get('title', '') + " - " + r.get('body', '')[:200],
                    price=price,
                    url=r.get('href', '')
                ))
        
        # 3. Limit to top 5
        return found_deals[:5]

    def _extract_price(self, text: str) -> float:
        """
        Heuristic to extract price from text snippet like "$299.00"
        """
        match = re.search(r'\$(\d+(?:,\d+)*(?:\.\d{2})?)', text)
        if match:
            return float(match.group(1).replace(',', ''))
        return 0.0
        
    def scan_query(self, query: str) -> Optional[DealSelection]:
        """
        The main entry point for the dashboard
        """
        deals = self.search(query)
        if deals:
            self.log(f"Search Agent returning {len(deals)} valid deals")
            return DealSelection(deals=deals)
        return None
