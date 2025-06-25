#!/usr/bin/env python3
"""
Tool Learning System - LLM Multi-Route Selection + arXiv Paper Search

Consolidated system for:
1. LLM-based intelligent multi-route selection 
2. arXiv paper search (live API + static dataset)
3. Response generation with combined route context
"""

import os
import json
import glob
import re
import time
import requests
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import logging

# Import prompts
from prompts import (
    get_route_selection_prompt,
    get_response_generation_prompt, 
    get_fallback_response
)

# Import arXiv library
try:
    import arxiv
    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False

logger = logging.getLogger(__name__)

class ToolLearningSystem:
    """Main tool learning system with LLM multi-routing and paper search"""
    
    def __init__(self, use_static_data: bool = False, ollama_model: str = "llama3.2"):
        self.use_static_data = use_static_data
        self.ollama_model = ollama_model
        self.dataset_dir = "dataset/arxiv"
        self.search_count = 0
        
        # Available routes based on dataset
        self.routes = {
            'searchPapers': 'Search for research papers on specific topics',
            'getAuthorInfo': 'Get information about authors and researchers', 
            'getCitations': 'Analyze citation counts and paper impact',
            'getRelatedPapers': 'Find papers related to specific research',
            'comparePapers': 'Compare different papers or approaches',
            'trendAnalysis': 'Analyze research trends over time',
            'journalAnalysis': 'Analyze journals and publication venues'
        }
        
        # Check system availability
        self.llm_available = self._check_ollama_connection()
        self.arxiv_available = ARXIV_AVAILABLE
        
        logger.info(f"Tool Learning System initialized")
        logger.info(f"LLM available: {self.llm_available}")
        logger.info(f"arXiv API: {self.arxiv_available}")
        logger.info(f"Static data mode: {self.use_static_data}")
    
    def _check_ollama_connection(self) -> bool:
        """Check if Ollama is available"""
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def select_route(self, query: str) -> Tuple[List[str], float, str]:
        """Select one or more routes for a query using LLM or fallback"""
        if self.llm_available:
            return self._select_route_llm(query)
        else:
            return self._select_route_fallback(query)
    
    def _select_route_llm(self, query: str) -> Tuple[List[str], float, str]:
        """Use LLM for multi-route selection"""
        prompt = get_route_selection_prompt(query)
        
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': self.ollama_model,
                    'prompt': prompt,
                    'stream': False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                selected_text = result['response'].strip()
                
                # Parse multiple routes
                routes = [route.strip() for route in selected_text.split(',')]
                valid_routes = []
                
                for route in routes:
                    if route in self.routes:
                        valid_routes.append(route)
                    else:
                        # Try to extract valid route
                        for valid_route in self.routes:
                            if valid_route.lower() in route.lower():
                                valid_routes.append(valid_route)
                                break
                
                if valid_routes:
                    confidence = 0.95 if len(valid_routes) == len(routes) else 0.85
                    return valid_routes, confidence, f"LLM selected {', '.join(valid_routes)}"
                            
        except Exception as e:
            logger.warning(f"LLM route selection failed: {e}")
        
        # Fallback if LLM fails
        return self._select_route_fallback(query)
    
    def _select_route_fallback(self, query: str) -> Tuple[List[str], float, str]:
        """Fallback route selection using keywords"""
        query_lower = query.lower()
        
        # Route patterns
        patterns = {
            'searchPapers': ['find', 'search', 'get', 'papers', 'research', 'articles'],
            'getAuthorInfo': ['who', 'author', 'researcher', 'publications', 'works'],
            'getCitations': ['citation', 'impact', 'cited', 'h-index', 'bibliometric'],
            'getRelatedPapers': ['related', 'similar', 'connected', 'building on'],
            'comparePapers': ['compare', 'difference', 'versus', 'vs', 'analysis'],
            'trendAnalysis': ['trends', 'evolution', 'development', 'over time'],
            'journalAnalysis': ['journal', 'venue', 'conference', 'impact factor']
        }
        
        scores = {}
        for route, keywords in patterns.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                scores[route] = score / len(keywords)
        
        if scores:
            # Select routes with high scores (above threshold or top 2)
            sorted_routes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            selected_routes = []
            
            # Take top route plus any others with similar scores
            if sorted_routes:
                top_score = sorted_routes[0][1]
                for route, score in sorted_routes:
                    if score >= top_score * 0.7 and len(selected_routes) < 3:
                        selected_routes.append(route)
            
            if selected_routes:
                avg_confidence = sum(scores[route] for route in selected_routes) / len(selected_routes)
                return selected_routes, avg_confidence, f"Keyword-based selection: {', '.join(selected_routes)}"
        
        return ['searchPapers'], 0.5, "Default route: searchPapers"
    
    def search_papers(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for papers using static data or live API"""
        self.search_count += 1
        
        if self.use_static_data:
            return self._search_static_dataset(query, max_results)
        else:
            return self._search_arxiv_live(query, max_results)
    
    def _search_static_dataset(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search static arXiv dataset files"""
        results = []
        
        if not os.path.exists(self.dataset_dir):
            logger.warning(f"Dataset directory not found: {self.dataset_dir}")
            return results
        
        # Get all JSONL files
        jsonl_files = glob.glob(os.path.join(self.dataset_dir, "*.jsonl"))
        
        if not jsonl_files:
            logger.warning(f"No dataset files found in {self.dataset_dir}")
            return results
        
        query_terms = [term.lower() for term in query.split()]
        
        for file_path in jsonl_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if len(results) >= max_results:
                            break
                        
                        line = line.strip()
                        if not line:
                            continue
                            
                        try:
                            paper = json.loads(line)
                            
                            # Simple relevance check
                            title = paper.get('title', '').lower()
                            abstract = paper.get('abstract', '').lower()
                            
                            # Check if any query term matches
                            if any(term in title or term in abstract for term in query_terms):
                                formatted_paper = self._format_paper(paper)
                                results.append(formatted_paper)
                                
                        except json.JSONDecodeError:
                            continue
                
                if len(results) >= max_results:
                    break
                    
            except Exception as e:
                logger.warning(f"Failed to load dataset file {file_path}: {e}")
        
        logger.info(f"Found {len(results)} papers from static dataset")
        return results[:max_results]
    
    def _search_arxiv_live(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search arXiv using live API"""
        results = []
        
        if not ARXIV_AVAILABLE:
            logger.warning("arXiv library not available")
            return self._search_static_dataset(query, max_results)
        
        try:
            logger.info(f"arXiv live search: '{query}'")
            
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            for paper in search.results():
                try:
                    formatted_paper = {
                        'title': paper.title.strip(),
                        'authors': [author.name for author in paper.authors][:5],
                        'abstract': paper.summary.strip()[:500] if paper.summary else '',
                        'year': paper.published.year if paper.published else datetime.now().year,
                        'doi': paper.doi if paper.doi else '',
                        'arxiv_id': paper.entry_id.split('/')[-1],
                        'source': 'arXiv Live',
                        'url': paper.entry_id,
                        'categories': [cat for cat in paper.categories] if paper.categories else []
                    }
                    
                    results.append(formatted_paper)
                    
                except Exception as e:
                    logger.warning(f"Failed to process arXiv paper: {e}")
                    continue
            
            logger.info(f"Retrieved {len(results)} papers from arXiv live API")
                    
        except Exception as e:
            logger.error(f"arXiv live search failed: {e}")
            # Fallback to static dataset
            return self._search_static_dataset(query, max_results)
            
        return results
    
    def _format_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Format paper data into consistent structure"""
        return {
            'title': paper.get('title', 'No title').strip(),
            'authors': self._extract_authors(paper),
            'abstract': self._extract_abstract(paper),
            'year': self._extract_year(paper),
            'doi': paper.get('doi', '').strip(),
            'arxiv_id': paper.get('arxiv_id', paper.get('id', '')).strip(),
            'source': 'arXiv Static',
            'url': paper.get('url', '').strip(),
            'categories': paper.get('categories', [])
        }
    
    def _extract_authors(self, paper: Dict[str, Any]) -> List[str]:
        """Extract author names from paper data"""
        authors = paper.get('authors', [])
        
        if isinstance(authors, list):
            author_names = []
            for author in authors:
                if isinstance(author, str):
                    author_names.append(author.strip())
                elif isinstance(author, dict):
                    name = author.get('name', '')
                    if not name:
                        first = author.get('first_name', '')
                        last = author.get('last_name', '')
                        name = f"{first} {last}".strip()
                    if name:
                        author_names.append(name)
            return author_names[:5]  # Limit to 5 authors
        elif isinstance(authors, str):
            return [name.strip() for name in authors.split(',') if name.strip()][:5]
        
        return ['Unknown Author']
    
    def _extract_abstract(self, paper: Dict[str, Any]) -> str:
        """Extract abstract from paper data"""
        abstract = paper.get('abstract', paper.get('summary', ''))
        if abstract:
            if len(abstract) > 500:
                return abstract[:500] + '...'
            return abstract.strip()
        return ''
    
    def _extract_year(self, paper: Dict[str, Any]) -> int:
        """Extract publication year from paper data"""
        year_fields = ['year', 'publication_year', 'date', 'published']
        
        for field in year_fields:
            if field in paper:
                year_value = paper[field]
                if isinstance(year_value, int) and 1900 <= year_value <= 2030:
                    return year_value
                elif isinstance(year_value, str):
                    year_match = re.search(r'(\d{4})', year_value)
                    if year_match:
                        year = int(year_match.group(1))
                        if 1900 <= year <= 2030:
                            return year
        
        return datetime.now().year
    
    def generate_response(self, query: str, routes: List[str], papers: List[Dict[str, Any]]) -> str:
        """Generate LLM response using query, routes, and papers"""
        if not self.llm_available:
            return self._generate_fallback_response(query, routes, papers)
        
        # Use centralized prompt generation
        prompt = get_response_generation_prompt(routes, query, papers)
        
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': self.ollama_model,
                    'prompt': prompt,
                    'stream': False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['response'].strip()
                
        except Exception as e:
            logger.warning(f"LLM response generation failed: {e}")
        
        return self._generate_fallback_response(query, routes, papers)
    
    def _generate_fallback_response(self, query: str, routes: List[str], papers: List[Dict[str, Any]]) -> str:
        """Generate fallback response when LLM is not available"""
        return get_fallback_response(query, routes, papers)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        # Count papers in static dataset
        static_count = 0
        if os.path.exists(self.dataset_dir):
            for file_path in glob.glob(os.path.join(self.dataset_dir, "*.jsonl")):
                try:
                    with open(file_path, 'r') as f:
                        static_count += sum(1 for line in f if line.strip())
                except Exception:
                    pass
        
        return {
            'total_searches': self.search_count,
            'llm_available': self.llm_available,
            'arxiv_available': self.arxiv_available,
            'static_data_mode': self.use_static_data,
            'static_papers_count': static_count,
            'available_routes': list(self.routes.keys()),
            'routes_count': len(self.routes),
            'multi_route_support': True
        }

def main():
    """Test the tool learning system"""
    print("🤖 Multi-Route Tool Learning System Test")
    print("=" * 45)
    
    # Initialize system
    system = ToolLearningSystem()
    
    # Test queries
    test_queries = [
        "find papers about machine learning",
        "who is Geoffrey Hinton and what are his papers",
        "compare BERT and GPT models and their citations",
        "trends in AI research and best venues"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        
        # Multi-route selection
        routes, confidence, explanation = system.select_route(query)
        print(f"   Routes: {', '.join(routes)} (confidence: {confidence:.3f})")
        
        # Paper search
        papers = system.search_papers(query, max_results=3)
        print(f"   Papers found: {len(papers)}")
        
        # Response generation
        response = system.generate_response(query, routes, papers)
        print(f"   Response: {response[:100]}...")
    
    # System statistics
    stats = system.get_statistics()
    print(f"\n📊 System Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")

if __name__ == "__main__":
    main() 