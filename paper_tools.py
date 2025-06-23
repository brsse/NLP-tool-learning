#!/usr/bin/env python3
"""
Real Paper Tools for Tool Learning Testing

Uses paperscraper for real paper data from multiple databases.
Uses static dataset files when available, APIs when not.
NO MOCK DATA - only real research papers.
"""

import os
import tempfile
import json
import glob
import re
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# Import paperscraper with comprehensive database support
try:
    # Use underlying libraries directly to avoid paperscraper bugs
    import arxiv
    import pymed
    from paperscraper.scholar import get_and_dump_scholar_papers  # Keep this one
    ARXIV_AVAILABLE = True
    PUBMED_AVAILABLE = True
    SCHOLAR_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False
    PUBMED_AVAILABLE = False
    SCHOLAR_AVAILABLE = False

# Disable Scholar API to avoid captcha issues - 2 APIs (arXiv + PubMed) are sufficient
SCHOLAR_AVAILABLE = False

PAPERSCRAPER_AVAILABLE = ARXIV_AVAILABLE or PUBMED_AVAILABLE

logger = logging.getLogger(__name__)

class PaperTools:
    """Real paper search tools using paperscraper APIs and static dataset files"""
    
    def __init__(self, use_static_data: bool = False):
        self.use_static_data = use_static_data
        self.search_count = 0
        self.cache_dir = "paper_cache"
        self.dataset_dir = "dataset"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Check individual API availability (arXiv + PubMed only)
        apis_available = []
        if ARXIV_AVAILABLE:
            apis_available.append("arXiv")
        if PUBMED_AVAILABLE:
            apis_available.append("PubMed")
        
        if not apis_available and not use_static_data:
            logger.warning("No APIs available and static data disabled. Limited functionality.")
        else:
            logger.info(f"Available APIs: {apis_available}")
        
    def search_papers(self, query: str, max_results: int = 10, database: str = "All") -> List[Dict[str, Any]]:
        """
        Search for papers using static dataset files or real APIs
        
        Args:
            query: Search query
            max_results: Maximum number of results
            database: Database to search ("All", "arXiv", "PubMed", "Scholar", "bioRxiv", "medRxiv", "chemRxiv")
            
        Returns:
            List of paper dictionaries
        """
        self.search_count += 1
        
        # Clean and sanitize query
        clean_query = self._sanitize_query(query)
        logger.info(f"Searching for '{clean_query}' in {database} (max: {max_results}) [{'Static' if self.use_static_data else 'Live'}]")
        
        if self.use_static_data:
            # Use static dataset files for consistent testing
            return self._search_static_datasets(clean_query, database, max_results)
        
        # Use real APIs only (arXiv + PubMed)
        results = []
        
        if not (ARXIV_AVAILABLE or PUBMED_AVAILABLE):
            logger.error("No APIs available and static data disabled")
            return []
        
        try:
            if database == "All":
                # Search both APIs: arXiv and PubMed
                per_db = max(1, max_results // 2)
                
                # arXiv first (most reliable)
                if ARXIV_AVAILABLE:
                    logger.info("Searching arXiv...")
                    arxiv_results = self._search_arxiv_safe(clean_query, per_db)
                    results.extend(arxiv_results)
                    logger.info(f"arXiv returned {len(arxiv_results)} results")
                
                # PubMed second
                if PUBMED_AVAILABLE and len(results) < max_results:
                    logger.info("Searching PubMed...")
                    remaining = max_results - len(results)
                    pubmed_results = self._search_pubmed_safe(clean_query, remaining)
                    results.extend(pubmed_results)
                    logger.info(f"PubMed returned {len(pubmed_results)} results")
                    
            elif database == "arXiv" and ARXIV_AVAILABLE:
                results = self._search_arxiv_safe(clean_query, max_results)
            elif database == "PubMed" and PUBMED_AVAILABLE:
                results = self._search_pubmed_safe(clean_query, max_results)
            elif database == "Scholar":
                logger.warning("Google Scholar API disabled to avoid captcha issues. Use arXiv or PubMed instead.")
                results = []
            elif database in ["bioRxiv", "medRxiv", "chemRxiv"]:
                # These require static dataset files
                results = self._search_static_datasets(clean_query, database, max_results)
            else:
                # Default to arXiv if available, otherwise try PubMed
                if ARXIV_AVAILABLE:
                    results = self._search_arxiv_safe(clean_query, max_results)
                elif PUBMED_AVAILABLE:
                    results = self._search_pubmed_safe(clean_query, max_results)
                else:
                    logger.warning(f"Database {database} not available")
                
        except Exception as e:
            logger.warning(f"Live API search failed: {e}")
            # Try static datasets as fallback
            results = self._search_static_datasets(clean_query, database, max_results)
        
        # Remove duplicates and limit results
        unique_results = self._remove_duplicates(results)
        final_results = unique_results[:max_results]
        
        logger.info(f"Returning {len(final_results)} unique results")
        return final_results
    
    def _sanitize_query(self, query: str) -> str:
        """Clean and sanitize the search query"""
        # Remove special characters that might cause issues
        clean_query = re.sub(r'[^\w\s\-\.]', ' ', query)
        
        # Normalize whitespace
        clean_query = ' '.join(clean_query.split())
        
        # Limit length to avoid API issues
        if len(clean_query) > 100:
            clean_query = clean_query[:100]
        
        return clean_query.strip()
    
    def _search_arxiv_safe(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Safe arXiv search using native arxiv library"""
        results = []
        
        if not ARXIV_AVAILABLE:
            logger.warning("arxiv library not available")
            return results
        
        try:
            logger.info(f"arXiv query: '{query}'")
            
            # Use native arxiv library - much more reliable
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            papers_found = 0
            for paper in search.results():
                try:
                    formatted_paper = {
                        'title': paper.title.strip(),
                        'authors': [author.name for author in paper.authors],
                        'abstract': paper.summary.strip() if paper.summary else '',
                        'year': paper.published.year if paper.published else datetime.now().year,
                        'doi': paper.doi if paper.doi else '',
                        'source': 'arXiv',
                        'citations': 0,  # arXiv doesn't provide citation counts
                        'journal': 'arXiv preprint',
                        'url': paper.entry_id
                    }
                    
                    if formatted_paper['title'] and formatted_paper['title'] != 'No title':
                        results.append(formatted_paper)
                        papers_found += 1
                        
                        if papers_found >= max_results:
                            break
                            
                except Exception as e:
                    logger.warning(f"Failed to process arXiv paper: {e}")
                    continue
            
            logger.info(f"Successfully retrieved {len(results)} papers from arXiv")
                    
        except Exception as e:
            logger.error(f"arXiv search failed: {e}")
            
        return results
    
    def _search_pubmed_safe(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Safe PubMed search using pymed library"""
        results = []
        
        if not PUBMED_AVAILABLE:
            logger.warning("pymed library not available")
            return results
        
        try:
            logger.info(f"PubMed query: '{query}'")
            
            # Use pymed library directly
            pubmed = pymed.PubMed(tool="NLP-Tool-Learning", email="research@example.com")
            
            # Search PubMed
            query_result = pubmed.query(query, max_results=max_results)
            
            papers_found = 0
            for article in query_result:
                try:
                    # Extract article data
                    title = article.title if article.title else "No title"
                    authors = []
                    
                    if article.authors:
                        for author in article.authors:
                            if hasattr(author, 'lastname') and hasattr(author, 'firstname'):
                                if author.lastname and author.firstname:
                                    authors.append(f"{author.firstname} {author.lastname}")
                            elif hasattr(author, 'name') and author.name:
                                authors.append(author.name)
                    
                    if not authors:
                        authors = ['Unknown Author']
                    
                    abstract = article.abstract if article.abstract else ''
                    
                    # Extract publication year
                    year = datetime.now().year
                    if article.publication_date:
                        try:
                            year = int(str(article.publication_date)[:4])
                        except:
                            pass
                    
                    formatted_paper = {
                        'title': title.strip(),
                        'authors': authors[:5],  # Limit to 5 authors
                        'abstract': abstract.strip() if abstract else '',
                        'year': year,
                        'doi': article.doi if article.doi else '',
                        'source': 'PubMed',
                        'citations': 0,  # PubMed doesn't provide citation counts directly
                        'journal': article.journal if article.journal else 'PubMed publication',
                        'url': f"https://pubmed.ncbi.nlm.nih.gov/{article.pubmed_id}/" if article.pubmed_id else ''
                    }
                    
                    if formatted_paper['title'] and formatted_paper['title'] != 'No title':
                        results.append(formatted_paper)
                        papers_found += 1
                        
                        if papers_found >= max_results:
                            break
                            
                except Exception as e:
                    logger.warning(f"Failed to process PubMed article: {e}")
                    continue
            
            logger.info(f"Successfully retrieved {len(results)} papers from PubMed")
            
            # Add delay to respect rate limits
            time.sleep(1)
                    
        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            
        return results
    
# Google Scholar API removed to avoid captcha issues
# Use arXiv and PubMed APIs instead for reliable results
    
    def _search_static_datasets(self, query: str, database: str, max_results: int) -> List[Dict[str, Any]]:
        """Search static dataset files from dataset/ folder"""
        results = []
        
        # Map database names to dataset folders
        dataset_mapping = {
            "bioRxiv": "biorxiv",
            "medRxiv": "medrxiv", 
            "chemRxiv": "chemrxiv",
            "arXiv": "arxiv",
            "PubMed": "pubmed",
            "Scholar": "scholar"
        }
        
        if database == "All":
            # Search all available dataset files
            for db_name, folder_name in dataset_mapping.items():
                db_results = self._load_dataset_files(folder_name, query, max_results // len(dataset_mapping))
                results.extend(db_results)
        elif database in dataset_mapping:
            folder_name = dataset_mapping[database]
            results = self._load_dataset_files(folder_name, query, max_results)
        else:
            logger.warning(f"Static dataset not available for {database}")
        
        return results
    
    def _load_dataset_files(self, folder_name: str, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Load papers from dataset files and filter by query"""
        results = []
        dataset_path = os.path.join(self.dataset_dir, folder_name)
        
        if not os.path.exists(dataset_path):
            logger.warning(f"Dataset folder not found: {dataset_path}")
            return results
        
        # Find all JSON/JSONL files in the dataset folder
        json_files = glob.glob(os.path.join(dataset_path, "*.json*"))
        
        if not json_files:
            logger.warning(f"No dataset files found in {dataset_path}")
            return results
        
        query_terms = query.lower().split()
        
        for file_path in json_files:
            try:
                papers = self._load_jsonl(file_path)
                
                # Filter papers by query relevance
                for paper in papers:
                    if len(results) >= max_results:
                        break
                    
                    # Simple relevance check
                    title = paper.get('title', '').lower()
                    abstract = paper.get('abstract', '').lower()
                    
                    # Check if any query term matches
                    if any(term in title or term in abstract for term in query_terms):
                        formatted_paper = self._format_paper(paper, folder_name.replace('rxiv', 'Rxiv'))
                        results.append(formatted_paper)
                
                if len(results) >= max_results:
                    break
                    
            except Exception as e:
                logger.warning(f"Failed to load dataset file {file_path}: {e}")
        
        return results[:max_results]
    
    def _format_paper(self, paper: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Format paper data into consistent structure"""
        formatted = {
            'title': paper.get('title', 'No title').strip(),
            'authors': self._extract_authors(paper),
            'abstract': self._extract_abstract(paper),
            'year': self._extract_year(paper),
            'doi': paper.get('doi', '').strip(),
            'source': source,
            'citations': self._extract_citations(paper),
            'journal': self._extract_journal(paper, source),
            'url': paper.get('url', '').strip()
        }
        
        return formatted
    
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
            # Truncate very long abstracts
            if len(abstract) > 500:
                return abstract[:500] + '...'
            return abstract.strip()
        return ''
    
    def _extract_year(self, paper: Dict[str, Any]) -> int:
        """Extract publication year from paper data"""
        year_fields = ['year', 'publication_year', 'date', 'published', 'pub_date']
        
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
    
    def _extract_citations(self, paper: Dict[str, Any]) -> int:
        """Extract citation count from paper data"""
        citation_fields = ['citations', 'citation_count', 'cited_by_count']
        
        for field in citation_fields:
            if field in paper:
                citations = paper[field]
                if isinstance(citations, int) and citations >= 0:
                    return citations
                elif isinstance(citations, str) and citations.isdigit():
                    return int(citations)
        
        return 0
    
    def _extract_journal(self, paper: Dict[str, Any], source: str) -> str:
        """Extract journal/venue information"""
        journal_fields = ['journal', 'venue', 'journal_name', 'publication']
        
        for field in journal_fields:
            if field in paper and paper[field]:
                return paper[field].strip()
        
        # Default based on source
        if source == "arXiv":
            return "arXiv preprint"
        elif source == "PubMed":
            return "PubMed publication"
        elif source == "Scholar":
            return "Google Scholar publication"
        
        return f"{source} publication"
    
    def _load_jsonl(self, filepath: str) -> List[Dict[str, Any]]:
        """Load papers from JSONL file"""
        papers = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                if filepath.endswith('.jsonl'):
                    # JSONL format - one JSON object per line
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                paper = json.loads(line)
                                papers.append(paper)
                            except json.JSONDecodeError:
                                continue  # Skip malformed lines
                else:
                    # Regular JSON format
                    data = json.load(f)
                    if isinstance(data, list):
                        papers = data
                    elif isinstance(data, dict) and 'papers' in data:
                        papers = data['papers']
                    else:
                        papers = [data]
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Failed to load JSONL file {filepath}: {e}")
        
        return papers
    
    def _remove_duplicates(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate papers based on title similarity"""
        unique_papers = []
        seen_titles = set()
        
        for paper in papers:
            title = paper.get('title', '').lower().strip()
            
            # Simple deduplication by title
            if title and title not in seen_titles and title != 'no title':
                seen_titles.add(title)
                unique_papers.append(paper)
        
        return unique_papers
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search statistics"""
        available_dbs = []
        if ARXIV_AVAILABLE:
            available_dbs.append('arXiv')
        if PUBMED_AVAILABLE:
            available_dbs.append('PubMed')
        
        # Check for static datasets
        static_dbs = []
        for db_name in ['arXiv', 'PubMed', 'bioRxiv', 'medRxiv', 'chemRxiv']:
            folder_name = db_name.lower().replace('rxiv', 'rxiv')
            dataset_path = os.path.join(self.dataset_dir, folder_name)
            if os.path.exists(dataset_path) and glob.glob(os.path.join(dataset_path, "*.json*")):
                static_dbs.append(db_name)
            
        return {
            'total_searches': self.search_count,
            'arxiv_available': ARXIV_AVAILABLE,
            'pubmed_available': PUBMED_AVAILABLE,
            'scholar_available': False,  # Disabled to avoid captcha
            'live_databases': available_dbs,
            'static_databases': static_dbs,
            'total_databases': len(available_dbs) + len(static_dbs)
        }

def main():
    """Test the real paper tools"""
    print("🔍 Testing Real Paper Tools")
    print("=" * 30)
    
    # Check library availability
    missing_libs = []
    if not ARXIV_AVAILABLE:
        missing_libs.append("arxiv")
    if not PUBMED_AVAILABLE:
        missing_libs.append("pymed")
    
    if missing_libs:
        print(f"❌ Missing libraries: {', '.join(missing_libs)}")
        print("Install with: pip install arxiv pymed")
        print("Testing will continue with available libraries...")
    
    tools = PaperTools()
    
    # Test each API individually (arXiv + PubMed only)
    test_queries = []
    if ARXIV_AVAILABLE:
        test_queries.append(("machine learning", "arXiv"))
    if PUBMED_AVAILABLE:
        test_queries.append(("artificial intelligence medical", "PubMed"))
    
    if not test_queries:
        print("❌ No APIs available for testing")
        return
    
    for query, database in test_queries:
        print(f"\n🔍 Testing {database} with query: '{query}'")
        try:
            results = tools.search_papers(query, max_results=3, database=database)
            
            if results:
                print(f"✅ Found {len(results)} papers from {database}:")
                for i, paper in enumerate(results, 1):
                    print(f"  {i}. {paper['title'][:60]}...")
                    print(f"     Authors: {', '.join(paper['authors'][:2])}")
                    print(f"     Year: {paper['year']}, Source: {paper['source']}")
            else:
                print(f"❌ No results from {database}")
                
        except Exception as e:
            print(f"❌ {database} search failed: {e}")
    
    # Test combined search if multiple APIs available
    if len(test_queries) > 1:
        print(f"\n🔍 Testing combined search...")
        try:
            results = tools.search_papers("deep learning", max_results=5, database="All")
            print(f"✅ Found {len(results)} papers from all databases:")
            for i, paper in enumerate(results, 1):
                print(f"  {i}. {paper['title'][:60]}...")
                print(f"     Source: {paper['source']}")
        except Exception as e:
            print(f"❌ Combined search failed: {e}")
    
    # Show stats
    stats = tools.get_search_stats()
    print(f"\n📊 System Stats:")
    print(f"   arXiv available: {stats['arxiv_available']}")
    print(f"   PubMed available: {stats['pubmed_available']}")
    print(f"   Scholar: Disabled (avoids captcha issues)")
    print(f"   Live APIs: {stats['live_databases']}")
    print(f"   Static datasets: {stats['static_databases']}")
    print(f"   Total databases: {stats['total_databases']}")
    print(f"   Total searches: {stats['total_searches']}")

if __name__ == "__main__":
    main()