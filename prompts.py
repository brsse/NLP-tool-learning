# Route Selection
ROUTE_SELECTION_PROMPT = """You are a route selector for a research paper system.

Available routes:
- searchPapers: Find research papers on topics
- getAuthorInfo: Get author information  
- getCitations: Analyze citations and impact
- getRelatedPapers: Find related research
- comparePapers: Compare different approaches
- trendAnalysis: Analyze research trends
- journalAnalysis: Analyze journals and venues

Query: "{query}"

Select the SINGLE best route. Respond with only the route name."""

# Response Generation
RESPONSE_GENERATION_PROMPT = """You are a research assistant. Answer the user's query using the provided papers.

Route: {route}
Query: {query}

Papers found:
{paper_context}

Provide a helpful response based on the papers."""

# No Papers Found
RESPONSE_NO_PAPERS = """I searched for papers related to "{query}" but didn't find relevant results.

Try rephrasing your query or using different keywords."""

# Helper Functions
def format_papers(papers: list, max_papers: int = 3) -> str:
    """Format papers into readable context"""
    if not papers:
        return "No papers found."
    
    context = ""
    for i, paper in enumerate(papers[:max_papers], 1):
        context += f"{i}. {paper.get('title', 'Unknown Title')}\n"
        context += f"   Authors: {', '.join(paper.get('authors', ['Unknown'])[:3])}\n"
        
        abstract = paper.get('abstract', '')
        if abstract:
            context += f"   Abstract: {abstract[:150]}...\n"
        context += "\n"
    
    return context

def get_selection_prompt(query: str) -> str:
    """Get route selection prompt"""
    return ROUTE_SELECTION_PROMPT.format(query=query)

def get_response_prompt(route: str, query: str, papers: list) -> str:
    """Get response generation prompt"""
    if not papers:
        return RESPONSE_NO_PAPERS.format(query=query)
    
    paper_context = format_papers(papers)
    return RESPONSE_GENERATION_PROMPT.format(route=route, query=query, paper_context=paper_context) 