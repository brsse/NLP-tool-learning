# ============================================================================
# ALL PROMPTS FOR TOOL LEARNING SYSTEM
# ============================================================================

# Route Selection Prompt
ROUTE_SELECTION_PROMPT = """You are a route selector for a research paper tool learning system.

Available routes:
- searchPapers: Find research papers on specific topics
- getAuthorInfo: Get author information and their publications  
- getCitations: Analyze citations and paper impact
- getRelatedPapers: Find related research papers
- comparePapers: Compare different papers or methods
- trendAnalysis: Analyze research trends over time
- journalAnalysis: Analyze journals and venues

Query: "{query}"

Select the most appropriate route(s). You can select one or multiple routes if the query would benefit from multiple approaches.
Respond with route names separated by commas (e.g., "searchPapers" or "searchPapers, getAuthorInfo")."""

# Response Generation Prompt  
RESPONSE_GENERATION_PROMPT = """You are a helpful research assistant. Answer the user's query using the provided papers.

Selected Routes: {routes}
Query: {query}

{paper_context}

Provide a comprehensive response that addresses the query using the selected routes. Organize your response to cover all relevant aspects."""

# No Papers Found Response
RESPONSE_NO_PAPERS = """I searched for papers related to "{query}" but didn't find any relevant results in the dataset.

Try rephrasing your query or using different keywords."""

# Fallback Response Template
FALLBACK_RESPONSE_TEMPLATE = """Based on your query '{query}' using routes [{routes}], I found {papers_count} relevant papers:

{papers_list}

{additional_papers_note}"""

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_papers_for_context(papers: list, max_papers: int = 5) -> str:
    """Format papers into readable context for LLM prompts"""
    if not papers:
        return "No papers found."
    
    context = "\n\nRelevant papers found:\n"
    for i, paper in enumerate(papers[:max_papers], 1):
        context += f"{i}. {paper.get('title', 'Unknown Title')}\n"
        context += f"   Authors: {', '.join(paper.get('authors', ['Unknown'])[:3])}\n"
        context += f"   Year: {paper.get('year', 'Unknown')}\n"
        if paper.get('abstract'):
            context += f"   Abstract: {paper['abstract'][:200]}...\n"
        context += "\n"
    
    return context

def format_papers_for_fallback(papers: list, max_display: int = 5) -> str:
    """Format papers for fallback response when LLM is not available"""
    if not papers:
        return ""
    
    papers_list = ""
    for i, paper in enumerate(papers[:max_display], 1):
        papers_list += f"\n{i}. **{paper.get('title', 'Unknown Title')}**\n"
        papers_list += f"   Authors: {', '.join(paper.get('authors', ['Unknown'])[:3])}\n"
        papers_list += f"   Year: {paper.get('year', 'Unknown')}\n"
        if paper.get('abstract'):
            papers_list += f"   Abstract: {paper['abstract'][:150]}...\n"
        papers_list += "\n"
    
    return papers_list

def get_route_selection_prompt(query: str) -> str:
    """Get formatted route selection prompt"""
    return ROUTE_SELECTION_PROMPT.format(query=query)

def get_response_generation_prompt(routes: list, query: str, papers: list) -> str:
    """Get formatted response generation prompt"""
    if not papers:
        return RESPONSE_NO_PAPERS.format(query=query)
    
    routes_str = ', '.join(routes) if isinstance(routes, list) else str(routes)
    paper_context = format_papers_for_context(papers)
    
    return RESPONSE_GENERATION_PROMPT.format(
        routes=routes_str, 
        query=query, 
        paper_context=paper_context
    )

def get_fallback_response(query: str, routes: list, papers: list) -> str:
    """Get formatted fallback response when LLM is not available"""
    if not papers:
        return f"I searched for papers related to '{query}' but didn't find any relevant results in the dataset."
    
    routes_str = ', '.join(routes) if isinstance(routes, list) else str(routes)
    papers_list = format_papers_for_fallback(papers)
    
    additional_papers_note = ""
    if len(papers) > 5:
        additional_papers_note = f"...and {len(papers) - 5} more papers."
    
    return FALLBACK_RESPONSE_TEMPLATE.format(
        query=query,
        routes=routes_str,
        papers_count=len(papers),
        papers_list=papers_list,
        additional_papers_note=additional_papers_note
    )

# ============================================================================
# LEGACY FUNCTIONS (for backward compatibility)
# ============================================================================

def format_papers(papers: list, max_papers: int = 5) -> str:
    """Legacy function - use format_papers_for_context instead"""
    return format_papers_for_context(papers, max_papers)

def get_selection_prompt(query: str) -> str:
    """Legacy function - use get_route_selection_prompt instead"""
    return get_route_selection_prompt(query)

def get_response_prompt(routes: list, query: str, papers: list) -> str:
    """Legacy function - use get_response_generation_prompt instead"""
    return get_response_generation_prompt(routes, query, papers) 