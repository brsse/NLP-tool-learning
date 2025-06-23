#!/usr/bin/env python3
"""
Prompt Engineering for Tool Learning Chatbot

Contains all prompts and templates for generating intelligent responses
based on tool routes and paper search results.
"""

from typing import Dict, List, Any, Optional

class PromptTemplates:
    """Collection of prompt templates for tool learning chatbot"""
    
    BASE_SYSTEM_PROMPT = """You are a research assistant with tool learning capabilities. You help users find and understand research papers.

User Query: "{query}"

Tool Selected: {route}
Route Explanation: {route_explanation}

"""
    
    PAPER_RESULTS_TEMPLATE = """FOUND {count} PAPER(S):
{paper_details}

IMPORTANT: Base your response primarily on these specific papers found. Mention titles, years, and details from the papers above.
"""
    
    NO_PAPERS_TEMPLATE = "No specific papers found for this query.\n"
    
    ROUTE_INSTRUCTIONS = {
        "searchPapers": "\nProvide a helpful summary of the found papers. Include specific titles, years, and key contributions from each paper.",
        
        "getAuthorInfo": "\nProvide information about the author based on their specific papers listed above. Mention each paper title, year, and the author's contribution. Focus on the actual research found.",
        
        "getCitations": "\nExplain the citation impact and significance of the specific papers found. Include citation counts and years.",
        
        "comparePapers": "\nCompare the approaches, methodologies, and findings of the specific papers found above.",
        
        "trendAnalysis": "\nAnalyze research trends based on the specific papers found, noting years and developments over time.",
        
        "journalAnalysis": "\nProvide information about publication venues based on the papers found.",
        
        "getRelatedPapers": "\nExplain how the found papers relate to each other and suggest additional research directions."
    }
    
    RESPONSE_INSTRUCTION = "\n\nProvide a concise, informative response (2-3 paragraphs maximum):"
    
    FALLBACK_RESPONSE_TEMPLATE = """**Route Selected**: {route}

**Explanation**: {route_explanation}

{paper_content}

💡 *For enhanced responses, please ensure Ollama is running with a compatible model.*"""

    @classmethod
    def build_prompt(cls, query: str, route: str, route_explanation: str, 
                    paper_results: Optional[List[Dict[str, Any]]] = None) -> str:
        """Build context-aware prompt for Ollama"""
        
        # Start with base prompt
        prompt = cls.BASE_SYSTEM_PROMPT.format(
            query=query,
            route=route,
            route_explanation=route_explanation
        )
        
        # Add paper results if available
        if paper_results and len(paper_results) > 0:
            paper_details = ""
            for i, paper in enumerate(paper_results[:3], 1):
                title = paper.get('title', 'No title')
                authors = ', '.join(paper.get('authors', ['Unknown'])[:3])
                year = paper.get('year', 'Unknown')
                citations = paper.get('citations', 0)
                abstract = paper.get('abstract', '')[:200] + '...' if paper.get('abstract') else 'No abstract'
                
                paper_details += f"""
PAPER {i}: "{title}" ({year})
Authors: {authors}
Citations: {citations}
Abstract: {abstract}
"""
            
            prompt += cls.PAPER_RESULTS_TEMPLATE.format(
                count=len(paper_results),
                paper_details=paper_details
            )
        else:
            prompt += cls.NO_PAPERS_TEMPLATE
        
        # Add route-specific instructions
        prompt += cls.ROUTE_INSTRUCTIONS.get(route, "")
        
        # Add final instruction
        prompt += cls.RESPONSE_INSTRUCTION
        
        return prompt
    
    @classmethod
    def build_fallback_response(cls, query: str, route: str, route_explanation: str, 
                               paper_results: Optional[List[Dict[str, Any]]] = None) -> str:
        """Generate fallback response when Ollama is not available"""
        
        if paper_results and len(paper_results) > 0:
            paper_content = f"**Found {len(paper_results)} papers:**\n\n"
            for i, paper in enumerate(paper_results[:3], 1):
                title = paper.get('title', 'No title')
                authors = ', '.join(paper.get('authors', ['Unknown'])[:2])
                year = paper.get('year', 'Unknown')
                paper_content += f"{i}. **{title}** ({year})\n   Authors: {authors}\n\n"
        else:
            paper_content = "**Note**: No papers found or paper search not available.\n\n"
        
        return cls.FALLBACK_RESPONSE_TEMPLATE.format(
            route=route,
            route_explanation=route_explanation,
            paper_content=paper_content
        )

class ConversationPrompts:
    """Prompts for maintaining conversation context"""
    
    CONTEXT_PROMPT = """Previous conversation context:
{context}

Continue the conversation naturally while maintaining context about the research topic."""
    
    FOLLOW_UP_PROMPTS = {
        "searchPapers": "Would you like me to search for related papers or get more details about any specific paper?",
        "getAuthorInfo": "Would you like to explore more papers by this author or get citation analysis?",
        "getCitations": "Would you like to compare these papers or explore related work?",
        "comparePapers": "Would you like to dive deeper into any specific comparison aspect?",
        "trendAnalysis": "Would you like to explore specific time periods or related research areas?",
        "journalAnalysis": "Would you like information about specific venues or publication strategies?",
        "getRelatedPapers": "Would you like to explore any of these related areas in more detail?"
    }
    
    @classmethod
    def get_follow_up(cls, route: str) -> str:
        """Get a follow-up question based on the route"""
        return cls.FOLLOW_UP_PROMPTS.get(route, "How else can I help with your research?")

class SystemMessages:
    """System messages for different scenarios"""
    
    ERROR_MESSAGES = {
        "api_error": "❌ Error connecting to research databases. Please try again later.",
        "parsing_error": "❌ Error processing your query. Please rephrase and try again.",
        "no_results": "ℹ️ No papers found for your query. Try different keywords or broader terms.",
        "ollama_offline": "⚠️ AI response generation is offline. Using fallback mode.",
        "timeout_error": "⏱️ Request timed out. Please try again with a more specific query."
    }
    
    SUCCESS_MESSAGES = {
        "search_complete": "✅ Search completed successfully!",
        "analysis_complete": "✅ Analysis completed!",
        "route_selected": "🛤️ Route selected: {route}"
    }
    
    @classmethod
    def get_error_message(cls, error_type: str, custom_message: str = None) -> str:
        """Get formatted error message"""
        base_message = cls.ERROR_MESSAGES.get(error_type, "❌ An error occurred.")
        if custom_message:
            return f"{base_message}\n\nDetails: {custom_message}"
        return base_message
    
    @classmethod
    def get_success_message(cls, success_type: str, **kwargs) -> str:
        """Get formatted success message"""
        message = cls.SUCCESS_MESSAGES.get(success_type, "✅ Operation completed!")
        return message.format(**kwargs) 