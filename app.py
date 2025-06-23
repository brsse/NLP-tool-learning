#!/usr/bin/env python3
"""
Simple Tool Learning Chatbot for Research Papers

A Streamlit chatbot that learns to route user queries to appropriate tools
for research paper search and analysis.
"""

import streamlit as st
import json
from datetime import datetime
from tool_learning_engine import ToolLearningEngine
from paper_tools import PaperTools
from ollama_client import OllamaClient

# Page config
st.set_page_config(
    page_title="Tool Learning Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'engine' not in st.session_state:
    st.session_state.engine = ToolLearningEngine()
if 'paper_tools' not in st.session_state:
    try:
        # Use live APIs by default (toggle available in sidebar)
        st.session_state.paper_tools = PaperTools(use_static_data=False)
    except ImportError as e:
        st.error(f"Paper search not available: {e}")
        st.session_state.paper_tools = None
    except Exception as e:
        st.error(f"Error initializing paper tools: {e}")
        st.session_state.paper_tools = None

if 'ollama_client' not in st.session_state:
    st.session_state.ollama_client = OllamaClient()

def main():
    st.title("🤖 Tool Learning Chatbot")
    st.markdown("*Learn to route research paper queries to the right tools*")
    
    # Sidebar with info
    with st.sidebar:
        st.header("📋 Available Tools")
        routes = st.session_state.engine.get_available_routes()
        for route in routes:
            st.write(f"• {route}")
        
        st.header("🤖 AI Engine")
        ollama_status = "🟢 Connected" if st.session_state.ollama_client.available else "🔴 Offline"
        st.write(f"Ollama: {ollama_status}")
        st.write(f"Model: {st.session_state.ollama_client.model}")
        if st.session_state.ollama_client.available:
            model_available = st.session_state.ollama_client.check_model_available()
            st.write(f"Model Status: {'✅ Ready' if model_available else '⚠️ Not Found'}")
        
        st.header("📊 Data Source")
        if st.session_state.paper_tools:
            current_mode = "Static Data" if st.session_state.paper_tools.use_static_data else "Live APIs"
            st.write(f"Mode: {current_mode}")
            
            # Show available APIs/datasets
            stats = st.session_state.paper_tools.get_search_stats()
            if not st.session_state.paper_tools.use_static_data:
                st.write("**Available APIs:**")
                for api in stats['live_databases']:
                    st.write(f"✅ {api}")
            else:
                st.write(f"**Static Datasets:** {len(stats['static_databases'])}")
            
            # Toggle between static and live mode
            use_static = st.toggle("Use Static Data", value=st.session_state.paper_tools.use_static_data, 
                                 help="Toggle between live APIs (arXiv, PubMed) and static dataset files")
            if use_static != st.session_state.paper_tools.use_static_data:
                try:
                    st.session_state.paper_tools = PaperTools(use_static_data=use_static)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error switching mode: {e}")
        
        st.header("📊 Session Stats")
        st.metric("Messages", len(st.session_state.messages))
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Chat interface
    st.header("💬 Chat")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "route_info" in message:
                st.caption(f"🛤️ Route: {message['route_info']['route']} (confidence: {message['route_info']['confidence']:.3f})")
    
    # Chat input
    if prompt := st.chat_input("Ask about research papers..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Process with tool learning engine
        route, confidence, explanation = st.session_state.engine.select_route(prompt)
        
        # Execute the selected tool and generate response with Ollama
        paper_results = None
        try:
            if st.session_state.paper_tools and route in ["searchPapers", "getAuthorInfo", "getCitations"]:
                with st.spinner("Searching for papers..."):
                    if route == "searchPapers":
                        paper_results = st.session_state.paper_tools.search_papers(prompt, max_results=5, database="All")
                    elif route == "getAuthorInfo":
                        author = extract_author_name(prompt)
                        paper_results = st.session_state.paper_tools.search_papers(author, max_results=5, database="All")
                    elif route == "getCitations":
                        paper_results = st.session_state.paper_tools.search_papers(prompt, max_results=3, database="Scholar")
            
            # Generate response using Ollama
            with st.spinner("Generating response..."):
                response = st.session_state.ollama_client.generate_response(
                    query=prompt,
                    route=route,
                    route_explanation=explanation,
                    paper_results=paper_results
                )
                
        except Exception as e:
            response = f"❌ Error executing {route}: {str(e)}\n\nPlease try a different query or check your internet connection."
        
        # Add assistant response
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "route_info": {
                "route": route,
                "confidence": confidence,
                "explanation": explanation
            }
        })
        
        st.rerun()

def format_search_results(results, query):
    """Format search results for display"""
    if not results:
        return f"No papers found for '{query}'"
    
    response = f"Found {len(results)} papers for '{query}':\n\n"
    for i, paper in enumerate(results[:5], 1):
        title = paper.get('title', 'No title')
        authors = ', '.join(paper.get('authors', ['Unknown']))
        year = paper.get('year', 'Unknown')
        response += f"{i}. **{title}**\n   Authors: {authors}\n   Year: {year}\n\n"
    
    return response

def format_author_results(results, author):
    """Format author search results"""
    if not results:
        return f"No papers found for author '{author}'"
    
    response = f"Papers by '{author}':\n\n"
    for i, paper in enumerate(results[:5], 1):
        title = paper.get('title', 'No title')
        year = paper.get('year', 'Unknown')
        citations = paper.get('citations', 0)
        response += f"{i}. **{title}** ({year})\n   Citations: {citations}\n\n"
    
    return response

def format_citation_results(results, query):
    """Format citation search results"""
    if not results:
        return f"No citation information found for '{query}'"
    
    response = f"Citation information for '{query}':\n\n"
    for i, paper in enumerate(results[:3], 1):
        title = paper.get('title', 'No title')
        citations = paper.get('citations', 0)
        year = paper.get('year', 'Unknown')
        response += f"{i}. **{title}** ({year})\n   Citations: {citations}\n\n"
    
    return response

def extract_author_name(query):
    """Extract author name from query"""
    query_lower = query.lower()
    # Simple extraction - look for patterns like "papers by X" or "author X"
    if "by " in query_lower:
        return query_lower.split("by ")[-1].strip()
    elif "author " in query_lower:
        return query_lower.split("author ")[-1].strip()
    else:
        return query.strip()

if __name__ == "__main__":
    main() 