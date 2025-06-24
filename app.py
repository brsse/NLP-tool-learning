import streamlit as st
import json
import time
from datetime import datetime

# Import our custom modules
try:
    from tool_learning import ToolLearningSystem
    from model import ModelManager
    SYSTEM_AVAILABLE = True
except ImportError as e:
    st.error(f"System modules not available: {e}")
    SYSTEM_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Tool Learning System",
    page_icon="🔬",
    layout="wide"
)

def initialize_session_state():
    """Initialize session state variables"""
    if 'tool_system' not in st.session_state:
        if SYSTEM_AVAILABLE:
            st.session_state.tool_system = ToolLearningSystem(use_static_data=False)  # Default to API, user can toggle
            st.session_state.model_manager = ModelManager()
        else:
            st.session_state.tool_system = None
            st.session_state.model_manager = None
    
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []

def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    st.title("🔬 Tool Learning System")
    st.markdown("**Intelligent arXiv Paper Search** | 3,500 papers across 7 AI/ML topics")
    
    if not SYSTEM_AVAILABLE:
        st.error("System not available. Please check the installation.")
        return
    
    # Sidebar settings
    st.sidebar.title("⚙️ Settings")
    
    # Model selection
    model_manager = st.session_state.model_manager
    available_models = model_manager.get_installed_models()
    
    if available_models:
        selected_model = st.sidebar.selectbox(
            "🤖 Model",
            available_models,
            index=0,
            help="Select the LLM model for route selection and response generation"
        )
    else:
        selected_model = model_manager.default_model
        st.sidebar.warning(f"No models installed. Using default: {selected_model}")
    
    # Data source toggle
    use_static = st.sidebar.checkbox(
        "📚 Use Static Dataset",
        value=True,
        help="Use local dataset (3,500 papers) instead of live arXiv API"
    )
    
    # Search settings
    max_results = st.sidebar.slider(
        "📄 Max Results",
        min_value=1,
        max_value=15,
        value=5,
        help="Maximum number of papers to retrieve"
    )
    
    # System status
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 System Status")
    
    if st.session_state.tool_system:
        stats = st.session_state.tool_system.get_statistics()
        st.sidebar.metric("Papers Available", f"{stats.get('static_papers_count', 0):,}")
        st.sidebar.metric("Total Searches", stats.get('total_searches', 0))
        
        llm_status = "✅ Available" if stats.get('llm_available') else "❌ Offline"
        st.sidebar.text(f"LLM: {llm_status}")
        
        arxiv_status = "✅ Available" if stats.get('arxiv_available') else "❌ Offline"
        st.sidebar.text(f"arXiv API: {arxiv_status}")
    
    # Main search interface
    st.subheader("🔍 Search Research Papers")
    
    # Query input
    query = st.text_input(
        "Enter your research query:",
        placeholder="e.g., find papers about BERT",
        help="Ask about papers, authors, citations, comparisons, trends, or journals"
    )
    
    # Search button
    if st.button("🔎 Search", type="primary"):
        if query:
            perform_search(query, selected_model, max_results, use_static)
        else:
            st.warning("Please enter a search query.")
    
    # Help section
    with st.expander("ℹ️ Help & Examples", expanded=False):
        st.markdown("""
        ### Available Routes
        - **searchPapers**: Find papers on specific topics
        - **getAuthorInfo**: Get author information and publications
        - **getCitations**: Analyze citations and paper impact
        - **getRelatedPapers**: Find related research papers
        - **comparePapers**: Compare different papers or approaches
        - **trendAnalysis**: Analyze research trends over time
        - **journalAnalysis**: Analyze journals and publication venues
        
        ### Example Queries 
        - "find papers about BERT"
        - "machine learning optimization"  
        - "deep learning research"
        - "neural network architectures"
        - "who is Geoffrey Hinton"
        - "compare different optimization algorithms"
        - "trends in machine learning"
        
        ### Dataset Coverage (3,500 papers)
        - Machine Learning: optimization, algorithms, learning theory
        - Deep Learning: neural networks, architectures, training
        - BERT: BERT models, variants, applications
        - Tool Learning: agent systems, tool use
        - Plus: RAG, Hallucination detection topics
        
        ### Tips
        - Use specific terms from ML/AI domains
        - Try different phrasings if you don't get good results
        - The system works best with topics covered in the static dataset
        """)

def perform_search(query: str, model: str, max_results: int, use_static: bool):
    """Perform the search and display results"""
    tool_system = st.session_state.tool_system
    
    # Update system settings
    tool_system.ollama_model = model
    tool_system.use_static_data = use_static
    
    start_time = time.time()
    
    with st.spinner("🔄 Processing your query..."):
        try:
            # Step 1: Route selection
            route, confidence, explanation = tool_system.select_route(query)
            
            # Step 2: Paper search
            papers = tool_system.search_papers(query, max_results=max_results)
            
            # Step 3: Response generation
            response = tool_system.generate_response(query, route, papers)
            
            end_time = time.time()
            
            # Display results
            display_results(query, route, confidence, papers, response, end_time - start_time, use_static)
            
            # Update search history
            st.session_state.search_history.append({
                'query': query,
                'route': route,
                'papers_count': len(papers),
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
            
        except Exception as e:
            st.error(f"Search failed: {str(e)}")

def display_results(query: str, route: str, confidence: float, papers: list, response: str, processing_time: float, use_static: bool):
    """Display search results"""
    st.markdown("---")
    st.subheader("📋 Search Results")
    
    # Route information
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Route", route)
    with col2:
        st.metric("Confidence", f"{confidence:.3f}")
    with col3:
        st.metric("Papers Found", len(papers))
    with col4:
        st.metric("Time", f"{processing_time:.2f}s")
    
    # Data source indicator
    source_text = "📚 Static Dataset" if use_static else "🌐 Live arXiv API"
    st.info(f"Data source: {source_text}")
    
    # AI Response
    st.subheader("🤖 AI Response")
    st.markdown(response)
    
    # Papers found
    if papers:
        st.subheader("📚 Found Papers")
        
        for i, paper in enumerate(papers, 1):
            with st.expander(f"📄 Paper {i}: {paper['title']}", expanded=i <= 3):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Authors:** {', '.join(paper['authors'][:5])}")
                    st.markdown(f"**Year:** {paper['year']}")
                    if paper.get('abstract'):
                        st.markdown(f"**Abstract:** {paper['abstract'][:300]}...")
                
                with col2:
                    st.markdown(f"**Source:** {paper.get('source', 'Unknown')}")
                    if paper.get('arxiv_id'):
                        st.markdown(f"**arXiv ID:** {paper['arxiv_id']}")
                    if paper.get('categories'):
                        st.markdown(f"**Categories:** {', '.join(paper['categories'][:3])}")
    else:
        st.warning("No papers found. Try different keywords or rephrase your query.")
    
    # Export option
    if st.button("📁 Export Results as JSON"):
        results = {
            'query': query,
            'route': route,
            'confidence': confidence,
            'papers_found': len(papers),
            'papers': papers,
            'response': response,
            'data_source': 'static' if use_static else 'live_api',
            'timestamp': datetime.now().isoformat()
        }
        
        json_str = json.dumps(results, indent=2, ensure_ascii=False)
        st.download_button(
            label="⬇️ Download JSON",
            data=json_str,
            file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main() 