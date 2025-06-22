import streamlit as st
import time
from tool_agent import run_advanced_agent
from prompts import ROUTE_CONFIGS

# Streamlit config
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced professional styling
st.markdown("""
<style>
/* Import SF Pro Display font */
@import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700&display=swap');

/* Global styling */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
    font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Chat message styling */
.user-message {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    color: white;
    padding: 16px 20px;
    border-radius: 18px;
    margin: 10px 0;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    font-weight: 500;
    line-height: 1.5;
}

.bot-message {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.8) 100%);
    color: #f1f5f9;
    padding: 20px;
    border-radius: 16px;
    margin: 10px 0;
    border: 1px solid rgba(148, 163, 184, 0.2);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    line-height: 1.6;
}

/* Sidebar styling */
.sidebar-metric {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(51, 65, 85, 0.6) 100%);
    padding: 16px;
    border-radius: 12px;
    margin: 12px 0;
    border: 1px solid rgba(148, 163, 184, 0.2);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.sidebar-metric h3 {
    color: #60a5fa;
    font-size: 16px;
    font-weight: 600;
    margin: 0 0 8px 0;
}

/* Button styling */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 24px;
    font-size: 14px;
    font-weight: 500;
    transition: all 0.2s ease;
    width: 100%;
    text-align: left;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb 0%, #4338ca 100%);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
}

/* Clean title */
.main-title {
    color: #f8fafc;
    font-size: 32px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 8px;
}

.subtitle {
    color: #94a3b8;
    font-size: 16px;
    text-align: center;
    margin-bottom: 30px;
}

/* Input styling */
.stChatInput > div {
    background-color: rgba(30, 41, 59, 0.8) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(148, 163, 184, 0.3) !important;
}

/* Hide streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {visibility: hidden;}

/* Loading message styling */
.loading-message {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px;
    margin: 10px 0;
    text-align: center;
    color: #60a5fa;
    font-weight: 500;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 0.8; }
    50% { opacity: 1; }
}

/* Clean route display */
.route-display {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(148, 163, 184, 0.3);
    border-radius: 8px;
    padding: 12px;
    margin: 8px 0;
    font-family: 'SF Mono', monospace;
    font-size: 13px;
    color: #60a5fa;
    word-wrap: break-word;
    overflow-wrap: break-word;
}

.route-arrow {
    color: #10b981;
    margin: 0 4px;
    font-weight: bold;
}

/* Enhanced agent reasoning styling - COMPACT & FIXED */
.reasoning-container {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.9) 100%);
    border: 1px solid rgba(148, 163, 184, 0.3);
    border-radius: 12px;
    margin: 12px 0;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    max-height: 400px;
    overflow-y: auto;
}

.reasoning-header {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    color: #60a5fa;
    padding: 12px 16px;
    font-weight: 600;
    font-size: 14px;
    border-bottom: 2px solid #334155;
    display: flex;
    align-items: center;
    justify-content: space-between;
    cursor: pointer;
    position: sticky;
    top: 0;
    z-index: 10;
}

.reasoning-header:hover {
    background: linear-gradient(135deg, #334155 0%, #475569 100%);
}

.reasoning-content {
    padding: 16px;
    color: #e2e8f0;
    line-height: 1.5;
    font-size: 13px;
}

.reasoning-section {
    background: rgba(30, 41, 59, 0.6);
    border-radius: 8px;
    padding: 12px;
    margin: 8px 0;
    border-left: 4px solid #60a5fa;
    font-size: 13px;
    word-wrap: break-word;
    overflow-wrap: break-word;
    white-space: pre-wrap;
    max-width: 100%;
}

.reasoning-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin: 12px 0;
}

.reasoning-metric {
    background: rgba(51, 65, 85, 0.6);
    padding: 10px;
    border-radius: 6px;
    border: 1px solid rgba(148, 163, 184, 0.2);
    font-size: 12px;
}

.reasoning-metric strong {
    color: #60a5fa;
    font-size: 12px;
    display: block;
    margin-bottom: 4px;
}

.step-execution {
    background: rgba(51, 65, 85, 0.4);
    border-radius: 6px;
    padding: 8px 10px;
    margin: 6px 0;
    font-family: 'SF Mono', monospace;
    font-size: 12px;
    border-left: 3px solid #60a5fa;
    word-wrap: break-word;
    overflow-wrap: break-word;
    line-height: 1.4;
}

.step-success {
    color: #10b981;
    font-weight: 600;
}

.step-error {
    color: #ef4444;
    font-weight: 600;
}

.confidence-analysis {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(16, 185, 129, 0.1) 100%);
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 8px;
    padding: 12px;
    margin: 10px 0;
    font-size: 12px;
}

/* Professional metrics */
.confidence-high { color: #10b981; }
.confidence-medium { color: #f59e0b; }
.confidence-low { color: #ef4444; }

/* Collapsible functionality */
.reasoning-collapsed .reasoning-content {
    display: none;
}

.reasoning-expanded .reasoning-content {
    display: block;
}

/* Responsive design */
@media (max-width: 768px) {
    .reasoning-grid {
        grid-template-columns: 1fr;
    }
    
    .reasoning-section {
        font-size: 12px;
        padding: 10px;
    }
    
    .reasoning-content {
        padding: 12px;
        font-size: 12px;
    }
    
    .reasoning-container {
        max-height: 300px;
    }
}

/* Scrollbar styling for reasoning container */
.reasoning-container::-webkit-scrollbar {
    width: 6px;
}

.reasoning-container::-webkit-scrollbar-track {
    background: rgba(30, 41, 59, 0.5);
    border-radius: 3px;
}

.reasoning-container::-webkit-scrollbar-thumb {
    background: rgba(96, 165, 250, 0.5);
    border-radius: 3px;
}

.reasoning-container::-webkit-scrollbar-thumb:hover {
    background: rgba(96, 165, 250, 0.7);
}
</style>
""", unsafe_allow_html=True)

# Professional title
st.markdown('<h1 class="main-title">🤖 AI Research Assistant</h1>', unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "show_thought_process" not in st.session_state:
    st.session_state["show_thought_process"] = False
if "is_processing" not in st.session_state:
    st.session_state["is_processing"] = False
if "reasoning_expanded" not in st.session_state:
    st.session_state["reasoning_expanded"] = {}

# Professional sidebar
with st.sidebar:
    st.markdown("## 🎯 Intelligence Hub")
    
    # Toggle for thought process
    st.session_state["show_thought_process"] = st.checkbox("Show Agent Reasoning", value=st.session_state.get("show_thought_process", False))
    
    # Show current agent state if available
    if "current_state" in st.session_state and not st.session_state.get("is_processing", False):
        state = st.session_state["current_state"]
        
        # Confidence metric
        confidence = state.get("confidence", 0.0)
        confidence_class = "confidence-high" if confidence > 0.8 else "confidence-medium" if confidence > 0.5 else "confidence-low"
        st.markdown(f"""
        <div class="sidebar-metric">
            <h3>🎯 Confidence</h3>
            <p class="{confidence_class}" style="font-size: 20px; font-weight: bold;">
                {confidence:.0%}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced route information
        route = state.get("route", "Unknown")
        if route and route != "Unknown":
            route_config = ROUTE_CONFIGS.get(route, {})
            actions = route_config.get("actions", [])
            data_types = route_config.get("data_types", [])
            description = route_config.get("description", "")
            
            actions_display = " → ".join(actions[:3]) if actions else "Standard processing"
            data_display = ", ".join(data_types[:3]) if data_types else "Various data"
            
            # Display route with clean formatting
            st.markdown(f"""
            <div class="sidebar-metric">
                <h3>🛣️ Execution Route</h3>
                <div class="route-display">
                    {route.replace(' → ', ' <span class="route-arrow">→</span> ')}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display actions with proper spacing
            if actions:
                st.markdown("**⚙️ Actions:**")
                st.markdown(f"`{actions_display}`", unsafe_allow_html=True)
            
            # Display data types with proper spacing
            if data_types:
                st.markdown("**📊 Data:**")
                st.markdown(f"`{data_display}`", unsafe_allow_html=True)
            
            # Display description
            if description:
                st.markdown(f"*{description}*")
        
        # Performance stats
        execution_metadata = state.get("execution_metadata", {})
        if execution_metadata:
            total_time = execution_metadata.get("total_time", 0)
            steps_executed = execution_metadata.get("steps_executed", 0)
            successful_steps = execution_metadata.get("successful_steps", 0)
            
            st.markdown(f"""
            <div class="sidebar-metric">
                <h3>⚡ Performance</h3>
                <p>⏱️ Response Time: {total_time:.1f}s</p>
                <p>🔧 Steps Executed: {steps_executed}</p>
                <p>✅ Success Rate: {successful_steps}/{steps_executed}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Quick examples covering ALL routes
    st.markdown("## 💡 Example Queries")
    
    # Examples designed to cover all possible routes
    example_queries = [
        ("🌐 What's new in AI this month", "web_search"),
        ("📚 Recent federated learning research papers", "research → analysis"), 
        ("🤖 Best text generation models on Hugging Face", "model_collection"),
        ("📊 Sentiment analysis datasets on Hugging Face", "dataset_collection"),
        ("🔍 Comprehensive AI landscape for computer vision", "data_collection → research → analysis"),
        ("🧠 Latest GPT developments and research", "web_search → research → analysis")
    ]
    
    for query_text, route_info in example_queries:
        if st.button(f"{query_text}", key=f"example_{hash(query_text)}", help=f"Route: {route_info}"):
            query_without_emoji = query_text.split(" ", 1)[1]  # Remove emoji
            # Immediately add to chat history and trigger processing
            st.session_state["chat_history"].append({
                "type": "user",
                "content": query_without_emoji,
                "timestamp": time.time()
            })
            st.session_state["process_query"] = query_without_emoji
            st.rerun()

# Main chat area
st.markdown("---")

# Display chat history FIRST - before any processing logic
if st.session_state["chat_history"]:
    for i, message in enumerate(st.session_state["chat_history"]):
        if message["type"] == "user":
            st.markdown(f"""
            <div class="user-message">
                🧑‍💻 {message["content"]}
            </div>
            """, unsafe_allow_html=True)
            
        elif message["type"] == "loading":
            st.markdown(f"""
            <div class="loading-message">
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
            
        else:  # bot message
            st.markdown(f"""
            <div class="bot-message">
                🤖 {message["content"]}
            </div>
            """, unsafe_allow_html=True)
            
            # Show enhanced agent reasoning if enabled - COMPACT & COLLAPSIBLE
            if st.session_state.get("show_thought_process", False) and "metadata" in message and not message["metadata"].get("error", False):
                metadata = message["metadata"]
                message_id = f"reasoning_{i}"
                
                # Initialize expansion state for this message
                if message_id not in st.session_state["reasoning_expanded"]:
                    st.session_state["reasoning_expanded"][message_id] = False
                
                # Toggle button for collapsible reasoning
                col1, col2 = st.columns([1, 20])
                with col1:
                    if st.button("🔽" if st.session_state["reasoning_expanded"][message_id] else "▶️", 
                               key=f"toggle_{message_id}", help="Toggle reasoning details"):
                        st.session_state["reasoning_expanded"][message_id] = not st.session_state["reasoning_expanded"][message_id]
                        st.rerun()
                
                with col2:
                    st.markdown(f"""
                    <div class="reasoning-header" style="margin-bottom: 0;">
                        🧠 Agent Intelligence Breakdown
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show content only if expanded
                if st.session_state["reasoning_expanded"][message_id]:
                    # Professional reasoning container
                    st.markdown(f"""
                    <div class="reasoning-container">
                        <div class="reasoning-content">
                    """, unsafe_allow_html=True)
                    
                    # Execution Analysis Section
                    route = metadata.get("route", "unknown")
                    confidence = metadata.get("confidence", 0.0)
                    total_time = metadata.get("execution_metadata", {}).get("total_time", 0)
                    
                    st.markdown(f"""
                        <div class="reasoning-section">
                            <strong style="color: #60a5fa;">🔍 Execution Analysis & Decision Process</strong><br><br>
                            <strong>🎯 Query Analysis:</strong> Determined optimal route based on intent detection<br>
                            <strong>🛣️ Route Selection:</strong> {route}<br>
                            <strong>⚡ Execution Speed:</strong> {total_time:.1f}s<br>
                            <strong>🎯 Result Quality:</strong> {confidence:.0%} confidence
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Route and Performance Grid
                    tool_executions = metadata.get("execution_metadata", {}).get("tool_executions", [])
                    if not tool_executions:
                        tool_executions = metadata.get("tool_executions", [])
                    
                    if tool_executions:
                        total_exec_time = sum(getattr(exec, 'execution_time', 0) if hasattr(exec, 'execution_time') else exec.get("execution_time", 0) for exec in tool_executions)
                        success_rate = sum(1 for exec in tool_executions if (getattr(exec, 'success', False) if hasattr(exec, 'success') else exec.get("success", False))) / len(tool_executions)
                        
                        st.markdown(f"""
                        <div class="reasoning-grid">
                            <div class="reasoning-metric">
                                <strong>📊 Performance Summary</strong>
                                Total execution: {total_exec_time:.2f}s<br>
                                Success rate: {success_rate:.0%}<br>
                                Tools used: {len(tool_executions)}
                            </div>
                            <div class="reasoning-metric">
                                <strong>🛣️ Route Analysis</strong>
                                Path: {route}<br>
                                Steps: {len(tool_executions)}<br>
                                Type: {'Multi-step' if '→' in route else 'Single-step'}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Detailed Tool Execution
                        st.markdown('<strong style="color: #60a5fa;">🔧 Tool Execution Details:</strong>', unsafe_allow_html=True)
                        
                        for i, tool_exec in enumerate(tool_executions):
                            tool_name = getattr(tool_exec, 'tool_name', None) if hasattr(tool_exec, 'tool_name') else tool_exec.get("tool_name", tool_exec.get("tool", "Unknown"))
                            success = getattr(tool_exec, 'success', False) if hasattr(tool_exec, 'success') else tool_exec.get("success", False)
                            exec_time = getattr(tool_exec, 'execution_time', 0) if hasattr(tool_exec, 'execution_time') else tool_exec.get("execution_time", 0)
                            
                            status_class = "step-success" if success else "step-error"
                            status_icon = "✅" if success else "❌"
                            
                            # Add output summary if available
                            output_summary = ""
                            output = getattr(tool_exec, 'output', None) if hasattr(tool_exec, 'output') else tool_exec.get("output", None)
                            if success and output:
                                if isinstance(output, dict):
                                    if "models" in output:
                                        output_summary = f" → Found {len(output['models'])} models"
                                    elif "datasets" in output:
                                        output_summary = f" → Found {len(output['datasets'])} datasets"
                                    elif "papers" in output:
                                        output_summary = f" → Found {len(output['papers'])} papers"
                            
                            st.markdown(f"""
                            <div class="step-execution">
                                <span class="{status_class}">{status_icon} Step {i+1}:</span> {tool_name} 
                                <span style="color: #94a3b8;">({exec_time:.2f}s{output_summary})</span>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Confidence Analysis
                    confidence_class = "step-success" if confidence > 0.8 else "step-error" if confidence < 0.5 else "color: #f59e0b"
                    confidence_desc = 'Reliable results' if confidence > 0.8 else 'Good results with minor gaps' if confidence > 0.5 else 'Consider refining query'
                    quality = 'High' if confidence > 0.8 else 'Medium' if confidence > 0.5 else 'Low'
                    
                    st.markdown(f"""
                        <div class="confidence-analysis">
                            <strong style="color: #60a5fa;">🎯 Confidence Analysis</strong><br><br>
                            <strong>Response confidence:</strong> <span style="{confidence_class}">{confidence:.0%}</span><br>
                            <strong>Data quality:</strong> {quality}<br>
                            <strong>Recommendation:</strong> {confidence_desc}
                        </div>
                    </div>
                    </div>
                    """, unsafe_allow_html=True)

else:
    # Welcome message
    st.markdown("""
    <div style="text-align: center; margin: 40px 0; color: #94a3b8;">
        <h2 style="color: #f1f5f9;">👋 Ready to assist with your research</h2>
        <p>I can search Hugging Face models & datasets, research academic papers, analyze trends, and more!</p>
        <p>Select an example query or ask me anything about AI, ML, or data science.</p>
    </div>
    """, unsafe_allow_html=True)

# Chat input
user_input = st.chat_input("Ask about AI research, models, datasets, or latest developments...", key="chat_input")
if user_input:
    st.session_state["chat_history"].append({
        "type": "user",
        "content": user_input,
        "timestamp": time.time()
    })
    st.rerun()  # Show user message immediately

# If the last message is a user message and no loading message, start processing
if (
    st.session_state["chat_history"]
    and st.session_state["chat_history"][-1]["type"] == "user"
    and not any(msg["type"] == "loading" for msg in st.session_state["chat_history"][-2:])
    and not st.session_state.get("is_processing", False)
):
    st.session_state["process_query"] = st.session_state["chat_history"][-1]["content"]
    st.session_state["chat_history"].append({
        "type": "loading",
        "content": "🤖 AI Agent is analyzing your query and determining the best execution route...",
        "timestamp": time.time()
    })
    st.rerun()

# Process new message
if "process_query" in st.session_state and not st.session_state.get("is_processing", False):
    query_to_process = st.session_state["process_query"]
    del st.session_state["process_query"]
    st.session_state["is_processing"] = True
    st.rerun()

# Handle processing state
if st.session_state.get("is_processing", False):
    # Get the most recent user query from chat history
    user_queries = [msg for msg in st.session_state["chat_history"] if msg["type"] == "user"]
    if user_queries:
        current_query = user_queries[-1]["content"]
        
        # Process with enhanced progress indication
        with st.spinner("🔄 Processing query with multi-tool intelligence..."):
            start_time = time.time()
            
            try:
                # Run the advanced agent
                result = run_advanced_agent(current_query, test_mode=False)
                
                # Store current state for sidebar
                st.session_state["current_state"] = result
                
                # Remove loading message and keep all other messages
                chat_without_loading = [msg for msg in st.session_state["chat_history"] if msg["type"] != "loading"]
                st.session_state["chat_history"] = chat_without_loading
                
                # Add bot response to chat
                st.session_state["chat_history"].append({
                    "type": "bot",
                    "content": result.get("answer", "I apologize, but I couldn't generate a response."),
                    "metadata": result,
                    "timestamp": time.time()
                })
                
            except Exception as e:
                # Remove loading message and keep all other messages
                chat_without_loading = [msg for msg in st.session_state["chat_history"] if msg["type"] != "loading"]
                st.session_state["chat_history"] = chat_without_loading
                
                st.session_state["chat_history"].append({
                    "type": "bot",
                    "content": f"❌ I encountered an error: {str(e)}",
                    "metadata": {"error": True},
                    "timestamp": time.time()
                })
            
            # Clear processing state
            st.session_state["is_processing"] = False
    else:
        # No user queries found, clear processing state
        st.session_state["is_processing"] = False
    
    st.rerun()

# Clear chat button
if st.session_state["chat_history"] and not st.session_state.get("is_processing", False):
    with st.sidebar:
        if st.button("🗑️ Clear Conversation", type="secondary", use_container_width=True):
            st.session_state["chat_history"] = []
            st.session_state["reasoning_expanded"] = {}
            if "current_state" in st.session_state:
                del st.session_state["current_state"]
            st.rerun()

# Professional footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 20px; font-size: 14px;'>
    <p>🚀 AI Research Assistant | Real-time Hugging Face Integration | Multi-tool Intelligence Engine</p>
</div>
""", unsafe_allow_html=True)
