"""
NLP Tool Learning Agent - Final Optimized Version
Clean, cohesive implementation with enhanced query processing and user-friendly output
"""

import time
import json
import requests
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

# LangChain imports
from langchain.tools import BaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from urllib.parse import urlencode

# Import configurations
from prompts import (
    PLANNING_PROMPT, SYNTHESIS_PROMPT, ENHANCED_SYNTHESIS_PROMPT, 
    COMPREHENSIVE_ANALYSIS_PROMPT, ACTION_TYPES, ROUTE_CONFIGS,
    ARXIV_CATEGORIES, HF_TASK_MAPPINGS, STOP_WORDS, QUERY_OPTIMIZATION
)

# LLM Setup - Optimized for speed
llama3_json = ChatOllama(model="llama3.2", temperature=0, timeout=10)
llama3 = ChatOllama(model="llama3.2", temperature=0.1, timeout=15)

@dataclass
class ToolExecution:
    tool_name: str
    input_params: Dict[str, Any]
    output: Any
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    step_number: int = 0

@dataclass
class ExecutionPlan:
    steps: List[Dict[str, Any]]
    estimated_time: float
    confidence: float
    reasoning: str
    route: str = ""
    route_reasoning: str = ""

class AdvancedGraphState(TypedDict, total=False):
    question: str
    current_step: int
    total_steps: int
    tool_executions: List[ToolExecution]
    execution_plan: Optional[ExecutionPlan]
    raw_data: Dict[str, Any]
    generation: str
    confidence_score: float
    debug_info: List[str]
    step_outputs: List[str]

class APIHelper:
    @staticmethod
    def retry_api_call(api_func, max_retries=3, base_delay=1):
        for attempt in range(max_retries):
            try:
                return api_func()
            except requests.exceptions.RetryError:
                time.sleep(base_delay * (2 ** attempt))
                continue
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(base_delay)
                continue
        raise Exception("All retry attempts failed")

    @staticmethod
    def extract_query_intent(query: str) -> Dict[str, Any]:
        """Enhanced intent extraction with precise classification"""
        query_lower = query.lower()
        entities = {}
        
        # Author detection with improved patterns
        import re
        author_patterns = [
            r'by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'from\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'([A-Z][a-z]+)\s+(?:models|papers|research|team)'
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, query)
            if match:
                entities['author'] = match.group(1)
                break
        
        # Platform detection
        if any(term in query_lower for term in ['hugging face', 'huggingface', 'hf']):
            entities['platform'] = 'huggingface'
        
        # PRECISE query type classification - most specific wins
        if any(term in query_lower for term in ['paper', 'research', 'study', 'publication']):
            query_type = 'papers'
        elif any(term in query_lower for term in ['dataset', 'data', 'corpus']) and 'model' not in query_lower:
            query_type = 'datasets'
        elif any(term in query_lower for term in ['model', 'transformer', 'gpt', 'bert', 'llm']) and 'dataset' not in query_lower:
            query_type = 'models'
        elif any(term in query_lower for term in ['news', 'latest', 'recent', 'current', 'new', 'development']):
            query_type = 'web_search'
        elif 'author' in entities:
            query_type = 'author_search'
        else:
            query_type = 'general'
        
        return {
            'type': query_type,
            'entities': entities,
            'keywords': [word for word in query_lower.split() if len(word) > 2]
        }

    @staticmethod
    def optimize_query(query: str) -> str:
        """Optimize query for better search results"""
        query_lower = query.lower()
        
        # Apply query optimizations
        for term, expanded in QUERY_OPTIMIZATION.items():
            if term in query_lower:
                # Add expanded terms while keeping original
                query = f"{query} {expanded}"
                break
        
        return query.strip()

# Core Tools - Enhanced and Optimized

class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Web search for current information"
    
    def _run(self, query: str, max_results: int = 5):
        # Optimize query for web search
        optimized_query = APIHelper.optimize_query(query)
        
        def search_api():
            wrapper = DuckDuckGoSearchAPIWrapper(max_results=max_results)
            search_tool = DuckDuckGoSearchRun(api_wrapper=wrapper)
            return search_tool.invoke(optimized_query)
        
        results = APIHelper.retry_api_call(search_api)
        return {"results": results, "query": optimized_query}

class ResearchTool(BaseTool):
    name: str = "research"
    description: str = "Search academic papers on arXiv"
    
    def _run(self, query: str, analysis_depth: str = "basic"):
        intent = APIHelper.extract_query_intent(query)
        papers = self._get_arxiv_papers(query, intent, 5)
        return {"papers": papers, "query": query}
    
    def _get_arxiv_papers(self, query: str, intent: Dict[str, Any], n: int):
        search_query = self._build_arxiv_query(query, intent)
        
        def fetch_papers():
            base_url = "http://export.arxiv.org/api/query?"
            params = {
                'search_query': search_query,
                'start': 0,
                'max_results': n,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            
            response = requests.get(base_url + urlencode(params), timeout=15)
            if response.status_code != 200:
                raise Exception(f"arXiv API error: {response.status_code}")
            
            return self._parse_arxiv_response(response.content, intent)
        
        return APIHelper.retry_api_call(fetch_papers)
    
    def _build_arxiv_query(self, query: str, intent: Dict[str, Any]) -> str:
        if 'author' in intent['entities']:
            return f"au:{intent['entities']['author']}"
        
        query_lower = query.lower()
        # Enhanced category matching
        for term, arxiv_query in ARXIV_CATEGORIES.items():
            if term in query_lower:
                return arxiv_query
        
        # Fallback with better keyword extraction
        words = [word for word in query_lower.split() if len(word) > 2 and word not in STOP_WORDS]
        return " AND ".join(words[:3]) if words else "cat:cs.AI"
    
    def _parse_arxiv_response(self, content: bytes, intent: Dict[str, Any]) -> List[Dict]:
        root = ET.fromstring(content)
        entries = root.findall('{http://www.w3.org/2005/Atom}entry')
        papers = []
        
        for entry in entries:
            title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip().replace('\n', ' ')
            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip().replace('\n', ' ')[:300] + "..."
            published = entry.find('{http://www.w3.org/2005/Atom}published').text[:10]
            arxiv_id = entry.find('{http://www.w3.org/2005/Atom}id').text.split('/')[-1]
            
            authors = [author.find('{http://www.w3.org/2005/Atom}name').text 
                      for author in entry.findall('{http://www.w3.org/2005/Atom}author')]
            
            # Filter out placeholder entries
            if title and title != "Unknown" and authors:
                papers.append({
                    "title": title,
                    "authors": authors[:5],
                    "abstract": summary,
                    "published_date": published,
                    "source": f"https://arxiv.org/abs/{arxiv_id}",
                    "arxiv_id": arxiv_id
                })
        
        return papers

class ModelCollectionTool(BaseTool):
    name: str = "model_collection"
    description: str = "Search AI models from Hugging Face"
    
    def _run(self, query: str, n: int = 5):
        intent = APIHelper.extract_query_intent(query)
        models = self._get_hf_models(query, intent, n)
        return {"models": models, "query": query}
    
    def _get_hf_models(self, query: str, intent: Dict[str, Any], n: int):
        search_terms = self._build_search_query(query, intent)
        
        def fetch_models():
            params = {
                "search": search_terms,
                "limit": min(n * 2, 20),
                "sort": "downloads",
                "direction": -1
            }
            
            response = requests.get("https://huggingface.co/api/models", params=params, timeout=15)
            if response.status_code != 200:
                raise Exception(f"HF API error: {response.status_code}")
            
            return self._parse_hf_models(response.json(), intent, n)
        
        return APIHelper.retry_api_call(fetch_models)
    
    def _build_search_query(self, query: str, intent: Dict[str, Any]) -> str:
        if 'author' in intent['entities']:
            return intent['entities']['author']
        
        # Enhanced query building with optimization
        optimized_query = APIHelper.optimize_query(query)
        query_lower = optimized_query.lower()
        
        if any(term in query_lower for term in ["top", "best", "popular"]):
            return ""  # Empty returns popular models
        
        # Extract key terms for better search
        key_terms = []
        for word in query_lower.split():
            if len(word) > 2 and word not in STOP_WORDS:
                key_terms.append(word)
        
        return " ".join(key_terms[:3])
    
    def _parse_hf_models(self, models_data: List[Dict], intent: Dict[str, Any], n: int) -> List[Dict]:
        models = []
        author_filter = intent['entities'].get('author', '').lower()
        
        for model_data in models_data[:n*2]:
            model_id = model_data.get("modelId", "")
            if not model_id:
                continue
                
            author = model_id.split("/")[0] if "/" in model_id else ""
            
            # Filter by author if specified
            if author_filter and author_filter not in author.lower():
                continue
            
            # Only include models with real data
            if author and author != "Unknown":
                models.append({
                    "name": model_id,
                    "author": author,
                    "model_type": model_data.get("pipeline_tag", "unknown"),
                    "downloads": model_data.get("downloads", 0),
                    "likes": model_data.get("likes", 0),
                    "description": model_data.get("description", f"Model for {model_data.get('pipeline_tag', 'various')} tasks")[:200],
                    "tags": model_data.get("tags", [])[:3],
                    "framework": "PyTorch" if "pytorch" in str(model_data.get("tags", [])).lower() else "TensorFlow"
                })
                
                if len(models) >= n:
                    break
        
        return models

class DatasetCollectionTool(BaseTool):
    name: str = "dataset_collection"
    description: str = "Search datasets from Hugging Face"
    
    def _run(self, query: str, n: int = 5):
        datasets = self._get_hf_datasets(query, n)
        return {"datasets": datasets, "query": query}
    
    def _get_hf_datasets(self, query: str, n: int):
        # Optimize query for dataset search
        optimized_query = APIHelper.optimize_query(query)
        
        def fetch_datasets():
            params = {
                "search": optimized_query.strip(),
                "limit": min(n * 2, 20),
                "sort": "downloads",
                "direction": -1
            }
            
            response = requests.get("https://huggingface.co/api/datasets", params=params, timeout=15)
            if response.status_code != 200:
                raise Exception(f"HF Datasets API error: {response.status_code}")
            
            datasets_data = response.json()
            datasets = []
            
            for dataset_data in datasets_data[:n]:
                dataset_id = dataset_data.get("id", "")
                if dataset_id:
                    author = dataset_id.split("/")[0] if "/" in dataset_id else ""
                    
                    # Only include datasets with real data
                    if author and author != "Unknown":
                        datasets.append({
                            "name": dataset_id,
                            "author": author,
                            "dataset_type": dataset_data.get("task_categories", ["unknown"])[0] if dataset_data.get("task_categories") else "unknown",
                            "downloads": dataset_data.get("downloads", 0),
                            "likes": dataset_data.get("likes", 0),
                            "description": dataset_data.get("description", f"Dataset for {dataset_data.get('task_categories', ['various'])[0]} tasks")[:200],
                            "tags": dataset_data.get("tags", [])[:3],
                            "size": f"{dataset_data.get('size_bytes', 0) // 1024 // 1024}MB" if dataset_data.get('size_bytes') else "Unknown",
                            "samples": dataset_data.get("num_rows", "Unknown")
                        })
            
            return datasets
        
        return APIHelper.retry_api_call(fetch_datasets)

class DataCollectionTool(BaseTool):
    name: str = "data_collection"
    description: str = "Comprehensive data collection combining models and datasets"
    
    def _run(self, query: str, n_models: int = 3, n_datasets: int = 3):
        model_tool = ModelCollectionTool()
        dataset_tool = DatasetCollectionTool()
        
        models_result = model_tool._run(query, n_models)
        datasets_result = dataset_tool._run(query, n_datasets)
        
        return {
            "models": models_result.get("models", []),
            "datasets": datasets_result.get("datasets", []),
            "query": query
        }

class AnalysisTool(BaseTool):
    name: str = "analysis"
    description: str = "Analyze and synthesize collected data"
    
    def _run(self, data: Dict[str, Any], analysis_type: str = "summary"):
        # Filter out placeholder data before analysis
        filtered_data = self._filter_real_data(data)
        
        if analysis_type == "comprehensive":
            return {"analysis": self._generate_comprehensive_analysis(filtered_data)}
        else:
            return {"analysis": self._generate_summary(filtered_data)}
    
    def _filter_real_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out placeholder and unknown data"""
        filtered = {}
        
        if "models" in data:
            filtered["models"] = [m for m in data["models"] 
                                if m.get('name') and 'Unknown' not in m.get('name', '') and m.get('author') != 'Unknown']
        
        if "datasets" in data:
            filtered["datasets"] = [d for d in data["datasets"] 
                                  if d.get('name') and 'Unknown' not in d.get('name', '') and d.get('author') != 'Unknown']
        
        if "papers" in data:
            filtered["papers"] = [p for p in data["papers"] 
                                if p.get('title') and 'Unknown' not in p.get('title', '')]
        
        if "web_results" in data:
            filtered["web_results"] = data["web_results"]
        
        return filtered
    
    def _generate_summary(self, data: Dict[str, Any]) -> str:
        summary_parts = []
        
        models = data.get("models", [])
        if models:
            summary_parts.append(f"Found {len(models)} AI models")
            if models[0].get('downloads', 0) > 0:
                top_model = max(models, key=lambda x: x.get("downloads", 0))
                summary_parts.append(f"Most popular: {top_model.get('name')} with {top_model.get('downloads', 0):,} downloads")
        
        datasets = data.get("datasets", [])
        if datasets:
            summary_parts.append(f"Found {len(datasets)} datasets")
        
        papers = data.get("papers", [])
        if papers:
            summary_parts.append(f"Found {len(papers)} research papers")
            if papers:
                summary_parts.append(f"Most recent: {papers[0].get('title')}")
        
        return ". ".join(summary_parts) + "." if summary_parts else "No significant data found"
    
    def _generate_comprehensive_analysis(self, data: Dict[str, Any]) -> str:
        analysis_parts = []
        
        models = data.get("models", [])
        if models:
            total_downloads = sum(model.get("downloads", 0) for model in models if model.get('downloads', 0) > 0)
            if total_downloads > 0:
                avg_downloads = total_downloads // len([m for m in models if m.get('downloads', 0) > 0])
                analysis_parts.append(f"Model Analysis: {len(models)} models with average {avg_downloads:,} downloads")
            
            frameworks = {}
            for model in models:
                fw = model.get("framework", "")
                if fw and fw != 'Unknown':
                    frameworks[fw] = frameworks.get(fw, 0) + 1
            
            if frameworks:
                dominant_fw = max(frameworks, key=frameworks.get)
                analysis_parts.append(f"Dominant framework: {dominant_fw}")
        
        datasets = data.get("datasets", [])
        if datasets:
            analysis_parts.append(f"Dataset Analysis: {len(datasets)} datasets covering various domains")
        
        papers = data.get("papers", [])
        if papers:
            analysis_parts.append(f"Research Analysis: {len(papers)} recent academic papers")
            
            all_titles = " ".join(paper.get("title", "") for paper in papers).lower()
            if "transformer" in all_titles:
                analysis_parts.append("Strong focus on transformer architectures")
            if "tool" in all_titles or "agent" in all_titles:
                analysis_parts.append("Active research in AI agents and tool use")
        
        return ". ".join(analysis_parts) + "." if analysis_parts else "Comprehensive analysis completed"

# Tool Registry
TOOL_REGISTRY = {
    "web_search": WebSearchTool(),
    "research": ResearchTool(),
    "analysis": AnalysisTool(),
    "model_collection": ModelCollectionTool(),
    "dataset_collection": DatasetCollectionTool(),
    "data_collection": DataCollectionTool(),
}

# Enhanced Execution Planning with FIXED routing

class ExecutionPlanner:
    def __init__(self, llm):
        self.llm = llm
        self.planning_prompt = PromptTemplate(
            template=PLANNING_PROMPT,
            input_variables=["question"]
        )
    
    def create_plan(self, question: str) -> ExecutionPlan:
        # FIXED: Use precise intent detection for routing
        intent = APIHelper.extract_query_intent(question)
        
        # Direct routing based on intent - MOST SPECIFIC WINS
        if intent['type'] == 'papers':
            return ExecutionPlan(
                steps=[{"tool": "research", "params": {"query": question}}],
                estimated_time=5.0, confidence=0.9, reasoning="Research paper search",
                route="research"
            )
        elif intent['type'] == 'datasets':
            return ExecutionPlan(
                steps=[{"tool": "dataset_collection", "params": {"query": question, "n": 5}}],
                estimated_time=5.0, confidence=0.9, reasoning="Dataset search",
                route="dataset_collection"
            )
        elif intent['type'] == 'models':
            return ExecutionPlan(
                steps=[{"tool": "model_collection", "params": {"query": question, "n": 5}}],
                estimated_time=5.0, confidence=0.9, reasoning="Model search",
                route="model_collection"
            )
        elif intent['type'] == 'web_search':
            return ExecutionPlan(
                steps=[{"tool": "web_search", "params": {"query": question}}],
                estimated_time=4.0, confidence=0.8, reasoning="Current information search",
                route="web_search"
            )
        else:
            # Fallback: prefer web search for general queries
            return ExecutionPlan(
                steps=[{"tool": "web_search", "params": {"query": question}}],
                estimated_time=4.0, confidence=0.7, reasoning="General web search",
                route="web_search"
            )

# Graph Nodes - Optimized Implementation

def planning_node(state: AdvancedGraphState) -> AdvancedGraphState:
    planner = ExecutionPlanner(llama3_json)
    plan = planner.create_plan(state["question"])
    
    return {
        **state,
        "execution_plan": plan,
        "total_steps": len(plan.steps),
        "current_step": 0
    }

def tool_execution_node(state: AdvancedGraphState) -> AdvancedGraphState:
    current_step = state.get("current_step", 0)
    plan = state.get("execution_plan", None)
    
    if not plan or current_step >= len(plan.steps):
        return state
    
    step = plan.steps[current_step]
    tool_name = step["tool"]
    params = step.get("params", {})
    
    # Ensure required parameters with optimization
    if tool_name in ["web_search", "research", "model_collection", "dataset_collection", "data_collection"]:
        if "query" not in params:
            params["query"] = APIHelper.optimize_query(state.get("question", ""))
    elif tool_name == "analysis":
        params["data"] = state.get("raw_data", {})
        if "analysis_type" not in params:
            params["analysis_type"] = "summary"
    
    # Execute tool
    tool = TOOL_REGISTRY[tool_name]
    start_time = time.time()
    
    try:
        result = tool._run(**params)
        execution = ToolExecution(
            tool_name=tool_name,
            input_params=params,
            output=result,
            execution_time=time.time() - start_time,
            success=True,
            step_number=current_step + 1
        )
    except Exception as e:
        execution = ToolExecution(
            tool_name=tool_name,
            input_params=params,
            output=None,
            execution_time=time.time() - start_time,
            success=False,
            error_message=str(e),
            step_number=current_step + 1
        )
    
    # Update state with filtered data
    tool_executions = state.get("tool_executions", [])
    tool_executions.append(execution)
    
    raw_data = state.get("raw_data", {})
    if execution.success and result:
        if tool_name == "web_search":
            raw_data["web_results"] = result
        elif tool_name == "research":
            raw_data["papers"] = result.get("papers", [])
        elif tool_name == "analysis":
            raw_data["analysis"] = result
        elif tool_name == "model_collection":
            raw_data["models"] = result.get("models", [])
        elif tool_name == "dataset_collection":
            raw_data["datasets"] = result.get("datasets", [])
        elif tool_name == "data_collection":
            raw_data["models"] = result.get("models", [])
            raw_data["datasets"] = result.get("datasets", [])
    
    return {
        **state,
        "tool_executions": tool_executions,
        "raw_data": raw_data,
        "current_step": current_step + 1
    }

def should_continue(state: AdvancedGraphState) -> str:
    current_step = state.get("current_step", 0)
    total_steps = state.get("total_steps", 0)
    return "continue" if current_step < total_steps else "synthesize"

def synthesis_node(state: AdvancedGraphState) -> AdvancedGraphState:
    # Use enhanced synthesis for ALL routes (no comprehensive analysis to avoid cross-contamination)
    synthesis_prompt = PromptTemplate(
        template=ENHANCED_SYNTHESIS_PROMPT,
        input_variables=["question", "tool_executions", "raw_data"]
    )
    
    exec_summary = []
    for i, exec_info in enumerate(state.get("tool_executions", [])):
        tool_name = exec_info.tool_name if hasattr(exec_info, 'tool_name') else exec_info.get("tool", "Unknown")
        success = exec_info.success if hasattr(exec_info, 'success') else exec_info.get("success", False)
        exec_summary.append(f"Step {i+1}: {tool_name} - {'✅' if success else '❌'}")
    
    try:
        synthesis_chain = synthesis_prompt | llama3 | StrOutputParser()
        generation = synthesis_chain.invoke({
            "question": state["question"],
            "tool_executions": "\n".join(exec_summary),
            "raw_data": json.dumps(state.get("raw_data", {}), indent=2)
        })
        
        if not generation or len(generation.strip()) < 10:
            generation = _generate_smart_fallback_response(state)
    except:
        generation = _generate_smart_fallback_response(state)
    
    # Calculate confidence
    successful_executions = sum(1 for exec_info in state.get("tool_executions", []) 
                              if (exec_info.success if hasattr(exec_info, 'success') else exec_info.get("success", False)))
    total_executions = len(state.get("tool_executions", []))
    confidence = successful_executions / total_executions if total_executions > 0 else 0.0
    
    return {
        **state,
        "generation": generation,
        "confidence_score": confidence
    }

def _generate_smart_fallback_response(state: AdvancedGraphState) -> str:
    """Enhanced fallback response that only shows what user asked for"""
    question = state.get("question", "")
    raw_data = state.get("raw_data", {})
    q_lower = question.lower()
    
    # STRICT detection - exact keyword matching
    asked_for_models = any(word in q_lower for word in ['model', 'transformer', 'gpt', 'bert', 'llm']) and 'dataset' not in q_lower
    asked_for_datasets = any(word in q_lower for word in ['dataset', 'data', 'corpus']) and 'model' not in q_lower  
    asked_for_papers = any(word in q_lower for word in ['paper', 'research', 'study', 'publication'])
    asked_for_web = any(word in q_lower for word in ['news', 'latest', 'recent', 'current', 'new', 'development'])
    asked_for_comprehensive = any(word in q_lower for word in ['comprehensive', 'landscape', 'overview', 'ecosystem'])
    
    response_parts = [f"# {question}\n"]
    
    # Filter data to only include real entries
    models = [m for m in raw_data.get("models", []) 
             if m.get('name') and 'Unknown' not in m.get('name', '') and m.get('author') != 'Unknown']
    datasets = [d for d in raw_data.get("datasets", []) 
               if d.get('name') and 'Unknown' not in d.get('name', '') and d.get('author') != 'Unknown']
    papers = [p for p in raw_data.get("papers", []) 
             if p.get('title') and 'Unknown' not in p.get('title', '')]
    
    # Only show sections for what user asked for AND what we found
    if models and (asked_for_models or asked_for_comprehensive):
        response_parts.append("## Models Found")
        for i, model in enumerate(models[:5], 1):
            response_parts.append(f"**{i}. {model['name']}**")
            if model.get('downloads', 0) > 0:
                response_parts.append(f"- Downloads: {model['downloads']:,}")
            if model.get('description'):
                desc = model['description'][:100] + "..." if len(model['description']) > 100 else model['description']
                response_parts.append(f"- Description: {desc}")
            response_parts.append("")
    elif asked_for_models and not models:
        response_parts.append("## Models\nNo models found for this query.\n")
    
    if datasets and (asked_for_datasets or asked_for_comprehensive):
        response_parts.append("## Datasets Found")
        for i, dataset in enumerate(datasets[:3], 1):
            response_parts.append(f"**{i}. {dataset['name']}**")
            if dataset.get('dataset_type') and 'unknown' not in dataset.get('dataset_type', ''):
                response_parts.append(f"- Type: {dataset['dataset_type']}")
            response_parts.append("")
    elif asked_for_datasets and not datasets:
        response_parts.append("## Datasets\nNo datasets found for this query.\n")
    
    if papers and (asked_for_papers or asked_for_comprehensive):
        response_parts.append("## Research Papers")
        for i, paper in enumerate(papers[:3], 1):
            response_parts.append(f"**{i}. {paper['title']}**")
            if paper.get('authors'):
                response_parts.append(f"- Authors: {', '.join(paper['authors'][:3])}")
            response_parts.append("")
    elif asked_for_papers and not papers:
        response_parts.append("## Research Papers\nNo research papers found for this query.\n")
    
    # Web results - only if user asked for current info
    if raw_data.get("web_results") and (asked_for_web or asked_for_comprehensive):
        response_parts.append("## Current Information")
        web_data = raw_data["web_results"]
        if isinstance(web_data, dict) and "results" in web_data:
            response_parts.append(str(web_data["results"])[:300] + "...")
        elif isinstance(web_data, str):
            response_parts.append(web_data[:300] + "...")
        response_parts.append("")
    
    return "\n".join(response_parts) if len(response_parts) > 1 else "No significant data found for this query."

# Workflow Creation

def create_advanced_workflow():
    workflow = StateGraph(AdvancedGraphState)
    
    workflow.add_node("planning", planning_node)
    workflow.add_node("tool_execution", tool_execution_node)
    workflow.add_node("synthesis", synthesis_node)
    
    workflow.add_edge("planning", "tool_execution")
    workflow.add_conditional_edges(
        "tool_execution",
        should_continue,
        {"continue": "tool_execution", "synthesize": "synthesis"}
    )
    workflow.add_edge("synthesis", END)
    workflow.set_entry_point("planning")
    
    return workflow.compile()

# Main Agent Function

advanced_agent = create_advanced_workflow()

def run_advanced_agent(query: str, test_mode: bool = False) -> Dict[str, Any]:
    """Run the optimized agent with enhanced query processing"""
    start_time = time.time()
    
    # Optimize the query before processing
    optimized_query = APIHelper.optimize_query(query)
    
    initial_state = {
        "question": optimized_query,
        "current_step": 0,
        "total_steps": 0,
        "tool_executions": [],
        "execution_plan": None,
        "raw_data": {},
        "generation": "",
        "confidence_score": 0.0,
        "debug_info": [],
        "step_outputs": []
    }
    
    try:
        result = advanced_agent.invoke(initial_state)
        total_time = time.time() - start_time
        
        # Extract route
        execution_plan = result.get("execution_plan", None)
        if execution_plan and hasattr(execution_plan, 'route'):
            route_str = execution_plan.route
        else:
            execution_path = [exec_info.tool_name if hasattr(exec_info, 'tool_name') else exec_info.get("tool", "Unknown")
                            for exec_info in result.get("tool_executions", [])]
            route_str = " → ".join(execution_path) if execution_path else "unknown"
        
        if test_mode:
            successful_steps = sum(1 for exec_info in result.get("tool_executions", []) 
                                 if (exec_info.success if hasattr(exec_info, 'success') else exec_info.get("success", False)))
            
            return {
                "query": query,
                "result": result.get("generation", ""),
                "route": route_str,
                "exe_time": total_time,
                "confidence": result.get("confidence_score", 0.0),
                "steps_executed": len(result.get("tool_executions", [])),
                "successful_steps": successful_steps,
                "error": None
            }
        
        # UI format with detailed metadata
        tool_executions = result.get("tool_executions", [])
        successful_steps = sum(1 for exec_info in tool_executions 
                             if (exec_info.success if hasattr(exec_info, 'success') else exec_info.get("success", False)))
        
        return {
            "question": query,  # Return original query for UI
            "answer": result.get("generation", ""),
            "confidence": result.get("confidence_score", 0.0),
            "execution_metadata": {
                "route": route_str,
                "total_time": total_time,
                "steps_executed": len(tool_executions),
                "successful_steps": successful_steps,
                "tool_executions": tool_executions
            },
            "route": route_str,
            "tool_executions": tool_executions
        }
        
    except Exception as e:
        total_time = time.time() - start_time
        
        if test_mode:
            return {
                "query": query,
                "result": "execution error",
                "route": "error",
                "exe_time": total_time,
                "confidence": 0.0,
                "steps_executed": 0,
                "successful_steps": 0,
                "error": str(e)
            }
        
        return {
            "question": query,
            "answer": f"❌ An error occurred: {str(e)}",
            "confidence": 0.0,
            "execution_metadata": {"route": "error", "total_time": total_time, "steps_executed": 0},
            "error": True
        }
