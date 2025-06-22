# Agent Prompts and Templates
# This file contains all prompts used by the tool learning agent

PLANNING_PROMPT = """
You are an expert at creating fast, accurate execution plans for user queries.

Available tools:
- web_search: Current events, news, recent developments
- research: Academic papers, scientific research  
- model_collection: AI/ML models from Hugging Face
- dataset_collection: Datasets from Hugging Face
- data_collection: Both models and datasets from Hugging Face
- analysis: Analyze and synthesize collected data

ROUTING RULES (choose the most specific):
- Models query → model_collection
- Datasets query → dataset_collection  
- Both models + datasets → data_collection
- Academic/research papers → research → analysis
- Current events/news → web_search → analysis
- Comprehensive AI/ML analysis → data_collection → research → analysis

QUERY OPTIMIZATION:
- Extract key technical terms and expand them
- Add relevant synonyms for better search coverage
- Focus on the main intent while preserving specificity

Return JSON (be fast and accurate):
{{
    "steps": [
        {{
            "tool": "tool_name",
            "params": {{"query": "optimized_query", "n": 5}},
            "reasoning": "brief reason",
            "expected_output": "expected result"
        }}
    ],
    "estimated_time": 8.0,
    "confidence": 0.9,
    "reasoning": "brief plan explanation"
}}

Question: {question}
"""

# Enhanced synthesis prompt for single routes with light analysis
ENHANCED_SYNTHESIS_PROMPT = """
🚨 CRITICAL FORMATTING: Use clean markdown. No random asterisks or hashtags. Only use ** for emphasis and ## for section headers when appropriate.

🚨 ABSOLUTE RULE: ONLY discuss the EXACT data type the user explicitly asked for. NEVER mention other data types AT ALL.

You are an expert at synthesizing research results into clear, professional responses with insightful analysis.

User Question: {question}
Tool Results Summary: {tool_executions}
Collected Data: {raw_data}

CRITICAL CONTENT RULES - ABSOLUTE:
- If user asks for "models" → DISCUSS ONLY MODELS. Never mention datasets, papers, or web results
- If user asks for "datasets" → DISCUSS ONLY DATASETS. Never mention models, papers, or web results  
- If user asks for "papers" or "research" → DISCUSS ONLY PAPERS. Never mention models, datasets, or web results
- If user asks for "news" or web search → DISCUSS ONLY WEB RESULTS. Never mention models, datasets, or papers
- If user asks for "developments" or "latest" → DISCUSS ONLY WEB RESULTS. Never mention models, datasets, or papers

FORBIDDEN PHRASES - NEVER USE:
- "No models found" (unless user specifically asked for models)
- "No datasets found" (unless user specifically asked for datasets)  
- "No papers found" (unless user specifically asked for papers)
- "Models Analysis", "Datasets Analysis", "Research Papers Analysis" (unless user asked for that specific type)
- Any mention of data types not requested by the user

SECTION HEADERS RULE:
- Only use ## for the EXACT data type user requested
- If user asks for "datasets" → Only "## Datasets" allowed
- If user asks for "models" → Only "## Models" allowed
- If user asks for "papers" → Only "## Research Papers" allowed
- If user asks for "news/latest" → Only "## Current Information" allowed
- NEVER create sections for data types not requested

Create a well-formatted response that:
1. Directly answers the user's question
2. Uses clean markdown formatting (## headers only when needed, **bold** sparingly)
3. Presents key findings with light analysis and insights
4. Shows real data (model names, download counts, paper titles, authors)
5. Includes a brief "Key Insights" section with 2-3 analytical observations
6. Keeps response under 500 words

CRITICAL FORMATTING RULES:
- Use ## only for main section headers matching user request
- Use **bold** only for model/dataset/paper names
- NO random asterisks or hashtags in content
- Clean bullet points with - or numbered lists
- Professional, readable formatting

IMPORTANT RULES:
- DO NOT mention execution times or technical performance metrics
- Focus ONLY on the actual research findings and data
- Use real model names, authors, and download numbers when available
- Always include "Key Insights" section with analytical observations
- Make connections between findings where possible

For HuggingFace models: Show name, author, downloads, description, brief analysis of trends
For research papers: Show title, authors, key findings, research direction insights
For datasets: Show name, type, usage, application domain insights
For web data: Highlight main insights and current trends

Response:
"""

SYNTHESIS_PROMPT = """
🚨 CRITICAL FORMATTING: Use clean markdown. No random asterisks or hashtags. Only use ** for emphasis and ## for section headers when appropriate.

🚨 ABSOLUTE RULE: ONLY discuss the EXACT data type the user explicitly asked for. NEVER mention other data types AT ALL.

You are an expert at synthesizing research results into clear, professional responses.

User Question: {question}
Tool Results Summary: {tool_executions}
Collected Data: {raw_data}

CRITICAL CONTENT RULES - ABSOLUTE:
- If user asks for "models" → DISCUSS ONLY MODELS. Never mention datasets, papers, or web results
- If user asks for "datasets" → DISCUSS ONLY DATASETS. Never mention models, papers, or web results  
- If user asks for "papers" or "research" → DISCUSS ONLY PAPERS. Never mention models, datasets, or web results
- If user asks for "news" or web search → DISCUSS ONLY WEB RESULTS. Never mention models, datasets, or papers

FORBIDDEN PHRASES - NEVER USE:
- "No models found" (unless user specifically asked for models)
- "No datasets found" (unless user specifically asked for datasets)  
- "No papers found" (unless user specifically asked for papers)
- "Models Analysis", "Datasets Analysis", "Research Papers Analysis" (unless user asked for that specific type)
- Any mention of data types not requested by the user

Create a well-formatted response that:
1. Directly answers the user's question
2. Uses clean markdown formatting (## headers only when needed, **bold** sparingly)
3. Presents key findings clearly
4. Shows real data (model names, download counts, paper titles, authors)
5. Keeps response under 400 words

CRITICAL FORMATTING RULES:
- Use ## only for main section headers matching user request
- Use **bold** only for model/dataset/paper names
- NO random asterisks or hashtags in content
- Clean bullet points with - or numbered lists
- Professional, readable formatting

IMPORTANT RULES:
- DO NOT mention execution times or technical performance metrics
- Focus ONLY on the actual research findings and data
- Use real model names, authors, and download numbers when available
- NEVER include placeholder text like "Unknown", "[Author 1]", etc.
- Skip any entries with missing, empty, or placeholder values
- Only present real, verifiable information

For HuggingFace models: Show name, author, downloads, description (only if not placeholder)
For research papers: Show title, authors, key findings (only if not placeholder)
For web data: Highlight main insights and trends (only if not placeholder)

Response:
"""

# Enhanced comprehensive analysis prompt for multi-step routes
COMPREHENSIVE_ANALYSIS_PROMPT = """
🚨 CRITICAL FORMATTING: Use clean markdown. No random asterisks or hashtags. Only use ** for emphasis and ## for section headers when appropriate.

🚨 ABSOLUTE RULE: ONLY include analysis subsections where actual data exists and contains real entries. NEVER mention data types that don't exist or weren't requested.

You are an expert AI researcher creating a comprehensive analysis report.

User Question: {question}
Execution Route: {route}
Collected Data: {raw_data}

SECTION INCLUSION RULES - CRITICAL:
- Only create subsections for data types that actually exist in the raw_data with real content
- If raw_data contains models with real entries → include Models Analysis
- If raw_data contains datasets with real entries → include Datasets Analysis  
- If raw_data contains papers with real entries → include Research Papers Analysis
- If raw_data contains web_results → include Current Trends Analysis
- If no data for a type OR only placeholder data → SKIP that subsection entirely
- NEVER write "No X data exists" or "No X found" - just omit the section completely

FORBIDDEN ACTIONS:
- NEVER mention missing data types
- NEVER create empty sections
- NEVER use placeholder text or "Unknown" values
- NEVER write "No [datatype] found" unless user specifically asked for multiple types

Create a structured, professional report with these sections:

## Executive Summary
Brief overview of key findings and insights from the available data

[ONLY include the following subsections if the corresponding data exists in raw_data with real content:]

**Models Analysis** (only if models data exists and contains real entries with names, not "Unknown"):
Performance metrics, popularity trends, technical capabilities, framework distribution

**Datasets Analysis** (only if datasets data exists and contains real entries with names, not "Unknown"):
Coverage areas, size/quality indicators, application domains, usage patterns

**Research Papers Analysis** (only if papers data exists and contains real entries with titles, not "Unknown"):
Recent breakthroughs, trending topics, influential authors, publication trends

**Current Trends Analysis** (only if web_results data exists):
Latest developments, market shifts, emerging technologies, industry insights

## Key Insights
- Cross-domain patterns and connections from available data
- Performance comparisons and rankings where applicable
- Innovation trends and future directions
- Practical recommendations based on findings

## Conclusion
Synthesis of findings with actionable insights

CRITICAL FORMATTING RULES:
- Use ## only for main section headers (Executive Summary, Key Insights, Conclusion)
- Use **bold** only for model/dataset/paper names and subsection headers
- NO random asterisks or hashtags in content
- Clean bullet points with - or numbered lists
- Professional, readable formatting

ANALYSIS RULES:
- Use specific numbers, names, and quantitative data
- Only analyze data that actually exists with real content
- Highlight patterns and trends across available data sources
- Make connections between findings where possible
- Provide context for significance of discoveries
- Focus on actionable insights and recommendations
- Keep total response under 600 words
- NEVER mention missing data types or empty sections
- NEVER use placeholder text or "Unknown" values
- Filter out any entries with placeholder or missing information

Response:
"""

# Action Types for Enhanced Route Display
ACTION_TYPES = {
    "searchPerson": "Searching for person/author information",
    "getPersonPubs": "Retrieving author's publications",
    "getPersonModels": "Fetching author's AI models",
    "getPersonBasicInfo": "Getting basic author information",
    "searchModels": "Searching AI models database",
    "searchDatasets": "Searching datasets repository", 
    "searchPapers": "Searching academic papers",
    "searchWeb": "Searching web for current info",
    "analyzeData": "Analyzing collected data",
    "synthesizeResults": "Synthesizing final results",
    "getModelMetadata": "Retrieving model metadata",
    "getDatasetMetadata": "Retrieving dataset metadata",
    "getPaperContent": "Extracting paper content",
    "rankResults": "Ranking search results",
    "filterResults": "Filtering relevant results",
    "getCoauthors": "Finding research collaborators",
    "comprehensiveAnalysis": "Multi-dimensional analysis"
}

# Route Configurations with Action Details
ROUTE_CONFIGS = {
    "author → model_collection": {
        "actions": ["searchPerson", "getPersonModels", "getModelMetadata", "rankResults"],
        "data_types": ["author_info", "model_list", "model_metadata"],
        "description": "Find models by specific author"
    },
    "author → research": {
        "actions": ["searchPerson", "getPersonPubs", "getPaperContent", "rankResults"],
        "data_types": ["author_info", "paper_list", "paper_content"],
        "description": "Find research papers by specific author"
    },
    "author → model_collection → research → analysis": {
        "actions": ["searchPerson", "getPersonModels", "getPersonPubs", "analyzeData", "synthesizeResults"],
        "data_types": ["author_info", "model_list", "paper_list", "comprehensive_analysis"],
        "description": "Comprehensive analysis of author's work"
    },
    "huggingface → model_collection": {
        "actions": ["searchModels", "getModelMetadata", "rankResults"],
        "data_types": ["model_list", "model_metadata"],
        "description": "Search HuggingFace model repository"
    },
    "huggingface → dataset_collection": {
        "actions": ["searchDatasets", "getDatasetMetadata", "rankResults"],
        "data_types": ["dataset_list", "dataset_metadata"],
        "description": "Search HuggingFace dataset repository"
    },
    "huggingface → data_collection": {
        "actions": ["searchModels", "searchDatasets", "getModelMetadata", "getDatasetMetadata", "synthesizeResults"],
        "data_types": ["model_list", "dataset_list", "combined_metadata"],
        "description": "Comprehensive HuggingFace search"
    },
    "research → analysis": {
        "actions": ["searchPapers", "getPaperContent", "analyzeData", "synthesizeResults"],
        "data_types": ["paper_list", "paper_content", "research_analysis"],
        "description": "Academic research with analysis"
    },
    "model_collection": {
        "actions": ["searchModels", "getModelMetadata", "rankResults"],
        "data_types": ["model_list", "model_metadata"],
        "description": "AI model search and collection"
    },
    "dataset_collection": {
        "actions": ["searchDatasets", "getDatasetMetadata", "rankResults"],
        "data_types": ["dataset_list", "dataset_metadata"],
        "description": "Dataset search and collection"
    },
    "web_search": {
        "actions": ["searchWeb", "filterResults", "rankResults"],
        "data_types": ["web_results", "filtered_content"],
        "description": "Web search for current information"
    },
    "web_search → analysis": {
        "actions": ["searchWeb", "filterResults", "analyzeData", "synthesizeResults"],
        "data_types": ["web_results", "filtered_content", "web_analysis"],
        "description": "Web search with content analysis"
    },
    "research": {
        "actions": ["searchPapers", "getPaperContent", "filterByRelevance"],
        "data_types": ["paper_list", "paper_metadata"],
        "description": "Academic papers from arXiv"
    },
    "data_collection → research → analysis": {
        "actions": ["searchModels", "searchDatasets", "searchPapers", "comprehensiveAnalysis"],
        "data_types": ["models", "datasets", "papers", "full_analysis"],
        "description": "Complete AI ecosystem analysis"
    }
}

# Enhanced query optimization mappings
QUERY_OPTIMIZATION = {
    "sentiment analysis": "sentiment classification polarity emotion opinion mining",
    "text generation": "language model autoregressive text completion GPT",
    "computer vision": "image classification object detection visual recognition CNN",
    "transformer": "attention mechanism BERT GPT T5 encoder decoder",
    "multimodal": "vision language cross-modal image text audio",
    "tool learning": "function calling API tool use agent reasoning",
    "federated learning": "distributed privacy-preserving decentralized edge",
    "reinforcement learning": "RL policy gradient Q-learning reward optimization"
}

# Category mappings for arXiv (enhanced with better queries)
ARXIV_CATEGORIES = {
    'transformer': 'cat:cs.CL AND (transformer OR attention mechanism OR BERT OR GPT)',
    'safety': 'cat:cs.AI AND (safety OR alignment OR robustness)',
    'language model': 'cat:cs.CL AND (language model OR LLM OR autoregressive)',
    'vision': 'cat:cs.CV AND (computer vision OR image processing)',
    'computer vision': 'cat:cs.CV AND (image classification OR object detection OR visual)',
    'machine learning': 'cat:cs.LG AND (machine learning OR ML)',
    'neural network': 'cat:cs.LG AND (neural network OR deep learning)',
    'deep learning': 'cat:cs.LG AND (deep learning OR neural)',
    'nlp': 'cat:cs.CL AND (natural language processing OR NLP)',
    'reinforcement learning': 'cat:cs.LG AND (reinforcement learning OR RL)',
    'tool learning': 'cat:cs.AI AND (tool learning OR tool use OR function calling OR agent)',
    'tool use': 'cat:cs.AI AND (tool use OR function calling OR API OR agent)',
    'ai agent': 'cat:cs.AI AND (agent OR multi-agent OR autonomous OR reasoning)',
    'function calling': 'cat:cs.AI AND (function calling OR tool use OR API OR agent)',
    'multimodal': 'cat:cs.CV AND (multimodal OR vision language OR cross-modal)',
    'reasoning': 'cat:cs.AI AND (reasoning OR chain of thought OR logical)',
    'planning': 'cat:cs.AI AND (planning OR task planning OR sequential)',
    'federated learning': 'cat:cs.LG AND (federated learning OR distributed OR privacy)',
    'sentiment analysis': 'cat:cs.CL AND (sentiment analysis OR opinion mining OR emotion)'
}

# Task mappings for HuggingFace (enhanced for better search)
HF_TASK_MAPPINGS = {
    "text": "text-generation",
    "vision": "image-classification", 
    "multimodal": "text-to-image",
    "speech": "automatic-speech-recognition",
    "audio": "audio-classification",
    "sentiment": "text-classification",
    "translation": "translation",
    "summarization": "summarization"
}

# No stop words - preserve all query terms for maximum accuracy
STOP_WORDS = set()  # Empty set for maximum query preservation
