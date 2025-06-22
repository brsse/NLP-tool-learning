# NLP Tool Learning Agent - Sample Dataset Documentation

## Overview
This dataset contains **13 comprehensive examples** covering all possible routes and action combinations in our NLP tool learning agent system, with data from HuggingFace and arXiv platforms.

## Dataset Structure

### Core Components
Each sample contains:
- **Query**: Natural language user request
- **Answer**: Structured response with metrics and findings
- **Base_Question_en**: Template pattern for similar queries
- **Route**: Execution path through the system
- **Actions**: Step-by-step action sequence 
- **Inputs**: Required input parameters
- **Outputs**: Expected output types
- **Entity_Information**: Extracted entities and metadata

## Route Coverage

### 🔍 **Single-Step Routes** (4 examples)
1. **model_collection**: Direct model search
2. **dataset_collection**: Direct dataset search  
3. **web_search**: Current information retrieval
4. **research**: Academic paper search

### 🔗 **Two-Step Routes** (5 examples)
1. **research → analysis**: Papers with insights
2. **web_search → analysis**: Web content analysis
3. **author → research**: Author publication search
4. **author → model_collection**: Author model discovery
5. **huggingface → data_collection**: Platform resource search

### 🌐 **Multi-Step Routes** (4 examples)
1. **author → model_collection → research → analysis**: Comprehensive author analysis
2. **data_collection → research → analysis**: Complete ecosystem analysis
3. **huggingface → model_collection**: Platform-specific model search
4. **huggingface → dataset_collection**: Platform-specific dataset search

## Action Sequence Examples

### 📋 **Person-Centric Actions**
```
searchPerson → getPersonModels → getModelMetadata
searchPerson → getPersonPubs → getPaperContent  
searchPerson → getCoauthors → getCoauthors
```

### 🔧 **Resource Discovery Actions**
```
searchModels → getModelMetadata → rankResults
searchDatasets → getDatasetMetadata → filterResults
searchPapers → getPaperContent → analyzeData
```

### 📊 **Analysis Actions**
```
searchWeb → filterResults → synthesizeResults
searchModels → searchDatasets → comprehensiveAnalysis
```

## Expected Output Categories

### 🤖 **Models**
- `model_list`: Array of discovered models
- `model_metadata`: Technical specifications
- `download_stats`: Usage metrics
- `framework_info`: Implementation details

### 📊 **Datasets** 
- `dataset_list`: Array of discovered datasets
- `dataset_metadata`: Dataset specifications
- `size_info`: Volume and scale metrics
- `usage_stats`: Application metrics

### 📚 **Papers**
- `paper_list`: Array of research papers
- `paper_content`: Abstract and findings
- `citation_info`: Impact metrics
- `research_trends`: Academic insights

### 🌐 **Web Results**
- `web_content`: Current information
- `filtered_results`: Relevant findings
- `trend_analysis`: Market insights

### 🧠 **Analysis**
- `insights`: Key observations
- `patterns`: Trend identification
- `recommendations`: Actionable advice
- `confidence_scores`: Reliability metrics

## Entity Types Covered

### 👤 **Person Entities**
- Author names (Yoshua Bengio, Hugging Face team)
- Organizations (OpenAI, Carnegie Mellon)
- Research interests and specializations

### 🏢 **Organization Entities**
- Companies (OpenAI, Hugging Face)
- Academic departments
- Research institutions

### 🔧 **Technical Entities**
- Model types (transformer, BERT, GPT)
- Dataset categories (multimodal, NLP, vision)
- Tasks (sentiment analysis, image classification)
- Domains (AI safety, federated learning)

### ⏰ **Temporal Entities**
- Time periods ("this month", "2020-2024")
- Publication years
- Trend durations

### 🌐 **Platform Entities**
- HuggingFace Hub
- arXiv repository
- Web search engines
- GitHub repositories

## Query Pattern Templates

### 🎯 **Basic Patterns**
- `"Find XXX by YYY for ZZZ task"`
- `"XXX datasets for YYY task"`
- `"Best XXX models on YYY platform"`
- `"Latest XXX research"`

### 🔍 **Advanced Patterns**
- `"Comprehensive analysis of XXX's contributions to YYY"`
- `"Complete XXX ecosystem analysis for YYY"`
- `"Current XXX developments and analysis"`
- `"XXX resources for YYY task on ZZZ platform"`

## Data Quality Metrics

### ✅ **Coverage Statistics**
- **Routes**: 13/13 possible combinations (100%)
- **Platforms**: HuggingFace + arXiv + Web (100%)
- **Data Types**: Models, Datasets, Papers, Web (100%)
- **Analysis Depth**: Basic → Comprehensive (100%)

### 📊 **Realistic Metrics**
- Download counts: 750K - 50M (realistic ranges)
- Paper counts: 8-25 per query (typical academic search)
- Model/dataset counts: 5-20 per query (practical limits)
- Confidence scores: 0.88-0.95 (high-quality results)

## Usage Examples

### 🔄 **Training Data**
Use for training query understanding, route prediction, and response generation models.

### 🧪 **Testing Framework**
Validate system performance across all supported route combinations.

### 📈 **Benchmarking**
Compare agent performance against expected outputs and confidence scores.

### 🎯 **Template Generation**
Extract patterns for generating similar queries and responses.

---

**Dataset Size**: 13 comprehensive examples  
**Format**: JSON with structured metadata  
**Platforms**: HuggingFace + arXiv + Web  
**Coverage**: 100% route combinations  
**Quality**: Production-ready with realistic metrics 