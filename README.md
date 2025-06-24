# Tool Learning System

🔬 **Intelligent arXiv Paper Search with LLM Route Selection**

LLM-powered system that intelligently routes research queries and searches through a curated arXiv dataset.

## ⚡ What You Have

- **2 Models**: `llama3.2`, `deepseek-r1` 
- **7 Routes**: Smart query routing for different research tasks
- **3,500 Papers**: Static arXiv dataset across 7 ML/AI topics
- **QA Testing**: Comprehensive testing with quality metrics

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start web interface
streamlit run app.py

# Test the system
python qa_test.py
```

## 🎯 Available Routes

| Route | Purpose | Example |
|-------|---------|---------|
| `searchPapers` | Find research papers | "find papers about BERT" |
| `getAuthorInfo` | Author information | "who is Geoffrey Hinton" |
| `getCitations` | Citation analysis | "citation count for BERT paper" |
| `getRelatedPapers` | Find related work | "papers related to transformers" |
| `comparePapers` | Compare approaches | "compare BERT vs GPT" |
| `trendAnalysis` | Research trends | "trends in machine learning" |
| `journalAnalysis` | Publication venues | "best ML conferences" |

## 📊 Dataset

```
dataset/arxiv/ (3,500 papers total)
├── machine_learning.jsonl     (500 papers)
├── deep_learning.jsonl        (500 papers)  
├── bert.jsonl                 (500 papers)
├── transformers.jsonl         (500 papers)
├── tool_learning.jsonl        (500 papers)
├── rag.jsonl                  (500 papers)
└── hallucination.jsonl        (500 papers)
```

## 🧪 Testing

```bash
# Run QA tests (sample)
python qa_test.py

# Results saved to: model/[model_name]/qa_test_results_[mode]_[timestamp].json
```

Each JSON contains **8 features**:
1. Route Performance
2. Response Quality  
3. Model Analysis
4. Difficulty Analysis
5. Additional Tests
6. Papers Analysis
7. Data Mode Info
8. Test Metadata

## 🔧 Key Files

- `app.py` - Streamlit web interface
- `tool_learning.py` - Core system (routing + search)
- `model.py` - Model management (llama3.2, deepseek-r1)
- `qa_test.py` - Testing system with quality metrics
- `qa.py` - 78 test cases with expected answers
- `prompts.py` - LLM prompts

---
**🔬 Ready to search 3,500 papers with intelligent LLM routing!** 