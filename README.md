# 🤖 NLP Tool Learning Chatbot

An intelligent research paper chatbot that learns to route user queries to the right tools for comprehensive academic search and analysis.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the chatbot
streamlit run app.py
```

Visit `http://localhost:8503` to start chatting with research papers!

## ⚡ Current Capabilities

### **Live APIs (Default)**
- **✅ arXiv**: CS, AI, ML papers (~0.1s response)
- **✅ PubMed**: Medical, biomedical research (~3s response)
- **🔄 Toggle**: Switch between live APIs and static datasets

### **Tool Learning Routes** 
- `searchPapers` (100% accuracy) - Find papers by keywords
- `comparePapers` (100% accuracy) - Compare research papers  
- `journalAnalysis` (100% accuracy) - Analyze publication venues
- `trendAnalysis` (85.7% accuracy) - Track research trends
- `getRelatedPapers` (71.4% accuracy) - Find related work
- `getCitations` (70% accuracy) - Citation analysis
- `getAuthorInfo` (60% accuracy) - Author research profiles

### **AI Engine**
- **LLaMA 3.2** via Ollama for intelligent responses
- **Route Selection** with confidence scoring
- **Context-Aware** prompts for each tool type

## 📊 Performance

**Overall QA Test Accuracy: 83.8%** (62/74 test cases)

**Strong Areas:**
- ✅ Basic paper search and comparison
- ✅ Journal analysis and venue selection
- ✅ Technical domain queries

**Areas for Improvement:**
- 🔄 Author name extraction (Dr./Prof. titles)
- 🔄 Domain-specific citation queries  
- 🔄 Multi-step workflow routing

## 🛠️ Setup Options

### **Option 1: Live APIs (Recommended)**
- Real-time search from arXiv and PubMed
- No setup required - works immediately
- Fast and always up-to-date

### **Option 2: Static Datasets**
```bash
# Download comprehensive datasets (~1.9 GB)
python dataset_downloader.py

# Toggle to static mode in app sidebar
```

## 📁 Project Structure

```
├── app.py                    # Streamlit chatbot interface
├── tool_learning_engine.py   # Route selection engine  
├── paper_tools.py           # arXiv/PubMed API integration
├── ollama_client.py         # LLaMA 3.2 interface
├── qa_test.py              # 74 comprehensive test cases
├── dataset_downloader.py    # Static data downloader
└── requirements.txt         # Dependencies
```

## 🎯 Improvement Roadmap

### **Immediate (Based on Current 83.8% Accuracy)**

1. **Author Detection Enhancement** (60% → 85%)
   - Improve Dr./Prof. title handling
   - Better name extraction from complex queries
   - Author disambiguation

2. **Citation Analysis Improvement** (70% → 85%)
   - Domain-specific citation patterns
   - Impact factor integration
   - Citation trend analysis

3. **Multi-Route Workflows** (50% → 75%)
   - Better compound query handling
   - Sequential tool chaining
   - Context preservation across routes

### **Advanced Features**

- 📈 **Real-time metrics dashboard**
- 🔍 **Advanced search filters** (date, impact factor, venue)
- 🤝 **Collaborative research suggestions**
- 📚 **Personal research library management**
- 🌐 **Multi-language paper support**

## 🧪 Testing

```bash
# Run comprehensive QA tests
python qa_test.py

# Test individual APIs
python paper_tools.py
```

## 🔧 Configuration

- **Live APIs**: Default mode, no configuration needed
- **Ollama**: Install for local LLM ([docs](https://ollama.ai))
- **Static Data**: Optional for offline usage

---

**Ready to explore research papers intelligently!** 🚀 