# Multi-Agent Research Assistant

**Real web search + AI APIs + PDF analysis in one simple file.**

## 🌐 **Real Web Integration**

- **Live Web Search**: DuckDuckGo, Wikipedia, arXiv, Reddit APIs
- **Content Extraction**: Real websites with BeautifulSoup
- **AI Summarization**: OpenAI GPT-3.5/4 integration
- **PDF Processing**: PyMuPDF/PyPDF2 for document analysis
- **Fallback System**: Mock data when APIs are unavailable

## 🚀 Quick Start

1. **Install:**
```bash
pip install aiohttp beautifulsoup4 requests PyPDF2 openai
```

2. **Set API Key (Optional):**
```bash
export OPENAI_API_KEY="your_openai_key_here"
```

3. **Run:**
```bash
python main.py
```

## 🔗 **API Integrations**

### **Web Search APIs:**
- **DuckDuckGo Instant Answers** - No API key needed
- **Wikipedia REST API** - Free, no limits
- **arXiv API** - Academic papers, free access
- **Reddit JSON API** - Community discussions

### **AI Services:**
- **OpenAI GPT-3.5/4** - Intelligent summarization
- **Fallback**: Extractive summarization without AI

### **Content Sources:**
- **Live websites** - Real-time content extraction
- **Academic papers** - arXiv integration
- **PDF documents** - Local file processing
- **Social discussions** - Reddit insights

## 💡 **Advanced Usage**

### **With OpenAI API:**
```python
import os
os.environ['OPENAI_API_KEY'] = 'your_key_here'

assistant = ResearchAssistant()
results = await assistant.research(
    query="machine learning healthcare applications",
    pdf_files=["research_paper.pdf"],
    max_web_results=10
)

# Gets AI-powered summary + real web data
```

### **Without APIs (Free Mode):**
```python
# Works without any API keys
assistant = ResearchAssistant()
results = await assistant.research(
    query="artificial intelligence trends",
    max_web_results=5
)

# Uses extractive summarization + free APIs
```

## 📊 **Real Data Sources**

The system automatically searches:

1. **🦆 DuckDuckGo** - Instant answers and topics
2. **📚 Wikipedia** - Comprehensive articles  
3. **🎓 arXiv** - Academic research papers
4. **💬 Reddit** - Community discussions
5. **🌐 Live Websites** - Real content extraction
6. **📄 PDF Files** - Your local documents

## 🎯 **Example Output**

```
🔬 Multi-Agent Research Assistant
🔍 Searching web for: quantum computing applications
   ✅ Found 3 DuckDuckGo results
   ✅ Found 1 Wikipedia article  
   ✅ Found 2 arXiv papers
   ✅ Extracted content from 6 websites
🤖 OpenAI API available for AI summarization
📝 Creating AI-powered summary
📊 Generating comprehensive report

✅ Research Complete!
📁 Task ID: abc123
🌐 Web sources: 6 (5 real + 1 fallback)
📄 PDF sources: 0
📂 Check: outputs/task_abc123/
```

## 📁 **Rich Output Files**

Each research creates:
```
outputs/task_abc123/
├── report.md        # Full research report
├── summary.md       # AI/extractive summary
└── results.json     # Complete structured data
```

## 🔧 **API Configuration**

### **Required (Free):**
- ✅ DuckDuckGo API - No key needed
- ✅ Wikipedia API - No key needed  
- ✅ arXiv API - No key needed
- ✅ Reddit JSON - No key needed

### **Optional (Enhanced):**
```bash
# For AI summarization
export OPENAI_API_KEY="sk-..."

# For Google Search (if you want to add it)
export GOOGLE_API_KEY="your_key"
export GOOGLE_CSE_ID="your_cse_id"
```

## 🚀 **Features**

### **Smart Web Search:**
- Multi-API approach for comprehensive coverage
- Automatic content extraction from websites
- Source diversity (academic + general + discussions)
- Relevance scoring and filtering

### **AI Integration:**
- OpenAI GPT for intelligent summarization
- Context-aware analysis across sources
- Professional report generation
- Fallback to extractive methods

### **PDF Analysis:**
- Advanced text extraction (PyMuPDF/PyPDF2)
- Key point identification  
- Section analysis
- Error handling for encrypted/corrupted files

### **Professional Output:**
- Markdown reports with proper formatting
- Source attribution and links
- Research statistics and insights
- JSON data for further processing

## 🛠️ **Troubleshooting**

### **No results found?**
```bash
# Check internet connection
ping google.com

# Install all dependencies  
pip install aiohttp beautifulsoup4 requests PyPDF2 openai lxml
```

### **API errors?**
- DuckDuckGo/Wikipedia are free - no setup needed
- OpenAI requires valid API key for AI features
- System works without OpenAI (uses extractive summarization)

### **PDF processing fails?**
```bash
# Install better PDF library
pip install PyMuPDF

# Or ensure PyPDF2 is updated
pip install --upgrade PyPDF2
```

## 🎯 **Real vs Mock Data**

The system intelligently uses:
- **Real APIs first** - Live data from DuckDuckGo, Wikipedia, etc.
- **Smart fallback** - Mock data only when real APIs fail
- **Transparent reporting** - Shows source of each result
- **Quality metrics** - Relevance scores for all content

## 📈 **Scaling Up**

To add more APIs:
```python
# Add to WebAgent._search_real_web()
google_results = await self._search_google(query, max_results//4)
results.extend(google_results)

# Add to WebAgent  
async def _search_google(self, query, max_results):
    # Your Google Custom Search implementation
    pass
```

## 💡 **Pro Tips**

1. **Set OpenAI key** for best summaries
2. **Use specific queries** for better results  
3. **Include PDFs** for comprehensive analysis
4. **Check source diversity** in outputs
5. **Customize APIs** in WebAgent class

---

**Real APIs • AI Integration • Professional Output • One Simple File** ⚡# Multi-Agent Research Assistant

**Simple, powerful research automation in one file.**

## 🚀 Quick Start

1. **Install:**
```bash
pip install aiohttp beautifulsoup4 requests PyPDF2
```

2. **Run:**
```bash
python main.py
```

That's it! 🎉

## 📋 What It Does

- **Web Search**: Finds relevant articles and content
- **PDF Analysis**: Extracts text and key points from PDFs  
- **Smart Summary**: Creates comprehensive research summaries
- **Professional Reports**: Generates markdown reports

## 💡 Usage

```python
import asyncio
from main import ResearchAssistant

async def research():
    assistant = ResearchAssistant()
    
    results = await assistant.research(
        query="your research topic here",
        pdf_files=["document.pdf"],  # optional
        max_web_results=8
    )
    
    print(f"Done! Check: outputs/task_{results['task_id']}/")

asyncio.run(research())
```

## 📁 Output

Each research task creates:
- `report.md` - Full research report
- `summary.md` - Executive summary  
- `results.json` - Complete data

## 🔧 Optional Features

**For PDF support:**
```bash
pip install PyMuPDF  # Better quality
# OR
pip install PyPDF2   # Lighter weight
```

**For AI summarization:**
```bash
pip install openai
# Set OPENAI_API_KEY environment variable
```

## 🎯 Example Output

```
🔬 Multi-Agent Research Assistant
========================================
INFO: 🔍 Searching web for: artificial intelligence in healthcare
INFO: 📝 Creating summary for 3 web + 0 PDF sources
INFO: 📊 Generating report for: artificial intelligence in healthcare
INFO: 💾 Results saved to: outputs/task_a1b2c3d4

✅ Research Complete!
📁 Task ID: a1b2c3d4
🌐 Web sources: 3
📄 PDF sources: 0
📂 Check: outputs/task_a1b2c3d4/
```

## 🛠️ Troubleshooting

**Import errors?**
```bash
pip install -r requirements.txt
```

**No PDF support?**
```bash
pip install PyMuPDF
```

**Want better web search?**
- The current version uses mock search results
- Replace `WebAgent.search()` with real API calls (Google, Bing, etc.)

---

**Everything in one file. No complexity. Just results.** ⚡