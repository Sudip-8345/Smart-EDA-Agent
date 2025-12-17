# AI EDA Agent

An intelligent Exploratory Data Analysis tool powered by AI. Upload any CSV file and get instant visualizations, statistical insights, and AI-generated analysis through a Streamlit interface.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Screenshots

### Dashboard Overview

The main interface has a sidebar for file upload and model selection, with tabs for different analysis views.

<img width="1788" height="894" alt="image" src="https://github.com/user-attachments/assets/a09053c6-c934-42ad-a80e-1fecd7c1bfed" />


### Visualizations Tab
- Correlation Heatmap - Identify relationships between variables
- Distribution Plots - Histograms with KDE overlay
- Box Plots - Outlier detection and quartile analysis
- Scatter Plots - Feature relationships with trend lines
- Violin Plots - Distribution by category
---
<img width="1788" height="894" alt="image" src="https://github.com/user-attachments/assets/3e266268-631d-4fb4-a3b4-100883475c18" />



### AI Insights Tab
- Analysis Plan - AI-generated data exploration strategy
- Key Insights - Actionable findings from your data
- Plot Summaries - AI explanations for each visualization

---

## Features

| Feature | Description |
|---------|-------------|
| CSV Upload | Drag and drop or browse to upload datasets |
| 15+ Chart Types | Heatmaps, histograms, boxplots, scatter, violin, etc. |
| AI Analysis | Get insights powered by Gemini, GPT, or local Ollama |
| Interactive Plots | Zoom, pan, hover with Plotly |
| Export Options | Download charts as PNG or HTML |
| Multi-Model Support | Switch between AI providers seamlessly |
| Local First | Works offline with Ollama models |

---
<img width="1788" height="894" alt="image" src="https://github.com/user-attachments/assets/0e28e3ac-efd0-4f99-9866-7229fdb81b49" />
---
## Tech Stack

| Category | Technology |
|----------|------------|
| Frontend | Streamlit |
| Visualization | Plotly, Matplotlib, Seaborn |
| Data Processing | Pandas, NumPy |
| AI Providers | Google Gemini, OpenAI GPT-4, Ollama (local) |
| HTTP Client | HTTPX |
| Environment | Python 3.10+, python-dotenv |

---

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/AI-EDA-Agent.git
cd AI-EDA-Agent
```

### 2. Create Virtual Environment
```bash
python -m venv myenv

# Windows
myenv\Scripts\activate

# Linux/Mac
source myenv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install streamlit plotly httpx python-dotenv openai
```

### 4. Configure API Keys (Optional)
Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

You can get API keys from:
- Google Gemini: https://aistudio.google.com/apikey
- OpenAI: https://platform.openai.com/api-keys

### 5. Install Ollama for Local AI (Recommended)
Download from https://ollama.com, then pull a fast model:
```bash
ollama pull llama3.2:1b
```

### 6. Run the App
```bash
python -m streamlit run streamlit_app.py
```

Open http://localhost:8501 in your browser.

---
<img width="1788" height="894" alt="image" src="https://github.com/user-attachments/assets/47ba785d-abd0-41b8-8b69-0d26bca458a8" />
---
## Project Structure

```
AI-EDA-Agent/
├── streamlit_app.py      # Main Streamlit frontend
├── app.py                # Alternative Chainlit backend
├── visualize.py          # Visualization generators
├── ai_text_analysis.py   # AI text generation functions
├── ai_vision_analysis.py # AI image/chart analysis
├── DataFrame_to_Str.py   # DataFrame info converter
├── save_figures.py       # Figure saving utilities
├── requirements.txt      # Python dependencies
├── .env                  # API keys (create this)
├── .gitignore            # Git ignore rules
└── sample_data/          # Example datasets
    ├── tip.csv
    └── Students Social Media Addiction.csv
```

---

## Supported AI Models

| Model | Provider | Speed | Rate Limits | Best For |
|-------|----------|-------|-------------|----------|
| Llama 3.2 1B | Ollama (Local) | Very Fast | None | Default, quick analysis |
| Gemini 2.0 Flash | Google | Fast | 15/min free | Quality insights |
| GPT-4o Mini | OpenAI | Fast | Pay-per-use | Detailed analysis |
| GPT-4o | OpenAI | Medium | Pay-per-use | Best quality |

---

## Challenges and Solutions

### 1. Gemini API Rate Limits

The free tier has strict limits and exhausts quickly during testing.

**Error:**
```
429 You exceeded your current quota
```

**Solution:** Switched to local Ollama models as the default. The `llama3.2:1b` model runs locally with no rate limits.

---

### 2. Large Vision Models Are Slow

The `llama3.2-vision` model (7.9GB) was taking 30-60 seconds per request, making the app unusable for quick analysis.

**Solution:** Replaced with `llama3.2:1b` (1.3GB) which is about 5x faster. For vision-specific tasks, Gemini can be used when quota is available.

---

### 3. Deprecated Google API Package

The `google-generativeai` package shows deprecation warnings:
```
FutureWarning: All support for the google.generativeai package has ended
```

**Solution:** The package still works for now. Wrapped imports in try/except for graceful handling. Will need to migrate to `google-genai` package in future updates.

---

### 4. Streamlit Arrow Serialization Errors

Some DataFrame dtypes caused serialization errors when displaying in Streamlit:
```
ArrowTypeError: Could not convert dtype('int64')
```

**Solution:** Convert dtypes to strings before display using `.astype(str)` for the problematic columns in the data type information display.

---

### 5. PowerShell Execution Policy on Windows

Virtual environment activation was blocked by default Windows security settings:
```
Scripts\Activate.ps1 cannot be loaded because running scripts is disabled
```

**Solution:** Use Python directly without activating the environment:
```bash
.\myenv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Or change the execution policy:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

---

## Sample Output

### Correlation Heatmap Analysis
```
The dataset shows strong positive correlation (0.89) between 
'total_bill' and 'tip', suggesting customers tip proportionally. 
'Size' has moderate correlation with both, indicating larger 
groups spend more.
```

### Distribution Insights
```
- 68% of total bills fall between $10-$25
- Tip distribution is right-skewed with mean $3.00
- Weekend dinners show 23% higher average bills
- Party size of 2 is most common (45% of records)
```

---

## Configuration

### Change Default Model
Edit `streamlit_app.py` and modify the ACTIVE_MODEL variable:
```python
ACTIVE_MODEL = "llama3.2-1b"  # Fast local model
# ACTIVE_MODEL = "gemini-2.5-flash"  # Cloud model with quota
```

### Add Custom Visualizations
Edit `visualize.py` to add new chart types in the `generate_visuals()` function.

---

## Requirements

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
streamlit>=1.28.0
plotly>=5.18.0
httpx>=0.25.0
python-dotenv>=1.0.0
google-generativeai>=0.3.0
openai>=1.0.0
Pillow>=10.0.0
tabulate>=0.9.0
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -m 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- Streamlit - Python web framework
- Plotly - Interactive visualization library
- Ollama - Local LLM inference
- Google Gemini - Multimodal AI
- OpenAI - GPT models
