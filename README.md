# Token Tracking Tool for LLM Apps

Track token usage, latency, and costs across **4 major LLM providers**!

## 🎯 Supported Providers

✅ **Hugging Face** - Free, runs locally (flan-t5-base)  
✅ **Google Gemini** - Gemini 2.5 Pro  
✅ **OpenAI** - GPT-4o-mini, GPT-4, GPT-3.5-turbo  
✅ **Anthropic** - Claude 3.5 Sonnet, Claude 3 Opus  

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
- `LANGCHAIN_API_KEY` - From https://smith.langchain.com
- `GOOGLE_API_KEY` - From https://aistudio.google.com/app/apikey
- `OPENAI_API_KEY` - From https://platform.openai.com/api-keys
- `ANTHROPIC_API_KEY` - From https://console.anthropic.com/settings/keys

Set your default provider:
```
LLM_PROVIDER=openai  # or huggingface, gemini, anthropic
```

### 3. Run the Tool

**CLI Version:**
```bash
python token_tracker.py
```

**Web Dashboard:**
```bash
python dashboard_app.py
# Open: http://localhost:5000
```

## 🔄 Switch Between Providers

**Method 1: Edit .env file**
```env
LLM_PROVIDER=openai      # Use OpenAI
LLM_PROVIDER=gemini      # Use Gemini
LLM_PROVIDER=anthropic   # Use Anthropic
LLM_PROVIDER=huggingface # Use Hugging Face
```

**Method 2: Use Web Dashboard**
Select provider from dropdown for each query!

## 📊 Model Configuration

| Provider | Default Model | Can Change To |
|----------|--------------|---------------|
| Hugging Face | flan-t5-base | Any HF model |
| Gemini | gemini-2.5-pro | gemini-2.5-flash, etc. |
| OpenAI | gpt-4o-mini | gpt-4, gpt-3.5-turbo |
| Anthropic | claude-3-5-sonnet | claude-3-opus, claude-3-haiku |

## 📈 Features

- **Real-time token tracking** for all providers
- **Cost estimation** per provider
- **Performance metrics** (latency, tokens/call)
- **Local JSON logs** with provider tags
- **LangSmith dashboard** integration
- **Web interface** with provider selection
- **Comparison analytics** across providers

## 📁 View Results

1. **Local logs:** `token_usage.json` or `dashboard_logs.json`
2. **LangSmith:** https://smith.langchain.com
3. **Web dashboard:** http://localhost:5000

## 🔒 Security

- Never commit `.env` file (protected by `.gitignore`)
- Use `.env.example` as template
- Rotate keys if exposed

## 🐛 Troubleshooting

**API Key Error:**
- Make sure API key is set in `.env` for your chosen provider

**Model Not Found:**
- Check model name in `MODEL_CONFIGS` dict in the code
- Update to latest model names from provider documentation

**torch version error:**
- Change to `torch>=2.2.0` in requirements.txt

## 🎯 Provider-Specific Notes

### Hugging Face
- Runs locally, no API key needed
- First run downloads model (~500MB)
- Slower but free

### Google Gemini
- Requires internet connection
- Fast, high quality
- Free tier available

### OpenAI
- Industry-standard models
- Pay-per-token pricing
- GPT-4o-mini recommended for cost

### Anthropic
- Claude 3.5 Sonnet is latest
- Known for long context
- Competitive pricing


**QA TESTING PICS:**


<img width="1920" height="1080" alt="Screenshot (28)" src="https://github.com/user-attachments/assets/10766253-7302-40a6-8f98-66e03e06592b" />






<img width="1920" height="1080" alt="Screenshot (29)" src="https://github.com/user-attachments/assets/b3011ad2-bdc6-42a8-acc7-496d767b1354" />


