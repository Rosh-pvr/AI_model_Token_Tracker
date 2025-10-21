# 🚀 QUICK START - Multi-Provider Edition

## Get Started in 3 Minutes!

### Step 1: Install (1 min)
```bash
pip install -r requirements.txt
```

### Step 2: Configure API Keys (1 min)
```bash
cp .env.example .env
```

Edit `.env` and add the API keys you need:

**Required for tracing:**
- `LANGCHAIN_API_KEY` → https://smith.langchain.com

**Pick at least one provider:**
- `GOOGLE_API_KEY` → https://aistudio.google.com/app/apikey
- `OPENAI_API_KEY` → https://platform.openai.com/api-keys
- `ANTHROPIC_API_KEY` → https://console.anthropic.com/settings/keys
- Hugging Face → No key needed (runs locally)

Set default provider:
```
LLM_PROVIDER=openai
```

### Step 3: Run! (30 sec)
```bash
# CLI
python token_tracker.py

# Web Dashboard
python dashboard_app.py
# → http://localhost:5000
```

---

## 🎯 Choose Your Provider

| Provider | Best For | Cost |
|----------|----------|------|
| 🤗 **Hugging Face** | Testing, Offline | Free |
| 💎 **Gemini** | Quality, Speed | Free tier |
| 🤖 **OpenAI** | Production, GPT-4 | Pay-per-use |
| 🧠 **Anthropic** | Long context, Claude | Pay-per-use |

---

## 🔄 Quick Provider Switch

Edit `.env`:
```
LLM_PROVIDER=gemini      # Use Gemini
LLM_PROVIDER=openai      # Use OpenAI
LLM_PROVIDER=anthropic   # Use Claude
LLM_PROVIDER=huggingface # Use HF (free)
```

Or use web dashboard dropdown!

---

**That's it! Start tracking! 🎉**

See README.md for details.
