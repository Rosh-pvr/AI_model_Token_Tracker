

#!/usr/bin/env python3
'''
Token Tracking Dashboard - Multi-Provider Web Interface
Supports: Hugging Face, Gemini, OpenAI, Anthropic
'''

import os, json, time
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, jsonify, request
from transformers import pipeline as hf_pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langsmith import traceable

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "Token_Dashboard")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)

MODEL_CONFIGS = {
    "huggingface": {"name": "google/flan-t5-base", "display": "🤗 Hugging Face"},
    "gemini": {"name": "models/gemini-2.5-pro", "display": "💎 Gemini 2.5 Pro"},
    "openai": {"name": "gpt-4o-mini", "display": "🤖 GPT-4o-mini"},
    "anthropic": {"name": "claude-3-5-sonnet-20241022", "display": "🧠 Claude 3.5"}
}

class TokenTracker:
    def __init__(self):
        self.log_file = "dashboard_logs.json"
        self.logs = []
        self.models = {}
        if os.path.exists(self.log_file):
            with open(self.log_file) as f:
                self.logs = json.load(f)
    def setup_model(self, provider="huggingface"):
        if provider in self.models:
            return self.models[provider]
        if provider == "huggingface":
            pipe = hf_pipeline("text2text-generation", model="google/flan-t5-base",
                               device=-1, max_new_tokens=60, do_sample=True, temperature=0.7)
            model = HuggingFacePipeline(pipeline=pipe)
        elif provider == "gemini":
            if not os.getenv("GOOGLE_API_KEY"):
                raise ValueError("GOOGLE_API_KEY not set in .env")
            model = ChatGoogleGenerativeAI(model="models/gemini-2.5-pro", temperature=0.7, max_output_tokens=512)
        elif provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY not set in .env")
            model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=512)
        elif provider == "anthropic":
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise ValueError("ANTHROPIC_API_KEY not set in .env")
            model = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.7, max_tokens=512)
        else:
            raise ValueError(f"Unknown provider: {provider}")
        self.models[provider] = model
        return model

    @traceable(name="Dashboard Query")
    def process_query(self, query, provider="huggingface"):
        start = time.time()
        llm = self.setup_model(provider)
        prompt_template = PromptTemplate.from_template(query)
        prompt_input = prompt_template.format()
        response = llm.invoke(prompt_input)
        if hasattr(response, "content"):
            result_text = response.content
        elif hasattr(response, "text"):
            result_text = response.text
        else:
            result_text = str(response)
        latency = time.time() - start

        entry = {
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "model": MODEL_CONFIGS[provider]["name"],
            "query": query,
            "response": result_text,
            "input_tokens": int(len(query.split()) * 1.3),
            "output_tokens": int(len(result_text.split()) * 1.3),
            "total_tokens": int((len(query.split()) + len(result_text.split())) * 1.3),
            "latency_seconds": round(latency, 3)
        }
        self.logs.append(entry)
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)
        return entry

    def get_stats(self, provider=None):
        logs = self.logs
        if provider:
            logs = [l for l in logs if l.get("provider") == provider]
        if not logs:
            return {"total_calls": 0, "total_tokens": 0, "average_latency": 0, "average_tokens": 0}
        total = len(logs)
        tokens = sum(l["total_tokens"] for l in logs)
        latency = sum(l["latency_seconds"] for l in logs) / total
        return {
            "total_calls": total,
            "total_tokens": tokens,
            "average_latency": round(latency, 3),
            "average_tokens": round(tokens / total, 2)
        }

tracker = TokenTracker()

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
<html><head><title>Multi-Provider Token Dashboard</title>
<style>
body {font-family: Arial; margin: 20px; background: #f5f5f5;}
.container {max-width: 1200px; margin: auto;}
.header {background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px;}
.stats {display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px;}
.card {background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
.num {font-size: 2em; font-weight: bold; color: #3498db;}
.section {background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 30px;}
select, textarea {width: 100%; padding: 12px; margin-bottom: 12px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px;}
button {background: #3498db; color: white; padding: 12px 24px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px;}
button:hover {background: #2980b9;}
button:disabled {background: #bdc3c7; cursor: not-allowed;}
.result {background: #ecf0f1; padding: 15px; margin-top: 15px; border-radius: 4px; display: none;}
.log {background: #f8f9fa; padding: 12px; margin: 8px 0; border-radius: 4px; border-left: 4px solid #3498db;}
.badge {display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; margin-right: 8px;}
.badge-hf {background: #ffd700; color: black;}
.badge-gemini {background: #4285f4; color: white;}
.badge-openai {background: #10a37f; color: white;}
.badge-anthropic {background: #d97757; color: white;}
.loading {display: none; color: #3498db; margin-top: 10px;}
</style></head>
<body>
<div class="container">
<div class="header">
<h1>🎯 Multi-Provider Token Tracker</h1>
<p>HuggingFace • Gemini • OpenAI • Anthropic</p>
</div>
<div class="stats" id="stats"></div>
<div class="section">
<h2>Test Query</h2>
<select id="provider">
<option value="huggingface">🤗 Hugging Face</option>
<option value="gemini">💎 Google Gemini 2.5 Pro</option>
<option value="openai">🤖 OpenAI GPT-4o-mini</option>
<option value="anthropic">🧠 Anthropic Claude 3.5</option>
</select>
<textarea id="query" rows="3">Explain quantum computing simply.</textarea>
<button onclick="run()" id="runBtn">Run Query</button>
<div id="loading" class="loading">⏳ Processing...</div>
<div id="result" class="result"></div>
</div>
<div class="section"><h2>Recent Logs</h2><div id="logs"></div></div>
</div>
<script>
function getBadge(p){
const b={'huggingface':'badge-hf','gemini':'badge-gemini','openai':'badge-openai','anthropic':'badge-anthropic'};
return '<span class="badge '+b[p]+'">'+p.toUpperCase()+'</span>';
}
function load(){
fetch('/api/stats').then(r=>r.json()).then(d=>{
document.getElementById('stats').innerHTML='<div class="card"><div class="num">'+d.total_calls+'</div><div>Queries</div></div><div class="card"><div class="num">'+d.total_tokens+'</div><div>Tokens</div></div><div class="card"><div class="num">'+d.average_latency+'s</div><div>Latency</div></div><div class="card"><div class="num">'+d.average_tokens+'</div><div>Avg Tokens</div></div>';
});
fetch('/api/logs').then(r=>r.json()).then(d=>{
let html=d.slice(-5).reverse().map(l=>'<div class="log">'+getBadge(l.provider)+'<b>'+l.model+'</b><br><b>Q:</b> '+l.query+'<br><b>A:</b> '+l.response+'<br><small>🕒 '+l.timestamp+' | 📊 '+l.total_tokens+' tokens | ⚡ '+l.latency_seconds+'s</small></div>').join('');
document.getElementById('logs').innerHTML=html||'<p>No logs yet</p>';
});}
function run(){
let q=document.getElementById('query').value;
let p=document.getElementById('provider').value;
let r=document.getElementById('result');
let btn=document.getElementById('runBtn');
let load_=document.getElementById('loading');
r.style.display='none';
btn.disabled=true;
load_.style.display='block';
fetch('/api/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q,provider:p})})
.then(res=>res.json()).then(d=>{
if(d.error){r.innerHTML='<p style="color:red"><b>Error:</b> '+d.error+'</p>';}
else{r.innerHTML='<h3>Result</h3>'+getBadge(d.provider)+'<p><b>Model:</b> '+d.model+'</p><p><b>Response:</b> '+d.response+'</p><p><b>Tokens:</b> '+d.total_tokens+' ('+d.input_tokens+'+'+d.output_tokens+')</p><p><b>Latency:</b> '+d.latency_seconds+'s</p>';}
r.style.display='block';load_.style.display='none';btn.disabled=false;load();
}).catch(e=>{r.innerHTML='<p style="color:red">Error: '+e.message+'</p>';r.style.display='block';load_.style.display='none';btn.disabled=false;});}
load();setInterval(load,30000);
</script></body></html>
    '''

@app.route('/api/stats')
def api_stats():
    return jsonify(tracker.get_stats())

@app.route('/api/logs')
def api_logs():
    return jsonify(tracker.logs)

@app.route('/api/query', methods=['POST'])
def api_query():
    data = request.json
    try:
        result = tracker.process_query(data.get('query',''), data.get('provider','huggingface'))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("\n🚀 Multi-Provider Dashboard: http://localhost:5000")
    print("⚠️  Configure all API keys in .env file!")
    print("\nSupported providers:")
    for p, cfg in MODEL_CONFIGS.items():
        print(f"  • {cfg['display']}")
    app.run(debug=True, port=5000)
