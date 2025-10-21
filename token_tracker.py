#!/usr/bin/env python3
"""
Token Tracking Tool for LLM Apps
Supports: Hugging Face, Google Gemini, OpenAI (ChatGPT), Anthropic (Claude)
"""

import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from transformers import pipeline as hf_pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langsmith import traceable

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "Token_Tracking_Multi_Provider")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["TOKENIZERS_PARALLELISM"] = os.getenv("TOKENIZERS_PARALLELISM", "false")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "huggingface").lower()

MODEL_CONFIGS = {
    "huggingface": {
        "name": "google/flan-t5-base",
        "display": "🤗 Hugging Face (flan-t5-base)"
    },
    "gemini": {
        "name": "models/gemini-2.5-pro",
        "display": "💎 Google Gemini 2.5 Pro"
    },
    "openai": {
        "name": "gpt-4o-mini",
        "display": "🤖 OpenAI GPT-4o-mini"
    },
    "anthropic": {
        "name": "claude-3-5-sonnet-20241022",
        "display": "🧠 Anthropic Claude 3.5 Sonnet"
    }
}

def setup_model(provider=None):
    if provider is None:
        provider = LLM_PROVIDER
    provider = provider.lower()
    if provider not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Choose from: {list(MODEL_CONFIGS.keys())}"
        )
    config = MODEL_CONFIGS[provider]
    print(f"\nLoading {config['display']}...")
    if provider == "huggingface":
        model_name = config["name"]
        pipe = hf_pipeline(
            "text2text-generation",
            model=model_name,
            tokenizer=model_name,
            device=-1,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.7
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        print("✓ Hugging Face model loaded!")
    elif provider == "gemini":
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY not found. Set it in .env file.")
        llm = ChatGoogleGenerativeAI(
            model=config["name"],
            temperature=0.7,
            max_output_tokens=512
        )
        print("✓ Google Gemini model loaded!")
    elif provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found. Set it in .env file.")
        llm = ChatOpenAI(
            model=config["name"],
            temperature=0.7,
            max_tokens=512
        )
        print("✓ OpenAI model loaded!")
    elif provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY not found. Set it in .env file.")
        llm = ChatAnthropic(
            model=config["name"],
            temperature=0.7,
            max_tokens=512
        )
        print("✓ Anthropic Claude model loaded!")
    return llm

class TokenTracker:
    def __init__(self, log_file="token_usage.json"):
        self.log_file = log_file
        self.load_logs()
    def load_logs(self):
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                self.logs = json.load(f)
        else:
            self.logs = []
    def log_usage(self, prompt, response, input_tokens, output_tokens, latency, provider, model_name):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "model": model_name,
            "prompt": prompt,
            "response": response,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "latency_seconds": round(latency, 3)
        }
        self.logs.append(entry)
        self.save_logs()
        return entry
    def save_logs(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)
    def get_statistics(self, provider=None):
        logs = self.logs
        if provider:
            logs = [log for log in logs if log.get("provider") == provider]
        if not logs:
            return {"message": "No logs available"}
        total_calls = len(logs)
        total_tokens = sum(log["total_tokens"] for log in logs)
        total_input = sum(log["input_tokens"] for log in logs)
        total_output = sum(log["output_tokens"] for log in logs)
        avg_latency = sum(log["latency_seconds"] for log in logs) / total_calls
        return {
            "total_calls": total_calls,
            "total_tokens": total_tokens,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "average_latency": round(avg_latency, 3),
            "average_tokens_per_call": round(total_tokens / total_calls, 2)
        }
    def print_statistics(self, provider=None):
        stats = self.get_statistics(provider)
        provider_text = f" ({provider.upper()})" if provider else " (ALL PROVIDERS)"
        print("\n" + "=" * 60)
        print(f"TOKEN USAGE STATISTICS{provider_text}")
        print("=" * 60)
        for key, value in stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        print("=" * 60 + "\n")

tracker = TokenTracker()

@traceable(name="Query Execution")
def run_query(llm, prompt_template, prompt_text, provider, model_name):
    start_time = time.time()
    prompt_input = prompt_template.format()
    response = llm.invoke(prompt_input)
    if hasattr(response, "content"):
        result_text = response.content
    elif hasattr(response, "text"):
        result_text = response.text
    else:
        result_text = str(response)
    latency = time.time() - start_time
    input_tokens = len(prompt_text.split()) * 1.3
    output_tokens = len(result_text.split()) * 1.3
    tracker.log_usage(
        prompt_text,
        result_text,
        int(input_tokens),
        int(output_tokens),
        latency,
        provider,
        model_name
    )
    return result_text

def main():
    print("\n" + "=" * 60)
    print("TOKEN TRACKING TOOL FOR LLM APPS")
    print("Multi-Provider: HuggingFace | Gemini | OpenAI | Anthropic")
    print("=" * 60 + "\n")

    if not os.getenv("LANGCHAIN_API_KEY"):
        print("⚠️  WARNING: LANGCHAIN_API_KEY not set!")
        print("   LangSmith tracing will not work.\n")

    try:
        llm = setup_model()
        model_name = MODEL_CONFIGS[LLM_PROVIDER]["name"]
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Set the correct API key in .env")
        print("2. Set LLM_PROVIDER to: huggingface, gemini, openai, or anthropic")
        return

    print("\n--- Example 1: Explain Gravity ---")
    prompt_text = "Explain gravity to a 10-year-old in simple terms."
    prompt_template = PromptTemplate.from_template(prompt_text)
    answer = run_query(llm, prompt_template, prompt_text, LLM_PROVIDER, model_name)
    print(f"Response: {answer}")

    print("\n--- Example 2: What is Photosynthesis ---")
    prompt_text2 = "What is photosynthesis? Explain briefly."
    custom_prompt = PromptTemplate.from_template(prompt_text2)
    answer2 = run_query(llm, custom_prompt, prompt_text2, LLM_PROVIDER, model_name)
    print(f"Response: {answer2}")

    tracker.print_statistics()
    tracker.print_statistics(provider=LLM_PROVIDER)
    print("\n✓ Logs saved to 'token_usage.json'")
    print("✓ Check LangSmith dashboard: https://smith.langchain.com")
    print("\nTo switch providers, edit LLM_PROVIDER in .env file")
    print(f"\nAvailable providers: {', '.join(MODEL_CONFIGS.keys())}")

if __name__ == "__main__":
    main()
