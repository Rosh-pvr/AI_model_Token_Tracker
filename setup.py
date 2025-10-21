#!/usr/bin/env python3
import os, shutil

print("🔧 Token Tracker Setup - Multi-Provider Edition")
print("Supports: Hugging Face, Gemini, OpenAI, Anthropic\n")

if not os.path.exists(".env"):
    if os.path.exists(".env.example"):
        shutil.copy(".env.example", ".env")
        print("✓ Created .env from template")
        print("⚠️  Edit .env and add your API keys!\n")
    else:
        print("❌ .env.example not found")
else:
    print("⚠️  .env already exists\n")

print("Next steps:")
print("1. Edit .env file:")
print("   - Add LANGCHAIN_API_KEY (required)")
print("   - Add provider keys (GOOGLE_API_KEY, OPENAI_API_KEY, etc.)")
print("   - Set LLM_PROVIDER (huggingface, gemini, openai, anthropic)")
print("2. pip install -r requirements.txt")
print("3. python token_tracker.py")
