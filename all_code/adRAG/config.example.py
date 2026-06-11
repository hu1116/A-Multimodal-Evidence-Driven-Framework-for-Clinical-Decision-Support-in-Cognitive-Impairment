"""Example local configuration for the RAG-LLM module.

Copy this file to ``config.py`` (which is git-ignored) and fill in your own
credentials. NEVER commit real API keys.

    cp adRAG/config.example.py adRAG/config.py
"""

# DeepSeek API credentials (used by ReportToCsv.py and Feature2Txt/generate_rules.py).
# Set DEEPSEEK_API_KEY to your own key; leave the base URL as-is unless you use a proxy.
DEEPSEEK_API_KEY = "YOUR_DEEPSEEK_API_KEY"          # <-- replace with your key
DEEPSEEK_BASE_URL = "https://api.deepseek.com"      # public DeepSeek endpoint
