# Environment Setup Guide

## Quick Start: Setting Up Your .env File

The Neuro-Adaptive Music Player v2 uses environment variables to manage API keys securely. Follow these steps to configure your environment:

### 1. Create Your .env File

Copy the example template to create your own `.env` file:

```bash
# On Windows (PowerShell)
Copy-Item .env.example .env

# On Linux/Mac
cp .env.example .env
```

### 2. Get Your OpenAI API Key

1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign in or create an account
3. Navigate to **API Keys** section
4. Click **"Create new secret key"**
5. Copy the key (starts with `sk-...`)

⚠️ **Important**: Save this key immediately - you won't be able to see it again!

### 3. Configure Your .env File

Open the `.env` file in a text editor and add your API key:

```env
# OpenAI API Configuration
OPENAI_API_KEY=sk-your-actual-api-key-here
OPENAI_MODEL=gpt-4o
OPENAI_TEMPERATURE=0.7
```

**Configuration Options**:

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key (required) | None | `sk-...` |
| `OPENAI_MODEL` | GPT model to use | `gpt-4o` | `gpt-4o`, `gpt-4`, `gpt-3.5-turbo` |
| `OPENAI_TEMPERATURE` | Creativity level (0-2) | `0.7` | `0.0` (focused) to `2.0` (creative) |

### 4. Verify Your Setup

Test that your configuration is working:

```bash
python examples/02_llm_recommendation_pipeline.py --mode simulated
```

You should see:
```
✓ LLM recommender initialized with gpt-4o
✓ OpenAI API connection successful
```

## Security Best Practices

### ✅ DO:
- ✅ Keep your `.env` file private (already in `.gitignore`)
- ✅ Use different API keys for development and production
- ✅ Rotate API keys regularly
- ✅ Set spending limits in your OpenAI account

### ❌ DON'T:
- ❌ Commit `.env` files to version control
- ❌ Share API keys in chat/email
- ❌ Hardcode API keys in your source code
- ❌ Use production keys for testing

## Troubleshooting

### Issue: "No OpenAI API key found"
**Solution**: Make sure your `.env` file exists and contains `OPENAI_API_KEY=sk-...`

### Issue: "OpenAI initialization failed: Incorrect API key"
**Solution**: 
1. Check that your key starts with `sk-`
2. Verify the key is active at [OpenAI Platform](https://platform.openai.com/api-keys)
3. Make sure there are no extra spaces or quotes around the key

### Issue: "Rate limit exceeded"
**Solution**: 
- You've hit your OpenAI API quota
- Check your usage at [OpenAI Usage](https://platform.openai.com/usage)
- Add credits or upgrade your plan

### Issue: "Module 'dotenv' not found"
**Solution**: Install the required package:
```bash
pip install python-dotenv
```

## Alternative: Using System Environment Variables

If you don't want to use `.env` files, you can set environment variables directly:

### Windows (PowerShell)
```powershell
$env:OPENAI_API_KEY = "sk-your-api-key-here"
```

### Linux/Mac (Bash)
```bash
export OPENAI_API_KEY="sk-your-api-key-here"
```

### Permanent Setup (Windows)
1. Search for "Environment Variables" in Windows
2. Click "Edit system environment variables"
3. Click "Environment Variables" button
4. Add `OPENAI_API_KEY` with your key

## Cost Considerations

Using the OpenAI API incurs costs based on tokens used:

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| GPT-4o | $2.50 | $10.00 |
| GPT-4 | $30.00 | $60.00 |
| GPT-3.5-turbo | $0.50 | $1.50 |

**Typical Usage**:
- Each music recommendation: ~200-500 tokens (~$0.001 - $0.005 with GPT-4o)
- 1 hour of use: ~50-100 recommendations (~$0.05 - $0.50)

💡 **Tip**: Start with `gpt-3.5-turbo` for testing to minimize costs.

## Advanced Configuration

### Using Multiple Environments

Create different `.env` files for different contexts:

```
.env.development   # For development
.env.production    # For production
.env.testing       # For automated tests
```

Load specific environment:
```python
from dotenv import load_dotenv
load_dotenv('.env.development')
```

### Custom Model Parameters

You can override defaults programmatically:

```python
from src.llm_music_recommender import LLMMusicRecommender

recommender = LLMMusicRecommender(
    model="gpt-4o",           # Use GPT-4o
    temperature=0.9,          # More creative
    max_tokens=1000           # Longer responses
)
```

## Support

- 📖 [OpenAI API Documentation](https://platform.openai.com/docs)
- 💬 [OpenAI Community Forum](https://community.openai.com/)
- 🐛 [Report Issues](https://github.com/alexv879/Neuro-Adaptive_Music_Player_v2/issues)

---

**Need Help?** Check the [CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md) for recent changes or the main [README.md](README.md) for general setup instructions.
