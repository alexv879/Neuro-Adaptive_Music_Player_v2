# API Key Security Best Practices

## 🔒 Critical Security Guidelines

### ❌ NEVER Do This:
```python
# DON'T hardcode API keys in code
api_key = "sk-abc123xyz..."  # ❌ WRONG!

# DON'T commit .env files
git add .env  # ❌ WRONG!

# DON'T share API keys in chat/email
"Here's my key: sk-..."  # ❌ WRONG!
```

### ✅ ALWAYS Do This:
```python
# ✅ Load from environment variables
import os
api_key = os.environ.get("OPENAI_API_KEY")

# ✅ Use .env files (excluded from git)
# Create .env from .env.example
cp .env.example .env

# ✅ Verify .env is gitignored
git check-ignore .env  # Should output: .env
```

---

## 🛡️ Setup: Secure Configuration

### Step 1: Create Your .env File

```bash
# Copy the template
Copy-Item .env.example .env  # Windows PowerShell
# or
cp .env.example .env  # Linux/Mac
```

### Step 2: Add Your API Key

Edit `.env` and add your OpenAI API key:
```env
OPENAI_API_KEY=sk-your-actual-api-key-here
OPENAI_MODEL=gpt-4o
OPENAI_TEMPERATURE=0.7
```

### Step 3: Verify Security

```bash
# Check .env is NOT tracked by git
git ls-files | grep ".env"
# Should only show: .env.example (not .env)

# Verify .gitignore contains .env
grep "^\.env$" .gitignore
# Should output: .env

# Test .env is properly ignored
git check-ignore .env
# Should output: .env
```

---

## 🚨 If You Accidentally Commit .env

### Immediate Actions:

1. **Remove from Git (but keep locally)**:
```bash
git rm --cached .env
git commit -m "Remove .env from tracking"
git push
```

2. **Rotate Your API Key**:
   - Go to [OpenAI Platform](https://platform.openai.com/api-keys)
   - Delete the exposed key
   - Create a new key
   - Update your `.env` file with new key

3. **Verify Clean History**:
```bash
# Check if .env exists in any commit
git log --all --full-history -- .env
# Should be empty after removal
```

---

## 🔧 Automated Protection

### Pre-commit Hook (Recommended)

Install the pre-commit hook to prevent accidental commits:

```bash
# Option 1: Use custom hooks directory (preferred)
git config core.hooksPath .githooks

# Option 2: Copy to .git/hooks/ (manual)
cp .githooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit  # Linux/Mac only
```

The hook will automatically:
- ✅ Block commits containing `.env` files
- ✅ Warn about other sensitive files
- ✅ Provide helpful error messages

### GitHub Actions Security Check

This repository includes a GitHub Actions workflow that:
- ✅ Scans for `.env` files in every push/PR
- ✅ Checks for hardcoded API keys
- ✅ Verifies `.gitignore` configuration
- ✅ Fails CI if security issues detected

See: `.github/workflows/security-checks.yml`

---

## 📋 Security Checklist

Before every commit, verify:

- [ ] `.env` file is NOT staged (`git status` should not show .env)
- [ ] No API keys hardcoded in Python files
- [ ] `.env.example` updated (if you added new variables)
- [ ] `.gitignore` includes `.env` entry
- [ ] Pre-commit hook is installed

Before sharing code:

- [ ] API keys removed from all files
- [ ] `.env` not included in archives/zips
- [ ] Credentials not in screenshots or logs
- [ ] `.env.example` has placeholder values only

---

## 🔑 Environment Variables Guide

### Required Variables:
```env
OPENAI_API_KEY=sk-...  # Required for LLM recommendations
```

### Optional Variables:
```env
OPENAI_MODEL=gpt-4o              # Default: gpt-4o
OPENAI_TEMPERATURE=0.7           # Default: 0.7
SPOTIFY_CLIENT_ID=...            # For Spotify integration
SPOTIFY_CLIENT_SECRET=...        # For Spotify integration
SPOTIFY_REDIRECT_URI=...         # For Spotify integration
```

### Loading Order:
1. `.env` file (highest priority)
2. System environment variables
3. Default values in code

---

## 🆘 Troubleshooting

### "No OpenAI API key found"
**Solution**: 
1. Check `.env` file exists: `ls .env` (should exist)
2. Check key is set: `grep OPENAI_API_KEY .env`
3. Verify no extra spaces or quotes around key
4. Restart application after editing `.env`

### "OpenAI authentication failed"
**Solution**:
1. Verify key format: Must start with `sk-`
2. Check key is active at [OpenAI Platform](https://platform.openai.com/api-keys)
3. Ensure no line breaks or hidden characters in `.env`

### ".env file committed by mistake"
**Solution**: Follow [If You Accidentally Commit .env](#-if-you-accidentally-commit-env) section above

---

## 📚 Additional Resources

- [OpenAI API Keys Management](https://platform.openai.com/api-keys)
- [OpenAI Best Practices](https://platform.openai.com/docs/guides/production-best-practices)
- [GitHub Security Best Practices](https://docs.github.com/en/code-security/getting-started/best-practices-for-preventing-data-leaks-in-your-organization)
- [Git Secrets Prevention](https://github.com/awslabs/git-secrets)

---

## ⚠️ Important Reminders

1. **API Keys are Sensitive**: Treat them like passwords
2. **Regular Rotation**: Change keys every 3-6 months
3. **Monitor Usage**: Check [OpenAI Usage Dashboard](https://platform.openai.com/usage) regularly
4. **Set Spending Limits**: Configure budget limits in OpenAI account
5. **Use Separate Keys**: Different keys for dev/prod/testing

---

**🔐 When in doubt, rotate your key!**

If you suspect your API key may have been exposed, rotate it immediately. It's better to be safe than sorry.
