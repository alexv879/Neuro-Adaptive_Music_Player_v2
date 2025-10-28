# GitHub Quick Start Commands

**Copy-paste these commands to upload your project to GitHub**

---

## 🚀 Fast Setup (5 Minutes)

### Step 1: Create Repository on GitHub Website

1. Go to https://github.com/new
2. Repository name: `Neuro-Adaptive-Music-Player-v2`
3. Description: `Real-time EEG emotion recognition with deep learning`
4. Choose Public or Private
5. **DO NOT** check any initialization boxes
6. Click "Create repository"

---

### Step 2: Run These Commands

**Open your terminal/Git Bash and run:**

```bash
# Navigate to project folder
cd "D:\AIUniversity\Applied Signals and Images Processing\assessment1\Neuro-Adaptive Music Player v2"

# Check if git is initialized (skip if already a repo)
git status
```

**If you see "not a git repository", run:**

```bash
# Initialize git
git init

# Add all files
git add .

# Create first commit
git commit -m "Initial commit: Complete Neuro-Adaptive Music Player v2 with research documentation"

# Rename branch to main
git branch -M main
```

**Replace YOUR_USERNAME with your actual GitHub username:**

```bash
# Connect to GitHub (CHANGE YOUR_USERNAME!)
git remote add origin https://github.com/YOUR_USERNAME/Neuro-Adaptive-Music-Player-v2.git

# Push to GitHub
git push -u origin main
```

**Example** (Alexandru's repository):
```bash
git remote add origin https://github.com/alexv879/Neuro-Adaptive_Music_Player_v2.git
git push -u origin main
```

---

## ✅ Verify Upload

Visit: `https://github.com/YOUR_USERNAME/Neuro-Adaptive-Music-Player-v2`

You should see your README.md and all files!

---

## 🔄 Future Updates

When you make changes:

```bash
# Navigate to project
cd "D:\AIUniversity\Applied Signals and Images Processing\assessment1\Neuro-Adaptive Music Player v2"

# Add all changes
git add .

# Commit with message
git commit -m "Description of changes"

# Push to GitHub
git push
```

---

## 🔑 Authentication

If prompted for password, use a **Personal Access Token**:

1. Go to: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Select scope: `repo`
4. Copy token and use as password

---

## ❌ Troubleshooting

**Already initialized as git repo?**
```bash
# Check remote
git remote -v

# If wrong or empty, set it:
git remote set-url origin https://github.com/YOUR_USERNAME/Neuro-Adaptive-Music-Player-v2.git

# Or add if none exists:
git remote add origin https://github.com/YOUR_USERNAME/Neuro-Adaptive-Music-Player-v2.git
```

**Accidentally uploaded .env file?**
```bash
git rm --cached .env
git commit -m "Remove .env file"
git push
# Then change your API keys immediately!
```

**Need to undo last commit?**
```bash
git reset --soft HEAD~1
```

---

## 📦 Complete Command Sequence (Copy All)

```bash
# 1. Navigate to project
cd "D:\AIUniversity\Applied Signals and Images Processing\assessment1\Neuro-Adaptive Music Player v2"

# 2. Initialize (if needed)
git init

# 3. Stage all files
git add .

# 4. Commit
git commit -m "Initial commit: Complete Neuro-Adaptive Music Player v2 with research documentation"

# 5. Set branch name
git branch -M main

# 6. Add remote (CHANGE YOUR_USERNAME!)
git remote add origin https://github.com/YOUR_USERNAME/Neuro-Adaptive-Music-Player-v2.git

# 7. Push to GitHub
git push -u origin main
```

---

**That's it! Your project is now on GitHub!** 🎉

For detailed instructions, see: [GITHUB_SETUP.md](GITHUB_SETUP.md)
