# GitHub Setup Guide

**Step-by-step instructions to upload your Neuro-Adaptive Music Player v2 to GitHub**

---

## Prerequisites

1. **Git installed** on your computer
   - Check: `git --version`
   - If not installed: Download from https://git-scm.com/downloads

2. **GitHub account** created
   - Sign up at https://github.com if you don't have one

---

## Option 1: Creating a New Repository (Recommended)

### Step 1: Create Repository on GitHub

1. Go to https://github.com
2. Click the **"+"** icon (top right) → **"New repository"**
3. Fill in the details:
   - **Repository name**: `Neuro-Adaptive-Music-Player-v2` (or your preferred name)
   - **Description**: `Real-time EEG-based emotion recognition with deep learning and adaptive music recommendation`
   - **Visibility**: Choose **Public** or **Private**
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
4. Click **"Create repository"**

### Step 2: Prepare Your Local Repository

Open your terminal/command prompt and navigate to your project:

```bash
cd "D:\AIUniversity\Applied Signals and Images Processing\assessment1\Neuro-Adaptive Music Player v2"
```

### Step 3: Initialize Git (if not already initialized)

Check if git is already initialized:

```bash
git status
```

If you get an error "not a git repository", initialize it:

```bash
git init
```

### Step 4: Stage All Files

Add all your files to git (excluding those in .gitignore):

```bash
# Stage all files
git add .

# Check what will be committed
git status
```

**Expected output**: You should see all your project files staged (in green)

### Step 5: Create Your First Commit

```bash
git commit -m "Initial commit: Complete Neuro-Adaptive Music Player v2 with research documentation"
```

### Step 6: Connect to GitHub Repository

Replace `YOUR_USERNAME` with your actual GitHub username:

```bash
git remote add origin https://github.com/YOUR_USERNAME/Neuro-Adaptive-Music-Player-v2.git
```

**Example**:
```bash
git remote add origin https://github.com/alexv879/Neuro-Adaptive_Music_Player_v2.git
```

### Step 7: Push to GitHub

```bash
# Push to main branch
git push -u origin main
```

**If you get an error about "master" vs "main":**

```bash
# Rename branch to main
git branch -M main

# Then push
git push -u origin main
```

**If prompted for authentication:**
- Use your GitHub username
- For password, use a **Personal Access Token** (not your GitHub password)
  - Create token at: https://github.com/settings/tokens
  - Select scopes: `repo` (full control of private repositories)

---

## Option 2: Adding to Existing Repository

If you already have a GitHub repository in your parent folder:

### Check Current Remote

```bash
cd "D:\AIUniversity\Applied Signals and Images Processing\assessment1\Neuro-Adaptive Music Player v2"
git remote -v
```

### If Already Connected

```bash
# Stage changes
git add .

# Commit
git commit -m "feat: Add complete Neuro-Adaptive Music Player v2"

# Push
git push
```

---

## Verification

After pushing, verify your repository:

1. Go to `https://github.com/YOUR_USERNAME/Neuro-Adaptive-Music-Player-v2`
2. You should see:
   - ✅ README.md displayed on the main page
   - ✅ All source files in `src/` folder
   - ✅ Research documentation (RESEARCH_PAPER.md, etc.)
   - ✅ Examples in `examples/` folder
   - ✅ LICENSE file
   - ❌ **NO .env file** (should be excluded by .gitignore)
   - ❌ **NO large data files** (should be excluded)

---

## Important Security Check

**CRITICAL**: Make sure sensitive files are NOT uploaded:

```bash
# Check what's being tracked
git ls-files | grep -E "\.env$|credentials|secrets|\.pkl$|\.h5$"
```

**This should return EMPTY**. If you see any sensitive files:

```bash
# Remove from git (but keep locally)
git rm --cached .env
git rm --cached path/to/sensitive/file

# Commit the removal
git commit -m "Remove sensitive files"

# Push changes
git push
```

---

## Repository Structure

After successful upload, your GitHub repo should look like:

```
Neuro-Adaptive-Music-Player-v2/
├── .github/                    # GitHub workflows (if any)
├── .gitignore                  # Git ignore rules
├── data/                       # Data directory (empty, just structure)
├── docs/                       # Additional documentation
├── examples/                   # Example usage scripts
│   ├── 01_complete_pipeline.py
│   └── README.md
├── models/                     # Model directory (empty, just structure)
├── src/                        # Source code
│   ├── config.py
│   ├── eeg_preprocessing.py
│   ├── eeg_features.py
│   ├── emotion_recognition_model.py
│   ├── model_personalization.py
│   ├── data_loaders.py
│   ├── live_eeg_handler.py
│   ├── music_recommendation.py
│   ├── llm_music_recommender.py
│   └── utils.py
├── tests/                      # Unit tests
├── .env.example                # Environment template
├── ALGORITHMS.md               # Algorithm documentation
├── ARCHITECTURE.md             # System architecture
├── CHANGELOG.md                # Version history
├── CITATIONS.md                # Quick citation reference
├── CONTRIBUTING.md             # Contribution guidelines
├── LICENSE                     # License file
├── README.md                   # Main documentation
├── RESEARCH_PAPER.md           # Academic research paper
├── RESEARCH_REFERENCES.md      # Complete bibliography
├── requirements.txt            # Python dependencies
├── SECURITY.md                 # Security policy
├── VERIFICATION_REPORT.md      # Verification report
└── VERIFICATION_SUMMARY.txt    # Verification summary
```

---

## Adding Repository Topics (Tags)

On your GitHub repository page:

1. Click **"⚙️ Settings"** (or the gear icon near "About")
2. Add topics/tags:
   - `eeg`
   - `emotion-recognition`
   - `deep-learning`
   - `brain-computer-interface`
   - `bci`
   - `music-recommendation`
   - `tensorflow`
   - `python`
   - `neuroscience`
   - `signal-processing`
   - `affective-computing`

---

## Making Your Repository Look Professional

### 1. Add a Repository Description

Click **"⚙️"** next to "About" and add:
```
Real-time EEG-based emotion recognition using CNN+BiLSTM with adaptive music recommendation. Production-ready code with 45+ research citations. By Alexandru Emanuel Vasile.
```

### 2. Pin Important Files

GitHub will automatically highlight:
- README.md (main page)
- LICENSE (license badge)
- CONTRIBUTING.md (contribution guidelines)

### 3. Add Repository Website (Optional)

If you have a demo or documentation website, add it in the "About" section.

---

## Common Issues and Solutions

### Issue 1: "Failed to push - remote rejected"

**Solution**: Pull first, then push
```bash
git pull origin main --rebase
git push origin main
```

### Issue 2: "Large files detected"

**Solution**: Remove large files and use .gitignore
```bash
# Find large files
find . -type f -size +100M

# Add to .gitignore
echo "path/to/large/file" >> .gitignore

# Remove from git
git rm --cached path/to/large/file
```

### Issue 3: "Authentication failed"

**Solution**: Use Personal Access Token instead of password
1. Go to https://github.com/settings/tokens
2. Generate new token (classic)
3. Select `repo` scope
4. Copy token and use as password when prompted

### Issue 4: ".env file uploaded by mistake"

**Solution**: Remove from git history
```bash
# Remove from tracking
git rm --cached .env

# Add to .gitignore (should already be there)
echo ".env" >> .gitignore

# Commit changes
git commit -m "Remove .env from tracking"

# Push
git push

# For sensitive data, consider changing API keys!
```

---

## Next Steps After Upload

1. **Share your repository**
   - Copy URL: `https://github.com/YOUR_USERNAME/Neuro-Adaptive-Music-Player-v2`
   - Add to your CV/portfolio

2. **Set up branch protection** (for collaborative projects)
   - Settings → Branches → Add rule for `main`
   - Require pull request reviews

3. **Enable GitHub Pages** (optional, for documentation)
   - Settings → Pages → Source: main branch, /docs folder

4. **Add badges to README** (already included)
   - Build status
   - Code coverage
   - License
   - Python version

5. **Create releases**
   - Go to "Releases" → "Create a new release"
   - Tag: `v2.0.0`
   - Title: `v2.0 - Production Release with Research Documentation`

---

## Updating Your Repository Later

When you make changes to your code:

```bash
# Navigate to project
cd "D:\AIUniversity\Applied Signals and Images Processing\assessment1\Neuro-Adaptive Music Player v2"

# Check status
git status

# Stage changes
git add .

# Commit with descriptive message
git commit -m "fix: Improve preprocessing performance by 10%"

# Push to GitHub
git push
```

### Commit Message Conventions

Use conventional commits:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `refactor:` - Code refactoring
- `test:` - Adding tests
- `perf:` - Performance improvements

**Examples**:
```bash
git commit -m "feat: Add real-time EEG streaming support"
git commit -m "fix: Resolve channel name mismatch in FAA computation"
git commit -m "docs: Update research references with new citations"
git commit -m "perf: Optimize band power extraction (5x speedup)"
```

---

## Additional Resources

- **GitHub Docs**: https://docs.github.com
- **Git Cheat Sheet**: https://education.github.com/git-cheat-sheet-education.pdf
- **Markdown Guide**: https://www.markdownguide.org
- **GitHub Desktop** (GUI alternative): https://desktop.github.com

---

## Getting Help

If you encounter issues:

1. **Check git status**: `git status`
2. **View recent commits**: `git log --oneline -5`
3. **Check remote**: `git remote -v`
4. **GitHub Support**: https://support.github.com

---

**That's it! Your professional EEG emotion recognition project is now on GitHub!** 🎉

Remember to:
- ✅ Keep .env file private (use .env.example for others)
- ✅ Never commit API keys or secrets
- ✅ Add datasets locally (they're in .gitignore)
- ✅ Commit regularly with clear messages
- ✅ Update documentation as you improve the code
