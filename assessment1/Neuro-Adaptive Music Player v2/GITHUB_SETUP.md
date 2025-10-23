# GitHub Repository Setup Guide

This guide will help you move the Neuro-Adaptive Music Player v2 to its own GitHub repository.

## Prerequisites

1. **Git installed**: Check with `git --version`
2. **GitHub account**: Sign up at [github.com](https://github.com)
3. **Git configured** with your credentials:
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```

## Step 1: Initialize Local Git Repository

Open a terminal in the project directory:

```bash
cd "d:\AIUniversity\Applied Signals and Images Processing\assessment1\Neuro-Adaptive Music Player v2"
```

Initialize git:

```bash
git init
```

Add all files:

```bash
git add .
```

Check what will be committed:

```bash
git status
```

Make the initial commit:

```bash
git commit -m "Initial commit: Neuro-Adaptive Music Player v2.0.0

- Complete EEG preprocessing pipeline with artifact detection
- Comprehensive feature extraction (band power, FAA, statistics)
- CNN+BiLSTM emotion recognition model with hierarchical outputs
- Extensive documentation (ARCHITECTURE.md, README.md)
- Repository infrastructure (LICENSE, CONTRIBUTING, CHANGELOG)
- Production-ready codebase with 100% docstring coverage

Core modules implemented:
- src/config.py: Configuration management
- src/eeg_preprocessing.py: Signal preprocessing
- src/eeg_features.py: Feature extraction
- src/emotion_recognition_model.py: Deep learning models

Total: 2,640 lines of production code + 3,500 lines of documentation"
```

## Step 2: Create GitHub Repository

### Option A: Via GitHub Web Interface (Recommended)

1. **Go to GitHub**: [https://github.com/new](https://github.com/new)

2. **Repository Settings**:
   - **Name**: `neuro-adaptive-music-player-v2`
   - **Description**: `Production-quality EEG-based emotion recognition system for adaptive music playback using CNN+BiLSTM deep learning`
   - **Visibility**: 
     - **Public** - If you want to showcase your work
     - **Private** - If you want to keep it confidential initially
   - **Initialize repository**: ⚠️ **DO NOT** check any boxes (no README, gitignore, or license)

3. **Click "Create repository"**

### Option B: Via GitHub CLI (Advanced)

If you have GitHub CLI installed:

```bash
gh repo create neuro-adaptive-music-player-v2 --public --description "Production-quality EEG-based emotion recognition for adaptive music" --source=.
```

## Step 3: Link Local Repository to GitHub

Copy the commands from your new GitHub repository page (replace `YOUR_USERNAME`):

```bash
git remote add origin https://github.com/YOUR_USERNAME/neuro-adaptive-music-player-v2.git
git branch -M main
git push -u origin main
```

**Alternative (SSH)**: If you have SSH keys set up:

```bash
git remote add origin git@github.com:YOUR_USERNAME/neuro-adaptive-music-player-v2.git
git branch -M main
git push -u origin main
```

## Step 4: Verify Upload

1. **Go to your repository**: `https://github.com/YOUR_USERNAME/neuro-adaptive-music-player-v2`

2. **Verify structure**:
   - ✅ `src/` directory with Python modules
   - ✅ `data/` directory with README
   - ✅ `models/`, `examples/`, `tests/`, `logs/` directories
   - ✅ Documentation files (README.md, ARCHITECTURE.md, etc.)
   - ✅ LICENSE, CONTRIBUTING.md, CHANGELOG.md
   - ✅ requirements.txt

3. **Check README renders correctly** (should display formatted Markdown)

## Step 5: Configure Repository Settings

### Set Repository Topics

Add topics for discoverability:
1. Go to repository page
2. Click ⚙️ next to "About"
3. Add topics:
   - `eeg`
   - `emotion-recognition`
   - `deep-learning`
   - `music-recommendation`
   - `brain-computer-interface`
   - `signal-processing`
   - `tensorflow`
   - `python`
   - `neuroscience`

### Set Homepage

Add project website (if any) or keep blank.

### Set License Display

GitHub should auto-detect `LICENSE` file and show "Proprietary" badge.

## Step 6: Set Up GitHub Features

### Enable Issues

1. Go to **Settings** → **General** → **Features**
2. Check ✅ **Issues**
3. Add issue templates (optional):
   ```bash
   mkdir -p .github/ISSUE_TEMPLATE
   # Create bug_report.md and feature_request.md
   ```

### Enable Projects (Optional)

For tracking development progress:
1. Go to **Projects** tab
2. Create project board with columns:
   - 📋 To Do
   - 🚧 In Progress
   - ✅ Done

### Create Development Branch

Protect `main` branch and use `dev` for active work:

```bash
git checkout -b dev
git push -u origin dev
```

Set `dev` as default branch (Settings → Branches → Default branch).

## Step 7: Add Collaborators (Optional)

If working with others:
1. Go to **Settings** → **Collaborators**
2. Click **Add people**
3. Enter GitHub usernames

## Step 8: Set Up GitHub Actions (Optional)

Create automated testing workflow:

```bash
mkdir -p .github/workflows
```

Create `.github/workflows/tests.yml`:

```yaml
name: Tests

on:
  push:
    branches: [ main, dev ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

Commit and push:

```bash
git add .github/workflows/tests.yml
git commit -m "Add GitHub Actions CI workflow"
git push
```

## Step 9: Add Badges to README (Optional)

Add status badges at the top of `README.md`:

```markdown
# Neuro-Adaptive Music Player v2

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)
[![Tests](https://github.com/YOUR_USERNAME/neuro-adaptive-music-player-v2/workflows/Tests/badge.svg)](https://github.com/YOUR_USERNAME/neuro-adaptive-music-player-v2/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Production-quality EEG-based emotion recognition system for adaptive music playback.
```

## Step 10: Create Release (Optional)

Tag the v2.0.0 release:

```bash
git tag -a v2.0.0 -m "Release v2.0.0: Complete core system rebuild

Core modules:
- EEG preprocessing with artifact detection
- Feature extraction (band power, FAA, statistics)
- CNN+BiLSTM emotion recognition model
- Comprehensive documentation and repository infrastructure

See CHANGELOG.md for full details."

git push origin v2.0.0
```

Create GitHub Release:
1. Go to **Releases** → **Create a new release**
2. Choose tag `v2.0.0`
3. Title: `v2.0.0 - Core System Rebuild`
4. Description: Copy from `CHANGELOG.md`
5. Click **Publish release**

## Common Issues

### Authentication Failed

**Problem**: `fatal: Authentication failed`

**Solution**: Use Personal Access Token (PAT):
1. Generate token: GitHub Settings → Developer settings → Personal access tokens → Tokens (classic) → Generate new token
2. Select scopes: `repo` (all)
3. Use token as password when pushing

Or set up SSH keys:
```bash
ssh-keygen -t ed25519 -C "your.email@example.com"
cat ~/.ssh/id_ed25519.pub  # Add to GitHub Settings → SSH keys
```

### Large Files Rejected

**Problem**: `remote: error: File X is 100MB; this exceeds GitHub's file size limit`

**Solution**: Use Git LFS for large files (models, datasets):
```bash
git lfs install
git lfs track "*.h5" "*.edf" "*.mat"
git add .gitattributes
git commit -m "Add Git LFS tracking"
git push
```

### Line Ending Issues (Windows)

**Problem**: Git converts line endings

**Solution**: Configure git:
```bash
git config --global core.autocrlf true
```

Or add `.gitattributes`:
```
* text=auto
*.py text eol=lf
*.sh text eol=lf
*.md text eol=lf
```

## Updating the Repository

After making changes:

```bash
# Check status
git status

# Stage changes
git add .

# Commit with descriptive message
git commit -m "Add data loading module for DEAP/SEED datasets"

# Push to GitHub
git push
```

## Branching Strategy

Recommended workflow:

- `main` - Stable releases only
- `dev` - Active development
- `feature/xxx` - New features
- `bugfix/xxx` - Bug fixes

```bash
# Create feature branch
git checkout -b feature/data-loaders dev

# Work on feature
# ... make changes ...

# Commit changes
git add .
git commit -m "Implement DEAP and SEED data loaders"

# Push to GitHub
git push -u origin feature/data-loaders

# Create Pull Request on GitHub
# After review, merge into dev
```

## Backup Strategy

GitHub is your primary backup, but also consider:

1. **Local backups**: Clone to multiple machines
2. **Archive releases**: Download release ZIPs
3. **External backup**: Google Drive, Dropbox, etc.

```bash
# Clone to another location
git clone https://github.com/YOUR_USERNAME/neuro-adaptive-music-player-v2.git backup/
```

## Migrating from Course Repository

Since this was originally in the course assessment folder, you might want to:

### Option 1: Keep Both Copies

Keep v2 in both locations:
- Original: For course submission
- GitHub: For development and portfolio

### Option 2: Remove from Course Repo

Add to course repo's `.gitignore`:
```
assessment1/Neuro-Adaptive Music Player v2/
```

Or delete and add a pointer:
```bash
# In course repo
rm -rf "assessment1/Neuro-Adaptive Music Player v2"

# Create README pointing to new repo
echo "# Neuro-Adaptive Music Player v2

This project has been moved to its own repository:
https://github.com/YOUR_USERNAME/neuro-adaptive-music-player-v2

The original v1.x code remains in assessment1.ipynb and assessment1 not concatenated.ipynb." > "assessment1/V2_MOVED_TO_GITHUB.md"
```

## Next Steps

After repository is set up:

1. **Share with others**: Send GitHub link to collaborators/advisors
2. **Continue development**: Implement remaining modules
3. **Add tests**: Write unit tests in `tests/`
4. **Create examples**: Add tutorial scripts in `examples/`
5. **Write documentation**: Expand README and ARCHITECTURE
6. **Publish release**: Tag v2.1.0 when next modules complete
7. **Promote project**: Share on LinkedIn, research groups, conferences

## Resources

- **Git Documentation**: [https://git-scm.com/doc](https://git-scm.com/doc)
- **GitHub Guides**: [https://guides.github.com/](https://guides.github.com/)
- **GitHub Actions**: [https://docs.github.com/en/actions](https://docs.github.com/en/actions)
- **Git LFS**: [https://git-lfs.github.com/](https://git-lfs.github.com/)

## Support

If you encounter issues:
1. Check GitHub's status page: [https://www.githubstatus.com/](https://www.githubstatus.com/)
2. Search GitHub Community: [https://github.community/](https://github.community/)
3. Ask in course/lab Slack/Discord

---

**Congratulations!** 🎉 Your project is now on GitHub and ready for collaborative development!
