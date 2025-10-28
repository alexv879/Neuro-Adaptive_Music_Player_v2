# GitHub Quick Start

## Fast Setup (5 Minutes)

### Step 1: Create Repository on GitHub

1. Go to https://github.com/new
2. Name: `Neuro-Adaptive-Music-Player-v2`
3. Description: `Real-time EEG emotion recognition with deep learning`
4. Choose Public/Private
5. Don't check initialization boxes
6. Click "Create repository"

### Step 2: Upload Code

```bash
# Navigate to project
cd "D:\AIUniversity\Applied Signals and Images Processing\assessment1\Neuro-Adaptive Music Player v2"

# Initialize git (if not already)
git init
git add .
git commit -m "Initial commit: Neuro-Adaptive Music Player v2"
git branch -M main

# Connect to GitHub (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/Neuro-Adaptive-Music-Player-v2.git

# Push to GitHub
git push -u origin main
```

### Step 3: Verify

Visit your repository: `https://github.com/YOUR_USERNAME/Neuro-Adaptive-Music-Player-v2`

## Next Steps

- Add `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET` to GitHub Secrets
- Enable GitHub Pages for documentation
- Set up CI/CD with GitHub Actions (optional)

---
*For detailed setup, see GITHUB_SETUP.md*
