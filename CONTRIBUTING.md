# Contributing to Neuro-Adaptive Music Player v2

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## 🎓 Academic Collaboration

This project is primarily for academic/educational purposes (CMP9780M Assessment). We welcome contributions from:
- Students learning EEG signal processing
- Researchers working on emotion recognition
- Developers interested in brain-computer interfaces

## 📋 Before You Start

1. **Read the Documentation**
   - [README.md](README.md) - Project overview
   - [ARCHITECTURE.md](ARCHITECTURE.md) - System design
   - [LICENSE](LICENSE) - Usage terms

2. **Check Existing Issues**
   - Look for open issues that match your interest
   - Comment on issues you'd like to work on
   - Avoid duplicate work

3. **Understand the Scope**
   - This is a proprietary project with educational use license
   - Commercial use requires explicit permission
   - All contributions must respect privacy and ethical guidelines

## 🚀 How to Contribute

### 1. Fork and Clone

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/neuro-adaptive-music-player-v2.git
cd neuro-adaptive-music-player-v2

# Add upstream remote
git remote add upstream https://github.com/alexv879/neuro-adaptive-music-player-v2.git
```

### 2. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation improvements
- `test/` - Test additions/improvements
- `refactor/` - Code refactoring

### 3. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8 mypy
```

### 4. Make Your Changes

Follow these guidelines:

#### Code Style
- **PEP 8** compliance (use `black` for formatting)
- **Type hints** for all functions
- **Docstrings** for all public methods (Google style)
- **Comments** for complex logic

```python
def extract_band_power(
    self,
    data: np.ndarray,
    band: Tuple[float, float],
    method: str = 'welch'
) -> np.ndarray:
    """
    Extract power in a specific frequency band.
    
    Args:
        data: EEG data of shape (n_channels, n_samples)
        band: Frequency band as (low_freq, high_freq) tuple
        method: Extraction method ('welch' or 'fft')
        
    Returns:
        Band power values for each channel
        
    Example:
        >>> extractor = EEGFeatureExtractor()
        >>> alpha_power = extractor.extract_band_power(data, (8, 13))
    """
    # Implementation
```

#### Testing
- Add tests for new features
- Ensure existing tests pass
- Aim for 80%+ code coverage

```python
# tests/test_features.py
def test_extract_band_power():
    """Test band power extraction with known signal."""
    extractor = EEGFeatureExtractor(fs=256)
    
    # Create 10 Hz sine wave
    t = np.arange(1280) / 256
    signal = np.sin(2 * np.pi * 10 * t)
    
    # Extract alpha power (8-13 Hz, should be high)
    alpha_power = extractor.extract_band_power_welch(
        signal.reshape(1, -1),
        band=(8, 13)
    )
    
    assert alpha_power > 0
    assert not np.isnan(alpha_power)
```

#### Documentation
- Update relevant docs for API changes
- Add examples for new features
- Keep ARCHITECTURE.md current

### 5. Run Tests and Checks

```bash
# Format code
black src/ tests/

# Check style
flake8 src/ tests/

# Type checking
mypy src/

# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### 6. Commit Your Changes

Follow conventional commit format:

```bash
git add .
git commit -m "feat: add differential entropy feature extraction"
# or
git commit -m "fix: resolve NaN in bandpass filter for short signals"
# or
git commit -m "docs: update README with SEED dataset example"
```

Commit message prefixes:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation only
- `test:` - Test additions/improvements
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `style:` - Code style changes (formatting, etc.)
- `chore:` - Maintenance tasks

### 7. Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create pull request on GitHub
# Provide clear description of changes
```

## 📝 Pull Request Guidelines

### PR Description Template

```markdown
## Description
Brief description of changes

## Motivation
Why is this change needed?

## Changes Made
- Change 1
- Change 2

## Testing
- [ ] Added tests for new functionality
- [ ] All existing tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No breaking changes (or documented)
- [ ] Commit messages follow convention

## Screenshots (if applicable)
```

### Review Process

1. **Automated Checks**
   - Tests must pass
   - Code coverage maintained
   - Style checks pass

2. **Code Review**
   - At least one maintainer approval required
   - Address all review comments
   - Keep PR scope focused

3. **Merge**
   - Squash and merge for clean history
   - Delete branch after merge

## 🐛 Reporting Bugs

Use the GitHub issue tracker:

**Bug Report Template:**
```markdown
**Description**
Clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Load data with...
2. Call function...
3. See error

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: [e.g., Windows 10]
- Python version: [e.g., 3.10.0]
- TensorFlow version: [e.g., 2.10.0]
- EEG device: [e.g., Muse 2]

**Code Sample**
```python
# Minimal code to reproduce
```

**Error Message**
```
Full error traceback
```

**Additional Context**
Any other relevant information
```

## 💡 Feature Requests

**Feature Request Template:**
```markdown
**Problem Statement**
What problem does this solve?

**Proposed Solution**
How should it work?

**Alternatives Considered**
Other approaches you've thought about

**Use Case**
Example of how you'd use this

**Additional Context**
Any other relevant information
```

## 🎯 Priority Areas for Contribution

### High Priority
1. **Dataset Loaders**
   - DEAP dataset loader
   - SEED dataset loader
   - EDF file loader
   - Custom CSV loader

2. **Live Streaming**
   - Serial/Bluetooth handler
   - Buffer management
   - Packet loss handling

3. **Testing**
   - Unit tests for all modules
   - Integration tests
   - Performance benchmarks

### Medium Priority
4. **Transfer Learning**
   - Fine-tuning implementation
   - Domain adaptation
   - Few-shot learning

5. **Music Integration**
   - Spotify API integration
   - Genre classification
   - Playlist generation

6. **Documentation**
   - Tutorial notebooks
   - Video tutorials
   - API reference (Sphinx)

### Nice to Have
7. **Advanced Features**
   - ICA artifact correction
   - IMU-based correction
   - Multi-modal fusion
   - Online learning

8. **Visualization**
   - Real-time EEG plots
   - Feature visualization
   - Model interpretation

## 🔬 Research Contributions

If contributing research-based improvements:

1. **Cite Sources**
   - Reference papers clearly
   - Link to implementations
   - Credit original authors

2. **Validate Results**
   - Test on standard datasets
   - Compare with baselines
   - Document performance

3. **Reproducibility**
   - Provide exact parameters
   - Include random seeds
   - Document hardware used

## 📖 Documentation Style

### Code Documentation
- Use Google-style docstrings
- Include type hints
- Provide examples
- Explain complex algorithms

### Markdown Documentation
- Use clear headings
- Include code examples
- Add diagrams where helpful
- Keep language concise

## ⚖️ Ethical Guidelines

All contributions must:
- Respect user privacy
- Avoid bias and discrimination
- Follow ethical AI principles
- Comply with applicable laws
- Consider potential misuse

## 🤝 Code of Conduct

### Our Pledge
- Be respectful and inclusive
- Welcome diverse perspectives
- Give constructive feedback
- Focus on what's best for the project

### Unacceptable Behavior
- Harassment or discrimination
- Personal attacks
- Trolling or insulting comments
- Inappropriate content

### Enforcement
Violations may result in:
- Warning
- Temporary ban
- Permanent ban

Report issues to: [your contact email]

## 📬 Contact

- **GitHub Issues**: For bugs and features
- **Email**: [your email] for private inquiries
- **Discussions**: Use GitHub Discussions for questions

## 🙏 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in academic publications (if applicable)

---

Thank you for contributing to advancing EEG-based emotion recognition! 🧠🎵

**Remember**: Quality over quantity. One well-tested, documented feature is better than ten half-finished ones.
