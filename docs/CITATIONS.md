# Quick Citation Reference

**For Inline Code Comments and Academic Use**

This document provides quick citations for commonly used algorithms and techniques in the codebase.

---

## Signal Processing

**Butterworth Filter**:
> Butterworth, S. (1930). On the theory of filter amplifiers. *Wireless Engineer*, 7(6), 536-541.
- **Used in**: `src/eeg_preprocessing.py:144-184`

**Welch's Method**:
> Welch, P. (1967). The use of fast Fourier transform for the estimation of power spectra. *IEEE Trans. Audio Electroacoustics*, 15(2), 70-73.
- **DOI**: 10.1109/TAU.1967.1161901
- **Used in**: `src/eeg_features.py:156-218`

---

## EEG Preprocessing

**BCILAB Pipeline**:
> Kothe, C. A., & Makeig, S. (2013). BCILAB: A platform for brain-computer interface development. *J. Neural Eng.*, 10(5), 056014.
- **DOI**: 10.1088/1741-2560/10/5/056014
- **Used in**: `src/eeg_preprocessing.py` - filter parameters (0.5-45 Hz)

**Artifact Detection**:
> Mullen, T. R., et al. (2015). Real-time neuroimaging and cognitive monitoring. *IEEE Trans. Biomed. Eng.*, 62(11), 2553-2567.
- **DOI**: 10.1109/TBME.2015.2481482
- **Used in**: `src/eeg_preprocessing.py:474-542` - thresholds (100μV, 50μV/sample)

---

## Emotion Recognition

**Differential Entropy Features**:
> Zheng, W. L., & Lu, B. L. (2015). Investigating critical frequency bands. *IEEE TAMD*, 7(3), 162-175.
- **DOI**: 10.1109/TAMD.2015.2431497
- **Used in**: `src/eeg_features.py:432-517`

**DEAP Dataset**:
> Koelstra, S., et al. (2012). DEAP: A database for emotion analysis. *IEEE T-AFFC*, 3(1), 18-31.
- **DOI**: 10.1109/T-AFFC.2011.25
- **Used in**: `src/data_loaders.py:162-261`

---

## Frontal Alpha Asymmetry

**Original Theory**:
> Davidson, R. J. (1992). Emotion and affective style. *Psych. Science*, 3(1), 39-43.
- **DOI**: 10.1111/j.1467-9280.1992.tb00254.x
- **Used in**: `src/eeg_features.py:337-403`

**Methodology**:
> Allen, J. J., et al. (2004). Issues and assumptions on FAA. *Biol. Psych.*, 67(1-2), 183-218.
- **DOI**: 10.1016/j.biopsycho.2004.03.007
- **Used in**: Log-power computation method

**Channel Selection**:
> Frantzidis, C. A., et al. (2010). Emotion aware computing. *IEEE TITB*, 14(3), 589-597.
- **DOI**: 10.1109/TITB.2010.2041553
- **Used in**: Fp1-Fp2, F3-F4, F7-F8 pairs

---

## Deep Learning

**EEGNet Architecture**:
> Lawhern, V. J., et al. (2018). EEGNet: A compact CNN. *J. Neural Eng.*, 15(5), 056013.
- **DOI**: 10.1088/1741-2552/aace8c
- **Used in**: `src/emotion_recognition_model.py:153-289` - CNN design principles

**BiLSTM for EEG**:
> Li, Y., et al. (2018). Bi-hemispheric discrepancy model. *IEEE TCDS*, 13(2), 354-367.
- **DOI**: 10.1109/TCDS.2020.2999337
- **Used in**: `src/emotion_recognition_model.py:207-227`

**LSTM**:
> Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.
- **DOI**: 10.1162/neco.1997.9.8.1735
- **Used in**: TensorFlow/Keras LSTM implementation

**Batch Normalization**:
> Ioffe, S., & Szegedy, C. (2015). Batch normalization. *ICML*, 448-456.
- **Used in**: `src/emotion_recognition_model.py:190, 203`

**Dropout**:
> Srivastava, N., et al. (2014). Dropout. *JMLR*, 15(1), 1929-1958.
- **URL**: http://jmlr.org/papers/v15/srivastava14a.html
- **Used in**: All model architectures

**Adam Optimizer**:
> Kingma, D. P., & Ba, J. (2014). Adam. *arXiv*:1412.6980.
- **arXiv**: https://arxiv.org/abs/1412.6980
- **Used in**: `src/emotion_recognition_model.py:167-169`

---

## Music Psychology

**Circumplex Model**:
> Russell, J. A. (1980). A circumplex model of affect. *J. Pers. Soc. Psych.*, 39(6), 1161-1178.
- **DOI**: 10.1037/h0077714
- **Used in**: `src/music_recommendation.py:176-250` - emotion-music mapping

**Music and Emotion**:
> Juslin, P. N., & Sloboda, J. A. (2001). *Music and emotion*. Oxford University Press.
- **ISBN**: 978-0192631886
- **Used in**: Music feature selection

**Brain Correlates**:
> Koelsch, S. (2014). Brain correlates of music-evoked emotions. *Nature Rev. Neurosci.*, 15(3), 170-180.
- **DOI**: 10.1038/nrn3666
- **Used in**: Theoretical foundation

---

## BibTeX Format

```bibtex
@article{zheng2015investigating,
  title={Investigating critical frequency bands and channels for EEG-based emotion recognition with deep neural networks},
  author={Zheng, Wei-Long and Lu, Bao-Liang},
  journal={IEEE Transactions on Autonomous Mental Development},
  volume={7},
  number={3},
  pages={162--175},
  year={2015},
  doi={10.1109/TAMD.2015.2431497}
}

@article{davidson1992emotion,
  title={Emotion and affective style: Hemispheric substrates},
  author={Davidson, Richard J},
  journal={Psychological Science},
  volume={3},
  number={1},
  pages={39--43},
  year={1992},
  doi={10.1111/j.1467-9280.1992.tb00254.x}
}

@article{welch1967use,
  title={The use of fast Fourier transform for the estimation of power spectra},
  author={Welch, Peter},
  journal={IEEE Transactions on Audio and Electroacoustics},
  volume={15},
  number={2},
  pages={70--73},
  year={1967},
  doi={10.1109/TAU.1967.1161901}
}

@article{koelstra2012deap,
  title={DEAP: A database for emotion analysis using physiological signals},
  author={Koelstra, Sander and Muhl, Christian and Soleymani, Mohammad and Lee, Jong-Seok and Yazdani, Ashkan and Ebrahimi, Touradj and Pun, Thierry and Nijholt, Anton and Patras, Ioannis},
  journal={IEEE Transactions on Affective Computing},
  volume={3},
  number={1},
  pages={18--31},
  year={2012},
  doi={10.1109/T-AFFC.2011.25}
}

@article{lawhern2018eegnet,
  title={EEGNet: A compact convolutional neural network for EEG-based brain-computer interfaces},
  author={Lawhern, Vernon J and Solon, Amelia J and Waytowich, Nicholas R and Gordon, Stephen M and Hung, Chou P and Lance, Brent J},
  journal={Journal of Neural Engineering},
  volume={15},
  number={5},
  pages={056013},
  year={2018},
  doi={10.1088/1741-2552/aace8c}
}
```

---

**Full Bibliography**: See `RESEARCH_REFERENCES.md` for complete list with abstracts and URLs.
