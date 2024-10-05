## Inputs

The following type of data are commonly used in MIR as input of deep learning models.

(lab_lms)
### Waveform

It is possible to use directly the audio waveform $x(n)$ as input to a model. In this case, the input is a 1-dimensional sequence over time.
Such a system is often denoted by end-to-end (E2E).
The first layer of the models then act as a learnable feature extractor.
It is often either a 1D-convolution [], a [TCN](lab_tcn) or a parametric front-end such as [SincNet](label_sincnet) or [LEAF](label_leaf).

More details can be found in the following tutorial.
Example of systems than use waveform as input are [Dieleman], [Pons], WavUNet, TasNet, ConvTasNet.

(lab_lms)=
### Log-Mel-Spectrogram (LMS)

Spectrogram (the magnitude of the Short Time Fourier Transform, i.e. the Fourier Transform performed over frame-analysis) can be converted to the Mel [REF] perceptual scale. The goal of this is
- to reduce the dimensionality of the data
- to mimic the decomposition of the frequencies performed by the cochlea into critical-bands
- to allows performing some invariance over small pitch modifications (hence LMS are invariant to the pitch and only represent the so-called timbre).

The conversion of amplitude  from linear to the log-scale allows
- to map the recording level of the audio to a constant: $\alpha x(n) \rightarrow \log(\alpha) + \log(X(\omega))$
- to mimic the compression of the amplitude performed by the inner-cell of the cochlea
- to change the distribution of the input
Usually, a $\log(1+C x)$ (with $C=10.000$) is used instead of a $\log(x)$ to avoid singularity in $x=0$.

Another view of the LMS, is to consider that those are equivalent to the MFCC but without the last DCT.
This DCT was necessary to decorrelate the dimensions and then allows covariance matrix in GMM-based system.
However, this decorrelation is not necessary for deep learning models.

```python
def f_get_lms(audio_v, sr_hz):
  C = 10000
  data_m = librosa.feature.melspectrogram(y=audio_v, sr=sr_hz, n_melsint=128)
  lms_m = np.log(1 + C*data_m)
  return lms_m
```


(lab_cqt)=
### Constant-Q-Transform (HCQT)

CQT was proposed in {cite}`Brown1991ConstantQ`


(lab_hcqt)=
### Harmonic-CQT (HCQT)

HCQT was proposed in {cite}`DBLP:conf/ismir/BittnerMSLB17`

```python
def f_get_hcqt(audio_v, sr_hz):
    """
    """
    h_l = [0.5, 1, 2, 3, 4, 5]
    BINS_PER_SEMITONE = 3
    BINS_PER_OCTAVE = 12 * BINS_PER_SEMITONE
    N_OCTAVES = 6
    N_BINS = N_OCTAVES * BINS_PER_OCTAVE
    FMIN = 32.7
    HOP_LENGTH = 512

    for idx, h in enumerate(h_l):
        A_m = np.abs(librosa.cqt(y=audio_v, sr=sr_hz, fmin=h*32.7, hop_length=HOP_LENGTH, bins_per_octave=BINS_PER_OCTAVE, n_bins=N_BINS))
        if idx==0:
            CQT_3m = np.zeros((len(h_l), A_m.shape[0], A_m.shape[1]))
        CQT_3m[idx,:,:] = A_m

    n_times = CQT_3m.shape[2]
    cqt_time_sec_v = librosa.frames_to_time(np.arange(n_times), sr=sr_hz, hop_length=HOP_LENGTH)
    cqt_frequency_hz_v = librosa.cqt_frequencies(n_bins=N_BINS, fmin=FMIN, bins_per_octave=BINS_PER_OCTAVE)

    return CQT_3m, cqt_time_sec_v, cqt_frequency_hz_v
```

### Audio augmentations

blablabla
