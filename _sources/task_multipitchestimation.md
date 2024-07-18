# Multi-Pitch-Estimation

- author: Geoffroy
- code: based on Bittner, Doras
- datasets: bach-10, MAPS

## Goal of the task ?

Multi-Pitch-Estimation aims at extracting information related to the simultaneously occuring pitches over time within an audio file.
The task can either consists in:

- estimating at each time frame the existing fundamental frequencies (in Hz)
- estimating the start time and end time of each musical note (expressed as MIDI note)
- assigning an instrument-name (source) to each note

The task has a long history.
First approches (signal-based) have focused on Single-Pitch-Estimation.
But as far as 2003, Klapuri et el {cite}`Klapuri2003IEEEMultipleF0` have proposed a signal-based method to iteratively estimate the Multiple-Pitch.
MPE then became a major research field, with method based on NMF or ??? (unsupervised method).

For this task, Deep Learning Approaches have become the standard, either based on
- Supervised Learning
- Unsupervised learning (PESTO)

We review here one of the most famous approaches proposed by Bittner et al {cite}`DBLP:conf/ismir/BittnerMSLB17` and show how we can extend it with the same front-end using a U-Net {cite}`Doras2009UNetMelody,Weiss2022TASLPMPE`.

## How is the task evaluated ?

To evaluate the performances of an MPE algorithm we rely on the metrics defined in {cite}`DBLP:conf/ismir/BayED09` and implemented in the [mir\_eval](https://craffel.github.io/mir_eval/) package.
By default, a frequency is "correct" if it is within 0.5 semitones of a reference frequency

|                | Predicted Positive | Predicted Negative | Total  |
|----------------|--------------------|--------------------|--------|
| **Actual Positive** | TP                 | FN                 | n_ref |
| **Actual Negative** | FP                 | TN                 |  |
| **Total**           | n_est            |             |  |

- Accuracy= $\frac{TP}{TP+FP+FN}$
- Precision= $\frac{TP}{TP+FN}$
- Recall= $\frac{TP}{TP+FP}$

Same but considering octave error as correct
- Chroma Accuracy, Precision, Recall

Example:
```python
freq = lambda midi : 440*2**((midi-69)/12)

ref_time = np.array([0.1, 0.2, 0.3])
ref_freqs = [np.array([freq(70), freq(72)]), np.array([freq(70), freq(72)]), np.array([freq(70), freq(72)])]

est_time = np.array([0.1, 0.2, 0.3])
est_freqs = [np.array([freq(70.4+12)]), np.array([freq(70), freq(72), freq(74)]), np.array([freq(70), freq(72)])]

mir_eval.multipitch.evaluate(ref_time, ref_freqs, est_time, est_freqs)

OrderedDict([('Precision', 0.6666666666666666),
             ('Recall', 0.6666666666666666),
             ('Accuracy', 0.5),
             ('Substitution Error', 0.16666666666666666),
             ('Miss Error', 0.16666666666666666),
             ('False Alarm Error', 0.16666666666666666),
             ('Total Error', 0.5),
             ('Chroma Precision', 0.8333333333333334),
             ('Chroma Recall', 0.8333333333333334),
             ('Chroma Accuracy', 0.7142857142857143),
             ('Chroma Substitution Error', 0.0),
             ('Chroma Miss Error', 0.16666666666666666),
             ('Chroma False Alarm Error', 0.16666666666666666),
             ('Chroma Total Error', 0.3333333333333333)])
```


## Some popular datasets

A (close to) exhaustive list of MIR datasets is available in the [ismir.net web site](https://ismir.net/resources/datasets/).

Popular datasets used for MPE are
| Dataset | Column 2 | Column 3 | Column 4 |
|---------|----------|----------|----------|
| Bach10  |          |          |          |
| Su      |          |          |          |
| MedleyDB|          |          |          |
| MIDI-fied piano: SMD, MAPS, MAESTRO | | | |
| Score-audio pairs: MusicNet, Winterreise Dataset | | | |


## How we can solve it using deep learning

We will implement two different models which both takes as input the [HCQT](lab_hcqt) features.
- the first is the traditional ConvNet proposed by {cite}`DBLP:conf/ismir/BittnerMSLB17`
- the second is the U-Net proposed by U-Net {cite}`Doras2009UNetMelody,Weiss2022TASLPMPE`

We will train and evaluate them on two different datasets
- Bach10 {cite}`DBLP:journals/taslp/DuanPZ10`: which is a multi-track datasets in which each track is annotated in pitch (continuous f0-value) over time
- MAPS {cite}`DBLP:journals/taslp/EmiyaBD10`: which is a piano dataset annotated as a set of notes (start,stop,value) over time
