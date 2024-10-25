(lab_multi_pitch)=
# Multi-Pitch-Estimation

## Goal of the task ?

Multi-Pitch-Estimation aims at extracting information related to the simultaneously occuring pitches over time within an audio file.
The task can either consists in:

- estimating at each time frame the existing fundamental frequencies (in Hz): $f_0(t)$
- estimating the start time and end time of each musical note (expressed as MIDI note): a list of [start_time, end_time, pitch]
- assigning an instrument-name (source) to each note: same as above with the instrument name assigned (see illustration below)

![flow_autotagging](/images/flow_multipitch.png)

The task has a long history.
First approches (signal-based) have focused on Single-Pitch-Estimation.
But as far as 2003, Klapuri et al {cite}`Klapuri2003IEEEMultipleF0` already proposed a signal-based method to iteratively estimate the Multiple-Pitches.
MPE then became a major research field, with method based on NMF or PLCA, SI-PLCA.

For this task, Deep Learning Approaches have become the standard, either based on
- Supervised Learning (for example {cite}`DBLP:conf/ismir/BittnerMSLB17`)
- Unsupervised learning (for example {cite}`DBLP:conf/ismir/RiouLHP23`)

We review here one of the most famous approaches proposed by Bittner et al {cite}`DBLP:conf/ismir/BittnerMSLB17` and show how we can extend it with the same front-end (Harmonic-CQT) using a U-Net {cite}`Doras2009UNetMelody,Weiss2022TASLPMPE`.

Fore more details, see the very good [tutorial on "Programming MIR Baselines from Scratch: Three Case Studies"](https://github.com/rabitt/ismir-2021-tutorial-case-studies)

## How is the task evaluated ?

To evaluate the performances of an MPE algorithm we rely on the metrics defined in {cite}`DBLP:conf/ismir/BayED09` and implemented in the [mir\_eval](https://craffel.github.io/mir_eval/#module-mir_eval.multipitch) package.
By default, an estimated frequency is considered "correct" if it is within 0.5 semitones of a reference frequency.

Using this, we compute at each time frame t:
- "True Positives" TP(t):  the number of F0s detected that correctly correspond to the ground-truth F0s
- "False Positives" FP(t): the number of F0s detected that do not exist in the ground-truth set
- "False Negatives" FN(t): represent the number of active sources in the groundtruth that are not reported

From this, one can compute
- Precision= $\frac{TP}{TP+FN}$
- Recall= $\frac{TP}{TP+FP}$
- Accuracy= $\frac{TP}{TP+FP+FN}$

We can also compute the same metrics but considering only the chroma estimation (independently of the octave estimated).
This leads to the Chroma Precision, Accuracy, Recall

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

Many datasets exist for mutli-pitch-estimation.
Those can be obtained by
- manually annotated full-tracks,
- annotating (or using mono-pitch estimation algorithm) the individual stems of a full-track (MedleyDB)
- using a MIDI-fied piano: SMD, MAPS, MAESTRO
- using audio to score synchronization: MusicNet, Winterreise

We have chosen the two following datasets since they represent two different types of annotations (continuous f0 annotations or segment-based midi-pitch annotations).

- Bach10 {cite}`DBLP:journals/taslp/DuanPZ10`.
It is a multi-track datasets in which each track is annotated in pitch (time, continuous f0-value) over for each time-frame.
- MAPS {cite}`DBLP:journals/taslp/EmiyaBD10`.
It is a piano dataset annotated as a sequence of notes (start,stop,midi-value) over time



## How we can solve it using deep learning

We will implement two different models which both takes as input the [Harmonic-CQT](lab_hcqt) features.

The first is the traditional ConvNet proposed by {cite}`DBLP:conf/ismir/BittnerMSLB17`
![model_MPE_deepsalience](/images/model_MPE_deepsalience.png)

The second is the U-Net proposed by U-Net {cite}`Doras2009UNetMelody,Weiss2022TASLPMPE`
![model_MPE_unet](/images/model_MPE_unet.png)

We illustrate a deep learning solution to this problem in the following [notebook](https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/blob/master/TUTO_task_Multi_Pitch_Estimation.ipynb) and study various configurations [ConvNet](https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/blob/master/config_bittner.yaml) or [U-Net](https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/blob/master/config_doras.yaml).

### Experiments

We will vary in turn
- the **inputs**: [CQT](lab_cqt) or [Harmonic-CQT](lab_hcqt)
- the model **blocks**: [Conv-2D](lab_conv2d), [Depthwise Separable Convolution](lab_depthwise), [ResNet](lab_resnet), [ConvNext](lab_convnext), [U-Net](lab_unet)
- the **datasets**: a small one (Bach10 with continous f0 annotation) a large one (MAPS with segments annotated in MIDI-pitch)

![expe](/images/expe_multipitch_P.png)

This can be done using the following files:
- (Main notebook)(https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/blob/master/TUTO_task_Multi_Pitch_Estimation.ipynb)
- (Config Conv2D)[https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/blob/master/config_bittner.yaml]
- (Config U-Net)[https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/blob/master/config_doras.yaml]

| Dataset   | Input   | Frontend   | Results   | Code |
|:---------- |:----------|:----------|:---------- |:---------- |
| Bach10     | CQT(H=1)       |  Conv2D            | P=0.84, R=0.71, Acc=0.63  | [LINK](https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/blob/master/TUTO_task_Multi_Pitch_Estimation.ipynb_D1-I1-C1.ipynb) |
| Bach10     | HCQT(H=6)      |  Conv2D            | P=0.92, R=0.79, Acc=0.74  | [LINK](https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/blob/master/TUTO_task_Multi_Pitch_Estimation.ipynb_D1-I2-C1.ipynb) |
| Bach10     | HCQT(H=6)      |  Conv2D/DepthSep   | P=0.92, R=0.78, Acc=0.74  | [LINK](https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/blob/master/TUTO_task_Multi_Pitch_Estimation.ipynb_D1-I2-C2.ipynb) |
| Bach10     | HCQT(H=6)      |  Conv2D/ResNet     | P=0.93, R=0.80, Acc=0.75  | [LINK](https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/blob/master/TUTO_task_Multi_Pitch_Estimation.ipynb_D1-I2-C3.ipynb) |
| Bach10     | HCQT(H=6)      |  Conv2D/ConvNext   | P=0.92, R=0.80, Acc=0.75  | [LINK](https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/blob/master/TUTO_task_Multi_Pitch_Estimation.ipynb_D1-I2-C4.ipynb) |
| Bach10     | HCQT(H=6)      |  U-Net             | P=0.91, R=0.78, Acc=0.73  | [LINK](https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/blob/master/TUTO_task_Multi_Pitch_Estimation.ipynb_D1-I2-Unet.ipynb) |
| --  | -- | -- | -- | -- |
| MAPS       | HCQT(H=6)      |  Conv2D            | P=0.86, R=0.75, Acc=0.67  | [LINK](https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/blob/master/TUTO_task_Multi_Pitch_Estimation.ipynb_D2-I2-C1.ipynb) |
| MAPS       | HCQT(H=6)      |  Conv2D/ResNet     | P=0.83, R=0.83, Acc=0.71  | [LINK](https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/blob/master/TUTO_task_Multi_Pitch_Estimation.ipynb_D2-I2-C3.ipynb) |
| MAPS       | HCQT(H=6)      |  U-Net             | P=0.84, R=0.81, Acc=0.70  | [LINK](https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/blob/master/TUTO_task_Multi_Pitch_Estimation.ipynb_D2-I2-Unet.ipynb) |
