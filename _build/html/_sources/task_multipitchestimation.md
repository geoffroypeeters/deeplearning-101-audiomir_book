(lab_multi_pitch)=
# Multi-Pitch-Estimation (MPE)

## Goal of MPE ?

Multi-Pitch-Estimation aims at extracting information related to the simultaneously occuring pitches over time within an audio file.
The task can either consists in:

1. estimating at each time frame the existing <mark>continuous fundamental frequencies</mark> (in Hz): $f_0(t)$
2. estimating the <mark>[start_time, end_time, pitch]</mark> of each musical note (expressed as MIDI note)
3. assigning an <mark>instrument-name</mark> (source) to the above(see illustration)

![flow_autotagging](/images/flow_multipitch.png)

### A very short history of MPE
The task has a long history.
- Early approaches focused on single pitch estimation using a <mark>signal-based method</mark>, such as the YIN {cite}`CheveigneJASA2002Pitch` algorithm.
- Next, the difficult case of multiple pitch estimation (MPE) (overlapping harmonics, ambiguous number of simultaneous pitches) was addressed using <mark>iterative</mark> estimation, as in Klapuri et al {cite}`Klapuri2003IEEEMultipleF0`.
- Subsequently, <mark>unsupervised methods</mark> aimed at reconstructing the signal using a mixture of models (with non-negative matrix factorisation NMF, probabilistic latent component analysis PLCA or invariant latent component analysis SI-PLCA) have been the main trend {cite}`DBLP:journals/taslp/FuentesBR13`.

**Deep learning era.**
- We review here one of the most famous approaches proposed by Bittner et al {cite}`DBLP:conf/ismir/BittnerMSLB17` and
- show how we can extend it with the same front-end (Harmonic-CQT) using a U-Net {cite}`Doras2009UNetMelody,Weiss2022TASLPMPE`.

The task is still very active today, especially using unsupervised learning approaches, more specifically the <mark>"equivariance"</mark> property, such as in SPICE {cite}`DBLP:journals/taslp/GfellerFRSTV20` or PESTO {cite}`DBLP:conf/ismir/RiouLHP23`


Fore more details, seethe very good tutorials
- ["Fundamental Frequency Estimation in Music"](https://ismir2018.ismir.net/pages/events-tutorial-06.html)
- ["Programming MIR Baselines from Scratch: Three Case Studies"](https://github.com/rabitt/ismir-2021-tutorial-case-studies)

## How is MPE evaluated ?

To evaluate the performances of an MPE algorithm we rely on the metrics defined in {cite}`DBLP:conf/ismir/BayED09` and implemented in the [mir\_eval](https://craffel.github.io/mir_eval/#module-mir_eval.multipitch) package.
By default, an estimated frequency is considered <mark>"correct"</mark> if it is within 0.5 semitones of a reference frequency.

Using this, we compute at each time frame t:
- <mark>"True Positives"</mark> TP(t):  the number of F0s detected that correctly correspond to the ground-truth F0s
- <mark>"False Positives"</mark> FP(t): the number of F0s detected that do not exist in the ground-truth set
- <mark>"False Negatives"</mark> FN(t): represent the number of active sources in the groundtruth that are not reported

From this, one can compute
- Precision= $\frac{TP}{TP+FN}$
- Recall= $\frac{TP}{TP+FP}$
- Accuracy= $\frac{TP}{TP+FP+FN}$

We can also compute the same metrics but considering only the <mark>chroma</mark> associated to the estimated pitch (independently of the octave estimated).
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


## Some popular datasets for MPE

A (close to) exhaustive list of MIR datasets is available in the [ismir.net web site](https://ismir.net/resources/datasets/).

MPE datasets can be obtained in several ways:
1. <mark>manually</mark> annotated full-tracks,
2. annotating (or using mono-pitch estimation algorithm) the individual <mark>stems</mark> of a full-track (MedleyDB)
3. using a <mark>MIDI-fied</mark> piano: SMD, MAPS, MAESTRO
4. using audio to score <mark>synchronization</mark>: MusicNet, Winterreise

We have chosen the two following datasets since they represent two different types of annotations:

### Bach10
Bach10 {cite}`DBLP:journals/taslp/DuanPZ10` is a small (ten tracks) but multi-track datasets in which each track is annotated in pitch (time, continuous f0-value) over for each time-frame.

```python
"entry": [
            {
                "filepath": [
                    {"value": "01-AchGottundHerr-violin.wav"}
                ],
                "f0": [
                    {"value": [
                            [
                                72.00969707905834,
                                72.00969707905834,
                                72.00763743216136,
                                72.00763743216136,
                                72.00763743216136,
                                72.03300373636725,
                                72.04641885597061,
                                ...
                              ]
                            ]}
                    {"time": [
                     0.023,
                     0.033,
                     0.043,
                     0.053,
                     0.063,
                     0.073,
                     0.083,
                     ...
                     ]}

```

### ENST MAPS
ENST MAPS (MIDI Aligned Piano Sounds) {cite}`DBLP:journals/taslp/EmiyaBD10` is a large (31 Go) piano dataset.
Four categories of sounds are provided: isolated notes, random chords, usual chords, pieces of music.
We only use the later for our experiment.
It is annotated as a sequence of notes (start,stop,midi-value) over time.


```python
"entry": [
            {
                "filepath": [
                    {"value": "MAPS_MUS-alb_se3_AkPnBcht.wav"}
                ],
                "pitchmidi": [
                    {"value": 67, "time": 0.500004, "duration": 0.26785899999999996},
                    {"value": 71, "time": 0.500004, "duration": 0.26785899999999996},
                    {"value": 43, "time": 0.500004, "duration": 1.0524360000000001},
                    ...
                ]
              }

```

## How can we solve MPE using deep learning ?

We will implement two different models which both takes as input the [Harmonic-CQT](lab_hcqt) features.

The first is the traditional ConvNet proposed by {cite}`DBLP:conf/ismir/BittnerMSLB17`
![model_MPE_deepsalience](/images/model_MPE_deepsalience.png)

The second is the U-Net proposed by U-Net {cite}`Doras2009UNetMelody,Weiss2022TASLPMPE`
![model_MPE_unet](/images/model_MPE_unet.png)


### Experiments

We illustrate a deep learning solution to this problem using the following files:
- (Main notebook)(https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/blob/master/TUTO_task_Multi_Pitch_Estimation.ipynb)
- (Config Conv2D)[https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/blob/master/config_bittner.yaml]
- (Config U-Net)[https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/blob/master/config_doras.yaml]

We will vary in turn
- the **inputs**: [CQT](lab_cqt) or [Harmonic-CQT](lab_hcqt)
- the model **blocks**: [Conv-2D](lab_conv2d), [Depthwise Separable Convolution](lab_depthwise), [ResNet](lab_resnet), [ConvNext](lab_convnext), [U-Net](lab_unet)
- the **datasets**: a small one (Bach10 with continous f0 annotation) a large one (MAPS with segments annotated in MIDI-pitch)

![expe](/images/expe_multipitch_P.png)


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
