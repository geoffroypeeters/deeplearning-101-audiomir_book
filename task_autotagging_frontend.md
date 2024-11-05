(lab_auto_tagging)=
# Auto-Tagging (front-ends)


## Goal of Auto-Tagging ?

Music auto-tagging is the task of assigning tags (such as genre, style, moods, instrumentation, chords) to a music track.

Tags can be

||Mutually exclusive (multi-class) |Non-mutually exclusive (multi-label)|
|----|----|----|
| **Global in time** | Music-genre | User-tags |
| **Time-based** | Chord-segments | Instrument-segments |



![flow_autotagging](/images/flow_autotagging.png)




### A very short history of Auto-Tagging
The task has a long history in MIR.
- As soon as <mark>2002 Tzanetakis</mark> et al. {cite}`DBLP:journals/taslp/TzanetakisC02` demonstrated that it is possible to estimate the `genre` using a set of low-level (hand-crafted) audio features (such as MFCC) and simple machine-learning models (such as Gaussian-Mixture-Models).
- Over years, the considered <mark>audio features</mark> improved  {cite}`Peeters2004AudioFeatures`, including block-features {cite}`Seyerlehner2010PHD` or speech-inspired features (Universal-Background-Models and Super-Vector {cite}`Charbuillet2011DAFX`), as well as the <mark>machine-learning</mark> models (moving to Random forest or Support-Vector-Machine).
- It also quickly appeared that the <mark>same feature/ML system could be trained to solve many tasks</mark> of tagging or segmentation (genre, mood, speech/music) {cite}`Peeters2007DAFXGenericClassification`, {cite}`Burred2009LSASMultiLabel`.

**Deep learning era.**
- We start the story with <mark>Dieleman</mark> {cite}`Dieleman2014Spotify` who proposes to use a <mark>Conv2d</mark> applied to a <mark>Log-Mel-Spectrogram</mark> with kernel extending over the whole frequency range, therefore performing only convolution over time.\
*The rational for this, is that, as opposed to natural images, sources in a T/F representation are not invariant by translation over frequencies and the adjacent frequencies are not necesseraly correlated (spacing between harmonics).*
- Despite this, Choi et al. {cite}`DBLP:conf/ismir/ChoiFS16` proposed (with success) to apply Computer Vision <mark>VGG-like architecture</mark> to a time-frequency representation.
- Later on, Pons et al. {cite}`DBLP:conf/cbmi/PonsLS16` proposed to <mark>design kernel shapes using musical consideration</mark> (with kernel extending over frequencies to represent timbre, over time to represent rhythm).
- In order to avoid having to choose the kernel shape and STFT parameters, it is been proposed to use directly the <mark>audio waveform</mark> as input, the "End-to-End" systems of Dieleman et al. {cite}`DBLP:conf/icassp/DielemanS14` or Lee et al. {cite}`DBLP:journals/corr/LeePKN17`.
- The task of auto-tagging has also close relationship with their equivalent task in Speech.

<mark>*We will develop here a model developed initially for speaker recognition by Ravanelli et al. {cite}`DBLP:conf/slt/RavanelliB18`.*</mark>

The task is still very active <mark>today</mark>.
For example
- in the <mark>supervised case</mark>,
	- MULE {cite}`DBLP:conf/ismir/McCallumKOGE22` which uses a more sophisticated ConvNet architecture (Short-Fast-Normalizer-Free Net F0) and training paradigm (contrastive learning)
	- PaSST {cite}`DBLP:conf/hear/KoutiniMSESW21` which uses Vit with tokenized (set of patches) spectrograms fed to a Transformer
- in the <mark>Self-Supervised-Learning case</mark>
	- with the so-called foundation models such as MERT {cite}`DBLP:conf/iclr/LiYZMCYXLRBGDLC24` (see second part of this tutorial).

Fore more details, see the very good tutorial
["musical classification"](https://music-classification.github.io/tutorial/landing-page.html).



### A very short history of Chord Estimation.

Chord estimation can be considered as a specific tagging application: it involves applying mutually exclusive labels (of chords) over segments of time.\
However, it has (at least) two specificities:
- chord transition follow <mark>musical rules</mark> which can be represented by a <mark>language model</mark>.
- some chord are equivalent, their spelling depends on the choice of the level of detail, and their choice on the

Therefore, <mark>ASR (Automatic Speech Recognition)</mark> inspired techniques has been developed at first {cite}`Sheh2003ISMIRchord` or {cite}`Papadopo2007CBMI` with
- an <mark>acoustic model</mark> representing $p(\text{chord}|\text{chroma})$ and
- a <mark>language model</mark>, often a Hidden Markov Model, representing $p(\text{chord}_{t}|\text{chord}_{t-1}).$

**Deep learning era.**
In the case of chord estimation, deep learning is also now commonly used.
One seminal paper for this is McFee at al. {cite}`DBLP:conf/ismir/McFeeB17`
- the model is a RCNN (a ConvNet followed by a <mark>bi-directional RNN</mark>, here GRU)
- the model is trained to use an <mark>inner representation</mark> which relates to the `root`, `bass` and `pitches` (the <mark>CREMA</mark>)
	- this allows learning representation which brings together close (but different) chords

<mark>*We will develop here a similar model based on the combination of Conv2d and Bi-LSTM but without the multi-task approach.*</mark>





## How is the task evaluated ?

We consider a set of classes $c \in \{1,\ldots,C\}$.

### Multi-class

In a **multi-class** problem, the classes are **mutually exclusive**.
- The outputs of the (neural network) model $o_c$ therefore go to a softmax function.
- The outputs of the softmax, $p_c$, then represent the probability $P(Y=c|X)$.
- The predicted class is then chosen as $\arg\max_c p_c$.

We evaluate the performances by computing the standard  
- <mark>Accuracy, Recall, Precision, F-measure</mark> for each class $c$ and then take the average over classes $c$.

```python
from sklearn.metrics import classification_report, confusion_matrix
classification_reports = classification_report(labels_idx,
																								labels_pred_idx,
																								output_dict=True)
cm = confusion_matrix(labels_idx, labels_pred_idx)
```

### Multi-label

In the **multi-label** problem, the classes are **NOT mutually exclusive**.
- Each $o_c$ therefore goes individually to a sigmoid function (multi-label is processed as a set of parallel independent binary classification problems).
- The outputs of the sigmoids $p_c$ then represent $P(Y_c=1|X)$.
- We then need to set a threshold $\tau$ on each $p_c$ to decide wether class $c$ exist or not.

Using a default threshold ($\tau=0.5$) of course allows to use the afore-mentioned metrics (Accuracy, Recall, Precision, F-measure).\
However, in practice, <mark>we want to measure the performances independently of the choice of a given threshold</mark>.

This can be using either
- the <mark>AUC (Area Under the Curve) of the ROC</mark>.
The ROC curve represents the values of TPrate versus FPrate for all possible choices of a threshold $\tau$.
The larger the AUC-ROC is (maximum of 1) the more discrimination is between the Positive and Negative classes.
A value of 0.5 indicates no discrimination (random system).
- the <mark>mean-Average-Precision (mAP)</mark>.
The mAP measures the AUC of the Precision versus Recall curve for all possible choices of a threshold $\tau)$.

*Note: The AUC-ROC is known to be sensitive to class imbalancing (in case of multi-label, negative examples are usually more numerous than positive ones, hence the FPrate is artificially low leading to good AUC of ROC).
In the opposite, mAP which relies on the Precision is less sensitive to class imbalancing and is therefore preferred.*

![AUC-ROC-MAP](/images/brick_roc_map_P.png)

```python
from sklearn.metrics import roc_auc_score, average_precision_score
roc_auc_score(labels_idx, labels_pred_prob, average="macro")
average_precision_score(labels_idx, labels_pred_prob, average="macro")
```

About the averages in `scikit-learn`:
- `macro` **average**: computes the metric independently for each class and then takes the average (i.e., all classes are treated equally, regardless of their frequency).
- `micro` **average**: aggregates the contributions of all classes before calculating the overall metric, essentially treating the problem as a single binary classification task across all samples




### Chord Estimation

In the following (for the sake of simplicity) we will evaluate our chord estimation system <mark>as a multi-class problem</mark>.

However, chord are not simple labels.
Indeed, chord annotation is partly subjective, some chords are equivalent, and the spelling of a chord depends on the choice of the level of detail (the choice of a dictionary).\
For this reason, `mir_eval` {cite}`DBLP:conf/ismir/RaffelMHSNLE14` or Pauwels et al. {cite}`DBLP:conf/icassp/PauwelsP13` proposed metrics that allows measuring the correctness of the `root`, the `major/minor` component, the `bass` or the constitution in terms of `chroma`.




## Some popular datasets

A (close to) exhaustive list of MIR datasets is available in the [ismir.net web site](https://ismir.net/resources/datasets/).

We have chosen the following ones since they are often used, they represent the multi-class, multi-label and chord estimation problems, and their audio is easely accessible.


#### GTZAN

[GTZAN](http://marsyas.info/downloads/datasets.html) contains 1000 audio files of 30s duration, each with a single (**multi-class**) genre label
- among 10 classes: 'blues','classical','country','disco','hiphop','jazz','metal','pop', 'reggae','rock'

Note that GTZAN has been criticized for the quality of its genre label {cite}`DBLP:journals/corr/Sturm13`; so results should be considered with cares.

```python
"entry": [
            {
                "filepath": [
                    {"value": "blues+++blues.00000.wav"}
                ],
                "genre": [
                    {"value": "blues"}
                ]
            },
            {
                "filepath": [
                    {"value": "blues+++blues.00001.wav"}
                ],
                "genre": [
                    {"value": "blues"}
                ]
            }
          ]
```
#### Magna-Tag-A-Tune

[Magna-Tag-A-Tune (MTT)](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset) is a **multi-label** large-scale dataset of 25,000 30-second music clips from various genres, each annotated with
- multiple tags describing genre, mood, instrumentation, and other musical attributes such as ('guitar', 'classical', 'slow', 'techno', 'strings', 'drums', 'electronic', 'rock', 'fast', 'piano', ...)

We only use a subset of this dataset by only selecting the most 50 used tags and further reducing the number of audio by 20.
```python
"entry": [
            {
                "filepath": [
                    {"value": "0+++american_bach_soloists-j_s__bach__cantatas_volume_v-01-gleichwie_der_regen_und_schnee_vom_himmel_fallt_bwv_18_i_sinfonia-117-146.mp3"}
                ],
                "tag": [
                    {"value": "classical"},
                    {"value": "violin"}
                ],
                "artist": [
                    {"value": "American Bach Soloists"}
                ],
                "album": [
                    {"value": "J.S. Bach - Cantatas Volume V"}
                ],
                "track_number": [
                    {"value": 1}
                ],
                "title": [
                    {"value": "Gleichwie der Regen und Schnee vom Himmel fallt BWV 18_ I Sinfonia"}
                ],
                "clip_id": [
                    {"value": 29}
                ],
                "original_url": [
                    {"value": "http://he3.magnatune.com/all/01--Gleichwie%20der%20Regen%20und%20Schnee%20vom%20Himmel%20fallt%20BWV%2018_%20I%20Sinfonia--ABS.mp3"}
                ],
                "segmentEnd": [
                    {"value": 146}
                ],
                "segmentStart": [
                    {"value": 117}
                ]
            },
          ]
```

#### RWC-Popular-Chord (AIST-Annotations)

[RWC-Popular-Chord (AIST-Annotations)](https://staff.aist.go.jp/m.goto/RWC-MDB/AIST-Annotation/){cite}`DBLP:conf/ismir/GotoHNO02`, {cite}`DBLP:conf/ismir/Goto06` is one of the earliest and remains one of the most comprehensive datasets, featuring annotations for genre, structure, beat, chords, and multiple pitches.\
We use the subset of 100 tracks named `Popular-Music-Dataset` and the **chord segments annotations** which we map to a simplified 25 elements dictionary `maj/min/N`. \
*This dataset has been made available online with Masataka Goto's permission specifically for this tutorial. For any other use, please contact Masataka Goto to obtain authorization.*

```python
"entry": [
            {
                "filepath": [
                    {"value": "001"}
                ],
                "chord": [
                    {"value": "N:-", "time": 0.0, "duration": 0.104},
                    {"value": "G#:min", "time": 0.104, "duration": 1.754},
                    {"value": "F#:maj", "time": 1.858,"duration": 1.7879999999999998},
                    {"value": "E:maj","time": 3.646,"duration": 1.7409999999999997},
                    {"value": "F#:maj", "time": 5.387, "duration": 3.6800000000000006},
                ]
            }
          ]
```





## How we can solve it using deep learning

Our goal is to show that we can <mark>solve the three tasks</mark> (multi-class GTZAN, multi-label MTT and chord segment estimation RWC-Pop) with a <mark>single code</mark>.
Depending on the task, we of course adapt the model (defined in the `.yaml` files).

<mark>multi-class/multi-label</mark>:
- GTZAN and RWC-Pop-Chord are **multi-class** problems $\Rightarrow$ softmax and categorial-CE
- MTT is **multi-label** $\Rightarrow$ sigmoids and BCEs

<mark>global/local</mark>:
- GTZAN and MTT have **global** annotations $\Rightarrow$ we reduce the time axis using [AutoPoolWeightSplit](lab_AttentionWeighting)
- RWC-Pop-Chord have **local** annotations with a language model $\Rightarrow$ we use a [RNN/bi-LSTM](lab_rnn).

For GTZAN and MTT our core model is the <mark>SincNet model</mark> illustrated below.

![sincnet](/images/brick_sincnet.png)\
**Figure**. *SincNet model. image source: SincNet {cite}`DBLP:conf/slt/RavanelliB18`*


We will vary in turn
- the **inputs $\Rightarrow$ front-end**:
	- Input: [waveform](lab_waveform) **$\Rightarrow$** Front-end: [Conv-1D](lab_conv1D), [TCN](lab_tcn), [SincNet](lab_sincnet),
	- Input: [Log-Mel-Spectrogram](lab_lms), [CQT](lab_cqt) **$\Rightarrow$** Front-end: [Conv-2d](lab_conv2d)
- the model **blocks**:
	- [Conv-1d](lab_conv1d), Linear and [AutoPoolWeightSplit](lab_AutoPoolWeightSplit) for multi-class, multi-label
	- [Conv-1d](lab_conv1d), Linear and [RNN/bi-LSTM](lab_rnn) for segment (chord over time)

![expe](/images/expe_autotagging_P.png)





### Experiments:

The code is available here:
- (Main notebook)(https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/blob/master/TUTO_task_Auto_Tagging.ipynb)
- (Config Auto-Tagging)[https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/blob/master/config_autotagging.yaml]
- (Config Chord)[https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/blob/master/config_chord.yaml]


| Dataset   | Input   | Frontend   | Model | Results   | Code |
|:---------- |:----------|:----------|:----------|:---------- |:---------- |
| GTZAN      | LMS       | Conv2d(128,5) | Conv1d/Linear/AutoPoolWeightSplit   | macroRecall: 0.56           | [LINK](https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/blob/master/TUTO_task_Auto_Tagging.ipynb_D1-I1-C1.ipynb) |
| GTZAN      | Waveform  | Conv1D 			 | Conv1d/Linear/AutoPoolWeightSplit   | macroRecall: 0.54           | [LINK](https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/blob/master/TUTO_task_Auto_Tagging.ipynb_D1-I2-C3.ipynb) |
| GTZAN      | Waveform  | TCN					 | Conv1d/Linear/AutoPoolWeightSplit   | macroRecall: 0.46           | [LINK](https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/blob/master/TUTO_task_Auto_Tagging.ipynb_D1-I2-C4.ipynb) |
| GTZAN      | Waveform  | SincNet/Abs   | Conv1d/Linear/AutoPoolWeightSplit   | macroRecall: 0.56           | [LINK](https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/blob/master/TUTO_task_Auto_Tagging.ipynb_D1-I2-C2.ipynb) |
| --  | -- | -- | -- | -- |
| MTT        | LMS       | Conv2d(128,5) | Conv1d/Linear/AutoPoolWeightSplit   | AUC: 0.81, avgPrec: 0.29    | [LINK](https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/blob/master/TUTO_task_Auto_Tagging.ipynb_D2-I1-C1.ipynb) |
| --  | -- | -- | -- | -- |
| RWC-Pop-Chord | CQT    | Conv2D(1,5)(5,1)* |	Conv1D/LSTM/Linear             | macroRecall: 0.54           | [LINK](https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/blob/master/TUTO_task_Auto_Tagging.ipynb_D3-I3-Chord.ipynb) |


### Code:

Illustrations of
- autotagging config file
- multi-class: results, CM and Tag-O-Gram
- multi-class:: learned filters SincNet, code SincNet
- multi-class: learned filters Conv1d
- multi-label: results, tag-o-gram:
- chord config file
- chord: training patches
- chord: results, tag-o-gram
