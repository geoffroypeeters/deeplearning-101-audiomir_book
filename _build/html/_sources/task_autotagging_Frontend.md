(lab_auto_tagging)=
# Auto-Tagging-FrontEnd


## Goal of the task ?

Music auto-tagging is the task of assigning tags (such as genre, style, moods, instrumentation, chords) to a music track.
Tags can be
- mutually exclusive (**multi-class** problem, such as genre) or not (**multi-label** problem, such as instrumentation)
- can be assigned **locally** in time (such as instrumentation-segments, or chord-label segments) or **globally** in time (such as for music-genre).

![flow_autotagging](/images/flow_autotagging.png)

**A very short history.**
The task has a long history in MIR.
As soon as 2002 Tzanetakis et al. {cite}`DBLP:journals/taslp/TzanetakisC02` demonstrated that it is possible to estimate the `genre` using a set of low-level (hand-crafted) audio features (such as MFCC) and simple machine-learning models (such as Gaussian-Mixture-Models).
The audio features considered improved over years {cite}`Peeters2004AudioFeatures`, from block-features {cite}`Seyerlehner2010PHD` to speech-inspired features (Universal-Background-Models and Super-Vector {cite}`Charbuillet2011DAFX`), as well as the machine-learning models (moving to Support-Vector-Machine).
It also quickly appeared that the same feature/ML system could be trained to solve many tasks of tagging or segmentation (genre, mood, speech/music) {cite}`Peeters2007DAFXGenericClassification`, {cite}`Burred2009LSASMultiLabel`.

Chord estimation can be considered as a specific tagging application: it involves applying chord-label-tags (mutually exclusive) over segments of time.
However, it has been considered as a specific task since chord transition follow musical rules which can be represented by a language model.
Therefore, ASR (Automatic Speech Recognition) inspired techniques has been developed at first {cite}`Sheh2003ISMIRchord` or {cite}`Papadopo2007CBMI` with an acoustic model representing $p(\text{chord}|\text{chroma})$ and a language model using HMM (Hidden Markov Model) representing $p(\text{chord}_{t}|\text{chord}_{t-1}).$

One of the first successful application of deep learning for the auto-tagging task is the work of Dieleman {cite}`Dieleman2014Spotify`.
In this a Conv2d is applied to a Log-Mel-Spectrogram using kernels which extend over the whole frequency range, therefore only a convolution over time is performed.
The rational for this, is that as opposed to natural images, objects in a T/F representation are not invariant by translation over frequencies and the adjacent frequencies are not necesseraly correlated (spacing between harmonics).
Despite this, Choi et al. {cite}`DBLP:conf/ismir/ChoiFS16` proposed (with success) to apply Computer Vision VGG-like architecture to a time-frequency representation.
Later on, Pons et al. {cite}`DBLP:conf/cbmi/PonsLS16` proposed to design kernel shapes using musical consideration (extending over fequencies to represent timbre, over time to represent rhythm).
Using directly the audio waveform (End-to-end) system has also been proposed for this task, such as in Dieleman et al. {cite}`DBLP:conf/icassp/DielemanS14` or Lee et al. {cite}`DBLP:journals/corr/LeePKN17`.
The task of auto-tagging has also close relationship with their equivalent task in Speech.

<mark>*We will develop here a model developed initially for speaker recognition by Ravanelli et al. {cite}`DBLP:conf/slt/RavanelliB18`.*</mark>

In the case of chord estimation, deep learning is also now commonly used.
One seminal paper proposed by McFee at al. {cite}`DBLP:conf/ismir/McFeeB17` relies on a RCNN (ConvNet followed by a bi-directional RNN, here GRU) to perform the task. Their model also forces an inner representation to relate to the `root`, `bass` and `pitches`. This is done using a multi-task approach (the model is trained to minimize several losses jointly). This favors the learning of representation which brings similar chords (but with different labels) closer.

<mark>*We will develop here a similar model based on the combination of Conv2d and Bi-LSTM but without the multi-task approach.*</mark>

The task is still very active today, even in the supervised case.
For example,
- MULE {cite}`DBLP:conf/ismir/McCallumKOGE22` use a more sophisticated ConvNet architecture (Short-Fast-Normalizer-Free Net F0) and training paradigm (contrastive learning based on artist, album or tags) and is trained on a very large music collection (Pandora), or
- PaSST {cite}`DBLP:conf/hear/KoutiniMSESW21` is based on the Vit, and use tokenized (set of patches) spectrograms fed to a Transformer

The task is still active nowadays especially using Self-Supervised-Learning {cite}`DBLP:conf/nips/YuanMLZCYZLHTDW23` (see second part).

Fore more details, see the very good [tutorial on "musical classification"](https://music-classification.github.io/tutorial/landing-page.html)


## How is the task evaluated ?

We consider a set of classes $c \in \{1,\ldots,C\}$.

### Multi-class

In a **multi-class** problem, the classes are mutually exclusive.
The outputs of the (neural network) model $o_c$ therefore go to a softmax function.
The outputs of the softmax, $p_c$, then represent the probability $P(Y=c|X)$.
The predicted class is then chosen as $\arg\max_c p_c)$.

We evaluate the performances by computing the standard  Accuracy, Recall, Precision, F-measure for each class $c$ and then take the average over classes $c$.
```python
from sklearn.metrics import classification_report, confusion_matrix
classification_reports = classification_report(labels_idx, labels_pred_idx, output_dict=True)
cm = confusion_matrix(labels_idx, labels_pred_idx)
```

### Multi-label

In the **multi-label** problem, the classes are NOR mutually exclusive.
Each $o_c$ therefore goes individually to a sigmoid function (multi-label is processed as a set of parallel independent binary classification problems).
The outputs of the sigmoids $p_c$ then represent $P(Y_c=1|X)$.
We then need to set a threshold $\tau$ on each $p_c$ to decide wether class $c$ exist or not.

Using a default threshold ($\tau=0.5$) of course allows to use the afore-mentioned metrics (Accuracy, Recall, Precision, F-measure).
However, in practice, we want to measure the performances independently of the choice of a given threshold.
This can be using either
- the **AUC (Area Under the Curve) of the ROC**.
The ROC curve represents the values of TPrate versus FPrate for all possible choices of a threshold $\tau$.
The larger the AUC-ROC is (maximum of 1) the more discrimination is between the Positive and Negative classes.
A value of 0.5 indicates no discrimination (random system).
- the **mean-Average-Precision (mAP)**.
The mAP measures the AUC of the Precision versus Recall curve for all possible choices of a threshold $\tau)$.

The AUC-ROC is known to be sensitive to class imbalancing (in case of multi-label, negative examples are usually more numerous than positive ones, hence the FPrate is artificially low leading to good AUC of ROC).
In the opposite, mAP which relies on the Precision is less sensitive to class imbalancing and is therefoe prefered.

![AUC-ROC-MAP](/images/brick_roc_map_P.png)

```python
from sklearn.metrics import roc_auc_score, average_precision_score
roc_auc_score(labels, predictions, average="macro")
average_precision_score(labels, predictions, average="macro")
```

Averages in scikitlearn:
- **Macro average**: computes the metric independently for each class and then takes the average (i.e., all classes are treated equally, regardless of their frequency).
- **Micro average**: aggregates the contributions of all classes before calculating the overall metric, essentially treating the problem as a single binary classification task across all samples

### Chord segments

Evaluating a chord estimation system can be done as a multi-class problem (for simplicity this what will be performed in the following).
However, chord are not simple labels.
Indeed, chord annotation is partly subjective, some chord are equivalent, and the spelling of a chord depends on the choice of the level of detail (the choice of a dictionary).
For this reason, `mir_eval` {cite}`DBLP:conf/ismir/RaffelMHSNLE14` or Pauwels et al. {cite}`DBLP:conf/icassp/PauwelsP13` proposed metrics that allows measuring the correctness of the `root`, the `major/minor` component, the `bass` or the constitution in terms of `chroma`.


## Some popular datasets

A (close to) exhaustive list of MIR datasets is available in the [ismir.net web site](https://ismir.net/resources/datasets/).

Many datasets exist for music-auto-tagging such as AcousticBrainz-Genre, AudioSet (music part), CAL10K, CAL500, FMA-Full/Medium/Small, IRMAS (instruments), Jamendo (vocal activity), MTG-Jamendo (genre, instruments, mood), Seyerlehner/*, ...

We have chosen the two following ones since they are often used, they represent the multi-class and multi-label problem, their audio is easely accessible.

For our implementations, we will consider the two following datasets

- [GTZAN](http://marsyas.info/downloads/datasets.html).
It contains 1000 audio files of 30s duration, each with a single (**multi-class**) genre label among 10 classes ('blues','classical','country','disco','hiphop','jazz','metal','pop', 'reggae','rock').
Although GTZAN has been criticized for the quality of its label we only used to exemplify our models.

- [MagneTagATune (MTT)](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset).
We only use a subset of this dataset by only selecting the most 50 used tags and further reducing the number of audio by 20.
It contains 21.108 files of 30s duration, each with multiple (**multi-label**) tag labels among 50 classes ('guitar', 'classical', 'slow', 'techno', 'strings', 'drums', 'electronic', 'rock', 'fast', 'piano', ...)

- [RWC-Popular-Chord (AIST-Annotations)](https://staff.aist.go.jp/m.goto/RWC-MDB/AIST-Annotation/){cite}`DBLP:conf/ismir/GotoHNO02`, {cite}`DBLP:conf/ismir/Goto06`
This dataset was made accessible online with the permission of Masataka Goto for the specific purpose of this tutorial.
For any other use, please contact Mastaka Goto for auhtorization.
The RWC dataset is one of the earliest and remains one of the most comprehensive datasets, featuring annotations for genre, structure, beat, chords, and multiple pitches.





## How we can solve it using deep learning


Auto-tagging is a classification problem and can be considered either as a multi-class (mutually exclusive classes, such as for GTZAN) or a multi-label (non-mutually exclusive classes, such as MTT).
Also, for the two considered datasets (GTZAN, MTT) the labels are assigned or the whole track duration.
We therefore need to design a model that map a time-serie of observation to a single output.
There have been many models proposed to do this.

For this tutorial, we focus on the model used in the SincNet paper illustrated below.

![sincnet](/images/brick_sincnet.png)

We illustrate a deep learning solution to this problem in the following [notebook](https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/blob/master/TUTO_task_Auto_Tagging.ipynb) and study various [configurations](https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/blob/master/config_autotagging.yaml).



### Experiments:

We will vary in turn
- the **inputs**: [waveform](lab_waveform), [Log-Mel-Spectrogram](lab_lms) or [CQT](lab_cqt)
- the **front-end**:
	- [Conv-2d](lab_conv2D) when the input is LMS or CQT
	- [SincNet](lab_sincnet), [Conv-1D](lab_conv1D) or [TCN](lab_tcn) when the input is waveform
- the model **blocks**:
	- [Conv-1d](lab_conv1d), Linear and [AutoPoolWeightSplit](lab_AutoPoolWeightSplit) for multi-class, multi-label
	- Conv-1d](lab_conv1d), Linear and [RNN/LSTM](lab_rnn) for segment (chord over time)

![expe](/images/expe_autotagging_P.png)

This can be done using the following files:
- (Main notebook)(https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/blob/master/TUTO_task_Auto_Tagging.ipynb)
- (Config Auto-Tagging)[https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/blob/master/config_autotagging.yaml]
- (Config Chord)[https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/blob/master/config_chord.yaml]

| Dataset   | Input   | Frontend   | Model | Results   | Code |
|:---------- |:----------|:----------|:----------|:---------- |:---------- |
| GTZAN      | LMS       | Conv2d(128,5) | Conv1d/Linear/AutoPoolWeightSplit   | macroRecall: 0.56           | [LINK](https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/blob/master/TUTO_task_Auto_Tagging.ipynb_D1-I1-C1.ipynb) |
| GTZAN      | Waveform  | SincNet/Abs   | Conv1d/Linear/AutoPoolWeightSplit   | macroRecall: 0.56           | [LINK](https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/blob/master/TUTO_task_Auto_Tagging.ipynb_D1-I2-C2.ipynb) |
| GTZAN      | Waveform  | Conv1D 			 | Conv1d/Linear/AutoPoolWeightSplit   | macroRecall: 0.54           | [LINK](https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/blob/master/TUTO_task_Auto_Tagging.ipynb_D1-I2-C3.ipynb) |
| GTZAN      | Waveform  | TCN					 | Conv1d/Linear/AutoPoolWeightSplit   | macroRecall: 0.46           | [LINK](https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/blob/master/TUTO_task_Auto_Tagging.ipynb_D1-I2-C4.ipynb) |
| MTT        | LMS       | Conv2d(128,5) | Conv1d/Linear/AutoPoolWeightSplit   | AUC: 0.81, avgPrec: 0.29    | [LINK](https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/blob/master/TUTO_task_Auto_Tagging.ipynb_D2-I1-C1.ipynb) |
| RWC-Pop-Chord | CQT    | Conv2D(1,5)(5,1)* |	Conv1D/LSTM/Linear             | macroRecall: 0.54           | [LINK](https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/blob/master/TUTO_task_Auto_Tagging.ipynb_D3-I3-Chord.ipynb) |
