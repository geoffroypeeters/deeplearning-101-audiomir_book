# Auto-Tagging-FrontEnd


## Goal of the task ?

Music auto-tagging is the task of assigning tags (such as genre , style, moods, instrumentation) to a music track.
Tags can be mutually exclusive (**multi-class** problem) or not (**multi-label** problem) can be assigned over time (such as for segmentation into singing segments) or globally (for the whole time duration of the track).

![flow_autotagging](/images/flow_autotagging.png)

The task is one of the first studied in MIR.
Already in 2002 Tzanetakis et al. {cite}`DBLP:journals/taslp/TzanetakisC02` demonstrated that it is possible to estimate the genre using audio featrues (MFCC) and simple machine-learning models (GMM).
Later on, VGG-like architecture has been proposed to solve the task {cite}`DBLP:conf/ismir/ChoiFS16`.
The task is still active nowadays trying to solve it using Self-Supervised-Learning or Foundation models Nowadays, attempted to do solve this task using MFCC/GMM-like systems) and still very active with the evaluation toward the use of Self-Supervised-Learning and Foundation models {cite}`DBLP:conf/nips/YuanMLZCYZLHTDW23`.

Fore more details, see the very good [tutorial on "musical classification"](https://music-classification.github.io/tutorial/landing-page.html)


## How is the task evaluated ?

We consider the classes $c \in \{1,\ldots,C\}$.

### Multi-class
For the **multi-class** problem, the outputs of the model $o_c$ go to a softmax function (mutually exclusive classes). The outputs $p_c$ then represent $P(Y_c=1|X)$. The predicted class is then chosen as $\argmax_c p_c)$.

We evaluate the performances by computing the standard  Accuracy, Recall, Precision, F-measure for each class $c$ and then taking the average over classes $c$.
```python
from sklearn.metrics import classification_report, confusion_matrix
classification_reports = classification_report(labels_idx, labels_pred_idx, output_dict=True)
cm = confusion_matrix(labels_idx, labels_pred_idx)
```

### Multi-label
For the **multi-label** problem, each $o_c$ of the model goes individually to a sigmoid function (multi-label is processed as a set of parallel independent binary classification problems). The outputs $p_c$ then represent $P(Y_c=1|X)$. We then need to set a threshold $\tau$ on each probability to make the decision wether class $c$ exist or not.

Using a default threshold ($\tau=0.5$) of course allows to use the afore-mentioned mertics (Accuracy, Recall, Precision, F-measure).
However, in practice, we want to measure the performances independently of the choice of a threshold.
This can be done either by
- measuring the **AUC (Area Under the Curve) of the ROC**. The ROC curve represents the values of TPrate versus FPrate for all possible choices of a threshold $\tau)$. The larger the AUC is (maximum of 1) the more discrimination is between the Positive and Negative classes. A value of 0.5 indicates no discrimination (random system).
- computing the **mean-Average-Precision (mAP)**. The mAP measures the AUC of the Precision versus Recall curve for all possible choices of a threshold $\tau)$.

The AUC of ROC is known to be sensitive to class imbalancing (in case of multi-label, negative examples are usually more numerous than positive ones, hence the FPrate is artificially low leading to good AUC of ROC).
In the opposite, mAP which relies on the Precision is less sensitive to class imbalancing and is therefoe prefered.

![auc](/images/brick_auc.png)
![map](/images/brick_map.png)

```python
from sklearn.metrics import roc_auc_score, average_precision_score
roc_auc_score(labels, predictions, average="macro")
average_precision_score(labels, predictions, average="macro")
```

Averages in scikitlearn:
- **Macro average**: computes the metric independently for each class and then takes the average (i.e., all classes are treated equally, regardless of their frequency).
- **Micro average**: aggregates the contributions of all classes before calculating the overall metric, essentially treating the problem as a single binary classification task across all samples


## Some popular datasets

A (close to) exhaustive list of MIR datasets is available in the [ismir.net web site](https://ismir.net/resources/datasets/).

Many datasets exist for music-auto-tagging such as AcousticBrainz-Genre, AudioSet (music part), CAL10K, CAL500, FMA-Full/Medium/Small, IRMAS (instruments), Jamendo (vocal activity), MTG-Jamendo (genre, instruments, mood), Seyerlehner/*, ...

We have chosen the two following ones since they are often used, they represent the multi-class and multi-label problem, their audio is easely accessible.

For our implementations, we will consider the two following datasets

- [GTZAN](http://marsyas.info/downloads/datasets.html).
It contains 1000 audio files of 30s duration, each with a single (multi-class) genre label among 10 classes ('blues','classical','country','disco','hiphop','jazz','metal','pop', 'reggae','rock').
Although GTZAN has been criticized for the quality of its label we only used to exemplify our models.

- [MagneTagATune (MTT)](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset).
We use a subset of this dataset by only selecting the most 50 used tags.
It contains 21.108 files of 30s duration, each with multiple (multi-label) tag labels among 50 classes ('guitar', 'classical', 'slow', 'techno', 'strings', 'drums', 'electronic', 'rock', 'fast', 'piano', ...)





## How we can solve it using deep learning

Auto-tagging is a classification problem and can be considered either as a multi-class (mutually exclusive classes, such as for GTZAN) or a multi-label (non-mutually exclusive classes, such as MTT).
Also, for the two considered datasets (GTZAN, MTT) the labels are assigned or the whole track duration.
We therefore need to design a model that map a time-serie of observation to a single output.
There have been many models proposed to do this.

For this tutorial, we focus on the model used in the SincNet paper illustrated below.

![sincnet](/images/brick_sincnet.png)

Using this architecture as a baseline, we will vary in turn
- the **inputs**: [waveform](lab_waveform) or [Log-Mel-Spectrogram](lab_lms)
- the **front-end**: [Conv-1D](lab_conv1D), [TCN](lab_tcn), [SincNet](lab_sincnet), [Conv-2D](lab_conv2D)
- the model **blocks**: [Conv-2D](lab_conv2d), [ResNet](lab_resnet), [ConvNext](lab_convnext)

![bricks](/images/main_bricks.png)
