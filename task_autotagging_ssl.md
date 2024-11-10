# Tutorial: Deep Learning 101 for Audio-based MIR - Self-supervised learning

*Geoffroy Peeters, Gabriel Meseguer-Brocal, Alain Riou, Stefan Lattner*

Self-supervised learning (SSL) is a paradigm in machine learning that aims to learn meaningful representations from data without relying on labels. The goal is to leverage the natural structure in data to define tasks that can guide the model’s learning process. This approach has grown in popularity for applications in computer vision, natural language processing, and audio processing, where labeled data is often scarce or costly to obtain.

In this tutorial, we’ll focus on **Siamese architectures**, which are particularly well-suited to SSL. Siamese networks learn by comparing **positive pairs** of inputs, effectively constructing a task from data itself to learn useful representations.

**Disclaimer:** This book incorporates bricks of Python code for the main components described. The full runnable code for training and evaluating a SSL model is provided in this [Jupyter notebook](https://colab.research.google.com/drive/1lVpAKC1Tc8BRDKYnyna6IplxzpW7rdby?usp=sharing).

### Key Concepts in Siamese Architectures for SSL

A Siamese network takes two input samples $x_1$ and $x_2$ and projects them into a shared latent space using a neural network $f_{\theta}$ with parameters $\theta$. The underlying idea is simple yet powerful:

1. **Positive pairs**: Each input pair $ (x_1, x_2) $ is chosen such that $x_1$ and $x_2$ are related in some way (e.g., two different views of the same object or nearby audio chunks in a music track). Since these inputs are “similar,” we want their representations in the latent space to be close.

2. **Learning Objective**: To achieve this, we optimize the parameters $\theta$ of $f_{\theta}$ by minimizing the distance between $f_{\theta}(x_1)$ and $f_{\theta}(x_2)$ in the latent space.

Training the model in this way encourages it to focus on commonalities between similar samples, capturing meaningful and invariant features in the data. The resulting learned representations are useful for downstream tasks, even when labeled data is unavailable.

![Siamese network](https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/raw/ssl/images/ssl/a4ece110-e64b-4362-ad4d-28cdcf396963.png)


## 1. Training a SSL model

In this part, we will implement the different components of a Siamese architecture, namely its structure, the underlying backbone architecture and the criterion to optimize. In addition, we implement the mechanism to create the positive pairs.

### 1.1. Building a dataset of positive pairs

The first step is to define what are the positive pairs that our model to process, and how to build them. In practice, this mechanism will be directly implemented in a standard PyTorch `Dataset`.

When dealing with music, common strategies to create pairs of similar points without supervision are:
- extracting different chunks from the same song
- applying audio effects to the audio chunk

Recall that the network will learn to find what the elements of the pair have **in common**. For example, if you use chunks of the same song, it will probably capture info such as tonality, genre, tempo, but not chords or timbre. On the contrary, if you transpose your audio to create pairs, it will learn to discard pitch information.

In other words, you control what your model captures by choosing how you compute pairs. And of course, you can combine different techniques!

Let's start by implementing a simple dataset that extracts chunks of audio and randomly apply several transforms.
The convenient thing when training a SSL model is that we do not need any label, so we can recursively explore folders and use any audio data we find, which makes the `__init__` function super simple.

For the transforms, we can directly use the transforms implemented in the [torchaudio-augmentations](https://github.com/Spijkervet/torchaudio-augmentations) repository from Janne Spijkervet (who did a tutorial about SSL in ISMIR 2021, btw). Our `__getitem__` function then just picks an audio, extracts two chunks from it and randomly applies transforms before returning the pair.


```python
class PairDataset(torch.utils.data.Dataset):
    """
    A custom dataset class that retrieves pairs of random audio chunks 
    from WAV files in a specified directory.

    Args:
        data_dir (str): Path to the directory containing WAV files.
        chunk_duration (float): Duration of the audio chunks in seconds. Defaults to 10 seconds.

    Attributes:
        paths (List[str]): List of paths to all WAV files in the dataset.
        chunk_duration (float): Duration of the audio chunks to be extracted.
        transforms (torch.nn.Module): Placeholder for audio augmentation/transforms.
    """
    def __init__(self, data_dir: str, chunk_duration: float = 10.):
        self.paths = glob.glob(f"{data_dir}/**/*.mp3", recursive=True)
        self.chunk_duration = chunk_duration

        # Define the set of transforms here
        self.transforms = nn.Sequential(*[
            RandomApply([PolarityInversion()], p=0.8),
            RandomApply([Noise(min_snr=0.001, max_snr=0.005)], p=0.3),
            RandomApply([Gain()], p=0.2),
            HighLowPass(sample_rate=22050), # this augmentation will always be applied in this aumgentation chain!
            RandomApply([Delay(sample_rate=22050)], p=0.5),
            RandomApply([Reverb(sample_rate=22050)], p=0.3)
        ])

    def __len__(self) -> int:
        """
        Returns the number of audio files in the dataset.

        Returns:
            int: The total number of WAV files in the dataset.
        """
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a pair of random audio chunks from a WAV file.

        Args:
            idx (int): Index of the file to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Two randomly extracted mono audio chunks 
            as tensors from the selected WAV file.
        """
        # Retrieve file path and audio info
        path = self.paths[idx]
        info = torchaudio.info(path)
        sr = info.sample_rate
        total_frames = info.num_frames

        # Calculate the number of frames per chunk
        num_frames = int(self.chunk_duration * sr)

        # Randomly select starting points for two chunks
        i1, i2 = torch.randint(0, total_frames - num_frames, (2,))

        # Load two audio chunks from the file
        x1, _ = torchaudio.load(path, frame_offset=i1, num_frames=num_frames)
        x2, _ = torchaudio.load(path, frame_offset=i2, num_frames=num_frames)

        # Apply transforms (if any) to both chunks
        x1 = self.transforms(x1)
        x2 = self.transforms(x2)
        
        # Convert stereo to mono by summing across the channel dimension
        x1 = x1.sum(dim=0)
        x2 = x2.sum(dim=0)

        return x1, x2
```



### 1.2. Implementing Siamese Networks as a `LightningModule`

Siamese Networks are a simple structure just composed of two... siamese networks, and each of them projects an element of the pair in the latent space. Then a criterion between the two projections is being optimized.

*But why only two? Couldn't we use more than only two views?* Actually yes, some recent works such as [this article from ICLR 2024 by Shidani et al.](https://arxiv.org/pdf/2403.05490) suggest that Siamese networks can be generalized to more than pairs of two views. However, as 99% of the papers, we will stick to pairs in this tutorial.

Let us build our Siamese networks as a `LightningModule`. Recall that it is just a training paradigm that does not depend on the underlying architectures, so we can make it quite modular. Our LightningModule just takes two main arguments:
- The architecture of the network itself
- The loss criterion that we want to optimize

That's it!

An interesting trick to improve the generalization abilities of the learned embeddings is not to use the outputs of the last layer of the network but a previous one after training. Doing so enables the mitigation of the misalignment between the training objective and the actual downstream applications. Some French researchers named this trick Guillotine Regularization and studied it in depth in [this journal paper from Bordes et al.](https://arxiv.org/pdf/2206.13378).

In practice, we implement this trick by splitting our network in two successive parts, usually referred to as the **encoder** and the **projector**. In our case, as often, we will use a domain-specific architecture for the encoder and a simple MLP with 2 layers for the projector.


```python
class SiameseNetwork(pl.LightningModule):
    """
    A Siamese Network implemented using PyTorch Lightning, designed to work with
    a backbone neural network and a loss function. The network projects two input
    samples into a latent space and optimizes their relationship via the provided loss function.

    Args:
        encoder (torch.nn.Module): The feature extractor model that projects inputs into a latent space.
        loss_fn (torch.nn.Module): The loss function used to optimize the network based on the similarity 
                                   or dissimilarity between the two inputs.
    """
    def __init__(self,
                 encoder: torch.nn.Module,
                 loss_fn: torch.nn.Module,
                 in_channels: int = 512,
                 out_channels: int = 64):
        super(SiameseNetwork, self).__init__()

        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, out_channels)
        )
        
        self.loss_fn = loss_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the backbone network.

        Args:
            x (torch.Tensor): Input tensor to be projected into the latent space.

        Returns:
            torch.Tensor: Latent representation of the input.
        """
        y = self.encoder(x)
        return self.projector(y)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Defines a single training step, including the forward pass and loss computation.

        Args:
            batch (Any): A batch of data, expected to contain two input tensors.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The computed loss for this step.
        """
        x1, x2 = batch

        # Project x1 and x2 into the latent space
        z1 = self.forward(x1)
        z2 = self.forward(x2)

        # Compute the loss based on z1 and z2
        loss = self.loss_fn(z1, z2)

        # Log the loss for visualization and monitoring
        self.log("loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures the optimizer for training.

        Returns:
            torch.optim.Optimizer: Adam optimizer with a learning rate of 1e-4.
        """
        return torch.optim.Adam(self.parameters(), lr=1e-4)
```

### 1.3. Architecture of the model

Now let us define those two components.
In the notebook, we use [SampleCNN](https://github.com/kyungyunlee/sampleCNN-pytorch) for the architecture of the backbone and we define it as a simple PyTorch `nn.Module`.
However, the choice of the architecture is independent from the paradigm: any neural architecuture can be used to do SSL, SampleCNN is just an example.


```python
class SampleCNN(nn.Module):
    def __init__(self):
        super(SampleCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU())
        ...
        
        

    def forward(self, x):
        x = x.view(x.shape[0], 1, -1)
        out = self.conv1(x)
        ...
        
        # average along time axis to get a single embedding per audio
        return out.mean(dim=-1)
```

### 1.4. Loss function

Recall that Siamese networks consist in pushing together the representations of inputs that are similar. However, if we only train our model to minimize the Mean Squared Error between positive pairs, we observe an undesirable phenomenon: the network will project everything to the same point, discarding any information from the input.

This phenomenon is called **collapse**, and a lot of research has been about how to prevent this phenomenon to happen. In particular, the most widely technique is to use a contrastive learning. In other words, we want to have **positive pairs** that we push together, but also **negative pairs** that we push far away from each other.

![Negative pairs](https://github.com/geoffroypeeters/deeplearning-101-audiomir_book/raw/ssl/images/ssl/5e51a5cc-b970-44f9-9ba6-7ccfc2833399.png)

*We know how to choose the positive pairs, but how to choose the negative ones?* Well, a simple yet effective is to say that everything that is not a positive pair is a negative pair. Practically speaking, since we anyway process batches of inputs, we use the other elements of a batch to create these negative pairs.
Given a pair of two batches of size $N$, we concatenate both into a big matrix $Z = (z_1, \dots, z_{2N}) \in \mathbb{R}^{2N \times d}$. For $1 \leq i \leq 2N$, let $i^+$ be the index of the corresponding positive pair (i.e. $i^+ = i \pm N$). Overall, the formula looks like this:

![Contrastive loss](https://github.com/geoffroypeeters/deeplearning-101-audiomir_book/raw/ssl/images/ssl/5f59560e-eda0-43b3-8def-6f24c39f3ff1.png)

where $\text{sim}(z_i, z_j)$ denotes the cosine similarity between vectors $z_i$ and $z_j$ and $\tau$ is a fixed temperature hyperparameter.


```python
class ContrastiveLoss(nn.Module):
    """
    A contrastive loss function designed for self-supervised learning. It computes
    the similarity between two sets of embeddings (z1, z2) and aims to maximize the similarity 
    between positive pairs (same inputs) and minimize it between negative pairs (different inputs).

    Args:
        temperature (float): A scaling factor applied to the similarity scores. Defaults to 0.1.
    """
    def __init__(self, temperature: float = 0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Computes the contrastive loss between two sets of embeddings.

        Args:
            z1 (torch.Tensor): A batch of embeddings from the first view (e.g., first audio chunk).
            z2 (torch.Tensor): A batch of embeddings from the second view (e.g., second audio chunk).

        Returns:
            torch.Tensor: The contrastive loss computed from the similarity between positive and negative pairs.
        """
        n = z1.size(0)
        
        # Concatenate z1 and z2 along the batch dimension and normalize them
        z = torch.nn.functional.normalize(torch.cat((z1, z2)))

        # Compute cosine similarity matrix scaled by temperature
        sim = torch.mm(z, z.t()).div_(self.temperature)

        # Positive loss: average of n-diagonal elements (corresponding to positive pairs)
        pos_loss = -torch.diag(sim, diagonal=n).mean()

        # Negative loss: log of the sum of the exponentiated similarities for negative pairs
        exp_sim = sim.exp_().clone()  # Avoid in-place ops that interfere with autograd
        exp_sim.fill_diagonal_(0)     # Set diagonal elements (positive pairs) to 0

        neg_loss = exp_sim.sum(dim=1).log_().mean()

        # Return the combined loss (positive + negative)
        return pos_loss + neg_loss

```

### 1.5. Train!

We now have all the elements to train a model:

- A `Dataset` object that yields positive pairs of audio chunks
- The general Siamese Networks training pipeline (in a `LightningModule`)
- An appropriate architecture for the encoder (SampleCNN, in a `nn.Module`)
- An objective to minimize that pushes the projections of positive pairs together, but also prevents collapse

Here is a minimal code example to train a SSL model using Lightning's `Trainer` with all the elements we described above:


```python
# build dataloader
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=96,              # use approximately 8 GB of memory with SampleCNN as encoder
                                         num_workers=os.cpu_count(), # use one worker for computing batches of pairs per CPU you have
                                         shuffle=True,               # elements of the dataset should be sent in a random order
                                         pin_memory=True,            # I never understood this but set it to True
                                         persistent_workers=True)    # reuse the existing workers instead of creating new ones between each epoch


# Build the model
sample_cnn = SampleCNN()
model = SiameseNetwork(sample_cnn, loss_fn=ContrastiveLoss(temperature=0.5))

# Define the trainer. To speed-up training and reduce memory footprint, we use 16-bit mixed precision
trainer = pl.Trainer(accelerator='gpu', max_epochs=500)

# Train!
trainer.fit(model, dataloader)
```

Note how modular are the different blocks; the dataset implementation is completely independent from the loss which is independent from the architecture, etc.

In practice you can imagine alternative ways to sample pairs, use a different frontend/architecture, optimize another loss function, etc. or ***any combination of these!*** Actually, all of these design choices are interesting research directions that led to many publications.



## 2. Evaluation

To evaluate SSL models, we typically combine the encoder and a linear probe. First, the encoder is frozen to prevent any further updates to its parameters. This ensures that we are evaluating the representations learned during the self-supervised phase. Next, a simple linear classifier (linear probe) is trained on top of these frozen features using labeled data. The performance of this linear probe, typically measured through metrics like accuracy, mean Average Precision (mAP), or ROC-AUC, provides an indication of the quality of the learned representations. This evaluation method effectively assesses how well the SSL model has captured useful features from the data.

![Evaluation](https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook/raw/ssl/images/ssl/c99fc9dc-3b54-4796-9339-c23415f52bff.png)

In this tutorial, we focus on a music tagging task on a subset of MagnaTagATune. It is modeled as a multilabel classification problem.

### 2.1. Loading the small annotated dataset

In this part, we rely on a small **annotated** dataset.


```python
pd.read_csv("mtt_ssl/train/annotations.csv")
```

| mp3_path                                                                                         | guitar | classical | slow | techno | drums | strings | rock | electronic | fast | ... | choir | no voice | dance | metal | voice | male voice | country | harp | male vocals | electro |
|--------------------------------------------------------------------------------------------------|--------|-----------|------|--------|-------|---------|------|------------|------|-----|-------|----------|-------|-------|-------|------------|---------|------|-------------|---------|
| 0/american_bach_soloists-j_s__bach__cantatas_v...                                               | 0      | 1         | 0    | 0      | 0     | 0       | 0    | 0          | 0    | ... | 0     | 0        | 0     | 0     | 0     | 0          | 0       | 0    | 0           | 0       |
| f/the_headroom_project-jetuton_andawai-01-lind...                                               | 0      | 0         | 0    | 0      | 0     | 0       | 0    | 0          | 0    | ... | 0     | 0        | 0     | 0     | 0     | 0          | 0       | 0    | 0           | 0       |
| 9/american_bach_soloists-heinrich_schutz__musi...                                               | 0      | 1         | 0    | 0      | 0     | 0       | 0    | 0          | 0    | ... | 0     | 0        | 0     | 0     | 0     | 0          | 0       | 0    | 0           | 0       |
| 9/american_bach_soloists-heinrich_schutz__musi...                                               | 0      | 0         | 0    | 0      | 0     | 0       | 0    | 0          | 0    | ... | 0     | 0        | 0     | 0     | 0     | 0          | 0       | 0    | 0           | 0       |
| 9/american_bach_soloists-heinrich_schutz__musi...                                               | 0      | 0         | 0    | 0      | 0     | 0       | 0    | 0          | 0    | ... | 0     | 0        | 0     | 0     | 0     | 1          | 0       | 0    | 0           | 0       |
| ...                                                                                             | ...    | ...       | ...  | ...    | ...   | ...     | ...  | ...        | ...  | ... | ...   | ...      | ...   | ...   | ...   | ...        | ...     | ...  | ...         | ...     |
| 3/musica_franca-boismortier__sonatas_for_two_b...                                               | 0      | 1         | 0    | 0      | 0     | 1       | 0    | 0          | 0    | ... | 0     | 0        | 0     | 0     | 0     | 0          | 0       | 0    | 0           | 0       |
| 3/musica_franca-boismortier__sonatas_for_two_b...                                               | 0      | 1         | 0    | 0      | 0     | 0       | 0    | 0          | 1    | ... | 0     | 1        | 0     | 0     | 0     | 0          | 0       | 0    | 0           | 0       |
| f/magnaloops-electronica_loops_1-43-osxivilion...                                               | 0      | 0         | 1    | 0      | 0     | 0       | 0    | 0          | 0    | ... | 0     | 1        | 0     | 0     | 0     | 0          | 0       | 0    | 0           | 0       |
| 8/jacob_heringman-blame_not_my_lute-57-lost_is...                                               | 1      | 0         | 1    | 0      | 0     | 1       | 0    | 0          | 0    | ... | 0     | 0        | 0     | 0     | 0     | 0          | 0       | 1    | 0           | 0       |
| 8/jacob_heringman-blame_not_my_lute-57-lost_is...                                               | 1      | 1         | 1    | 0      | 0     | 1       | 0    | 0          | 0    | ... | 0     | 0        | 0     | 0     | 0     | 0          | 0       | 0    | 0           | 0       |

<p>2594 rows × 51 columns</p>



The quantity of annotated data is 10x smaller than the non-annotated one. We will now use this small quantity of data to train a simple linear probe in a supervised way, on top of the learned embeddings.

Let us first build our `MultiLabelDataset` class.


```python
class MultiLabelDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, duration: float = None, sample_rate: int = 22050):
        """
        Args:
            data_dir (str): Root directory of the dataset, containing the annotations.csv and audio files.
            duration (float, optional): Duration of audio samples in seconds. If None, load full audio.
            sample_rate (int): The sample rate to use when loading audio.
        """
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.duration = duration

        # Define the path to the annotations file inside the data_dir
        annotations_file = os.path.join(data_dir, "annotations.csv")

        # Load annotations from CSV
        self.annotations = pd.read_csv(annotations_file)

        # Extract wav paths and labels from the CSV
        self.paths, self.labels = [], []
        for path, label in zip(self.annotations["mp3_path"].values, self.annotations.drop(columns=["mp3_path"]).values):
            if os.path.exists(os.path.join(self.data_dir, path)):
                self.paths.append(path)
                self.labels.append(label)

        self.num_frames = int(self.sample_rate * self.duration)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        # Get the file path and the corresponding label
        audio_path = os.path.join(self.data_dir, self.paths[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # Load audio file
        waveform, sr = torchaudio.load(audio_path, num_frames=self.num_frames)

        return waveform, label

```

### 2.2. Linear probe as a `LightningModule`

We then define our linear probe as a `LightningModule`. This probe takes as argument the backbone model that we previously trained, as well as its output dimension and the number of labels of our supervised task.

We use the same naming convention as earlier for the encoder, enabling us to load the pretrained encoder in the linear probe seamlessly.

Since we are in a multilabel scenario, the linear probe is followed by a sigmoid function that individually maps all outputs in $[0, 1]$. We then optimize the Binary Cross-Entropy between the predicted activation probabilities and our actual annotations.

We evaluate the performances of our model by measuring two metrics: **mean Average Precision** and **ROC-AUC**.

##### Mean Average Precision (mAP)

Mean Average Precision (mAP) is a performance metric used to evaluate the accuracy of a multilabel classification model. In this context, it measures how well the model ranks relevant labels higher than irrelevant ones across multiple classes. The mAP is the mean of the average precision scores for each label, calculated as follows:

1. **Precision and Recall:** For each label, calculate the precision and recall at each prediction threshold.
   - **Precision** is the ratio of true positive predictions to the total number of positive predictions.
   - **Recall** is the ratio of true positive predictions to the total number of actual positives.
   
2. **Average Precision (AP):** For each label, plot the precision-recall curve and calculate the area under this curve. This gives the AP for that label.

3. **Mean Average Precision (mAP):** Compute the mean of the AP values across all labels to obtain the mAP score.

The mAP provides a single score that summarizes the performance of the model across all labels, taking into account both the precision and recall.

##### ROC-AUC

The Receiver Operating Characteristic (ROC) curve and the Area Under the Curve (AUC) are used to evaluate the performance of a classification model. In the context of multilabel classification, the ROC-AUC is calculated for each label independently and then averaged.

1. **ROC Curve:** For each label, plot the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings.
   - **True Positive Rate (TPR)** is the ratio of true positives to the total number of actual positives.
   - **False Positive Rate (FPR)** is the ratio of false positives to the total number of actual negatives.

2. **AUC (Area Under the Curve):** The AUC represents the probability that the model ranks a randomly chosen positive instance higher than a randomly chosen negative instance. An AUC of 1 indicates perfect performance, while an AUC of 0.5 indicates performance no better than random guessing.

3. **Mean ROC-AUC:** Calculate the AUC for each label and then take the average of these AUC scores to obtain the mean ROC-AUC. This provides a single metric that reflects the model's overall ability to discriminate between the positive and negative classes across all labels.

For simplicity, we do not focus here on the specific implementation of these metrics and use the existing one from `scikit-learn`.



```python
class LinearProbe(pl.LightningModule):
    def __init__(self,
                 backbone: torch.nn.Module,
                 backbone_dim: int,
                 num_labels: int,
                 precomputed_embeddings: bool = False):
        super(LinearProbe, self).__init__()

        self.encoder = backbone
        self.probe = torch.nn.Linear(backbone_dim, num_labels)
        self.loss_fn = torch.nn.BCELoss()  # Use binary cross-entropy for multi-label classification
        self.precomputed_embeddings = precomputed_embeddings

        # During evaluation, freeze the backbone and discard the final projection layer
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Store predictions and labels for the entire epoch
        self.all_preds = []
        self.all_labels = []

    def forward(self, x):
        with torch.no_grad():
            z = self.encoder(x)

        return torch.sigmoid(self.probe(z))

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)

        loss = self.loss_fn(preds, y)
        self.log(f"train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        preds = self.forward(x)
        loss = self.loss_fn(preds, y)
        self.log("val_loss", loss)

        # Store predictions and labels for metric calculation at the end of the epoch
        self.all_preds.append(preds.cpu())
        self.all_labels.append(y.cpu())

        return preds, y

    def on_validation_epoch_end(self):
        # Clear the lists for the next epoch
        self.all_preds = []
        self.all_labels = []

    def calculate_metrics(self, preds, labels):
        # Convert to numpy arrays for metric calculation
        preds_np = preds.numpy()
        labels_np = labels.numpy()

        # Calculate mAP and ROC AUC
        mAP = average_precision_score(labels_np, preds_np, average='macro')  # mAP for multi-label
        roc_auc = roc_auc_score(labels_np, preds_np, average='macro')  # ROC AUC for multi-label

        # Log the metrics
        self.log("mAP", mAP, prog_bar=True)
        self.log("ROC-AUC", roc_auc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.probe.parameters(), lr=1e-3)
        return optimizer
```


```python
probe = LinearProbe(backbone=model.encoder, backbone_dim=512, num_labels=50)

# create the training and validation datasets
train_set = MultiLabelDataset("mtt_ssl/train", duration=25.)
val_set = MultiLabelDataset("mtt_ssl/test", duration=25.)
```



### 2.5. Downstream evaluation

Finally, let's train our **linear probe**.


```python
train_dataloader = torch.utils.data.DataLoader(train_set,
                                               batch_size=256,
                                               num_workers=os.cpu_count(),
                                               shuffle=True,
                                               pin_memory=True,
                                               persistent_workers=True)

val_dataloader = torch.utils.data.DataLoader(val_set,
                                             batch_size=256,
                                             num_workers=os.cpu_count(),
                                             shuffle=False,
                                             pin_memory=False,
                                             persistent_workers=True)

# callbacks
trainer = pl.Trainer(accelerator='gpu', max_epochs=500)

trainer.fit(probe, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
```

That's it! Once your big SSL model is trained, such a simple linear classifier and a few annotated samples are all you need!

## 3. Conclusion

In this tutorial, we saw how to both train and evaluate an SSL model based on Siamese networks. We focused in particular on contrastive learning, which is the most widely used technique, but of course there are many others.

Overall, the main directions of research are:
- *How to create positive pairs?* We can use transforms as in this notebook, however one can instead use masking, or even multimodal data (e.g. an audio and its description) to build multimodal latent spaces (see [CLIP](https://arxiv.org/abs/2103.00020), [CLAP](https://arxiv.org/abs/2206.04769), etc.)
- *How to prevent collapse?* In this notebook, we covered the contrastive loss but there are several other techniques, such as directly optimizing the batch statistics ([Wang et al.](https://arxiv.org/abs/2005.10242), [VICReg](https://arxiv.org/abs/2105.04906)...) or breaking the symmetry between the two branches of the Siamese network ([BYOL](https://arxiv.org/abs/2006.07733), etc.)

For going more into details, we strongly encourage the reader to check this great [Cookbook of Self-Supervised Learning](https://arxiv.org/abs/2304.12210).
