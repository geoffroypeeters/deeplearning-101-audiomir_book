# Introduction

## Organisation of the book

The first part of the book, **Tasks** describe a subset of typical audio-based MIR tasks.
To facilitate the reading of the book, we follow a similar structure to describe each of the audio-based MIR task we consider.
We describe in turn:
- What is the **goal** of the task ?
- What are the perfomrance measures for the task, how is the task **evaluated** ?
- What are the popular **datasets** for the taslk (used to train system or evaluate the performances of)
- How we can solve it using **deep learning**. This part refers to bricks that are described individually in the second part of the book.

The second pat of the book, **Deep Learning Bricks**, described each brick individually.
We have chosen to separate the description of the bricks from the tasks in which they can be used to emphasise the fact that the same brick can be used for several tasks.
We want also to emphasise the fact that those are just bricks.

![concept1](/images/main_concept1.png)

## Simplifying the development

To make our life easier and **facilitate the reading of the code of the notebooks** we will rely on the following elements.
- for **datasets** (audio and annotations): .hdf5 (for audio) and .pyjama (for annotations), described below
- for **deep learning**: pytorch (a python library for deep learning)
  - for the dataset/dataloader
  - for the models, the losses, the optimizers
- for **training**: torchlighning (a library which stands on top of pytorch and which facilitates training and deployment of models)





### Datasets using  .hdf5 and .pyjama file

In the first part of this tutorial, each dataset will be saved as a pair of files: one in .hdf5 format for the audio and the other in .pyjama format for the annotations.

A single [.hdf5](https://docs.h5py.org/) file contains all the audio data of a dataset.
Each `key` corresponds to an entry.
An entry corresponds to a specific audiofile.
Its array contains the audio waveform.
Its attribute `sr_hz` provides the sampling rate of the audio waveform.

```python
with h5py.File(hdf5_audio_file, 'r') as hdf5_fid:
    audiofile_l = [key for key in hdf5_fid['/'].keys()]
    key = audiofile_l[0]
    pp.pprint(f"audio shape: {hdf5_fid[key][:].shape}")
    pp.pprint(f"audio sample-rate: {hdf5_fid[key].attrs['sr_hz']}")
```

A single [.pyjama](https://github.com/geoffroypeeters/pyjama) contains all the annotations of all the files of a dataset.
The values of the `filepath` field of the .pyjama file correspond to the `key` values of the .hdf5 file.

```python
with open(pyjama_annot_file, encoding = "utf-8") as json_fid:
    data_d = json.load(json_fid)
audiofile_l = [entry['filepath'][0]['value'] for entry in entry_l]
entry_l = data_d['collection']['entry']
pp.pprint(entry_l[0:2])
```

```python
{'collection': {'descriptiondefinition': {'album': ...,
                                          'artist': ...,
                                          'filepath': ...,
                                          'original_url': {...,
                                          'tag': ...,
                                          'title': ...,
                                          'pitchmidi': ...},
                'entry': [
													{
													 'album': [{'value': 'J.S. Bach - Cantatas Volume V'}],
                           'artist': [{'value': 'American Bach Soloists'}],
                           'filepath': [{'value': '0+++american_bach_soloists-j_s__bach__cantatas_volume_v-01-gleichwie_der_regen_und_schnee_vom_himmel_fallt_bwv_18_i_sinfonia-117-146.mp3'}],
                           'original_url': [{'value': 'http://he3.magnatune.com/all/01--Gleichwie%20der%20Regen%20und%20Schnee%20vom%20Himmel%20fallt%20BWV%2018_%20I%20Sinfonia--ABS.mp3'}],
                           'tag': [{'value': 'classical'}, {'value': 'violin'}],
                           'title': [{'value': 'Gleichwie der Regen und Schnee vom Himmel fallt BWV 18_ I Sinfonia'}],
                           },
                          {
                           'album': [{'value': 'J.S. Bach - Cantatas Volume V'}],
                           'artist': [{'value': 'American Bach Soloists'}],
                           'filepath': [{'value': '0+++american_bach_soloists-j_s__bach__cantatas_volume_v-09-weinen_klagen_sorgen_zagen_bwv_12_iv_aria__kreuz_und_krone_sind_verbunden-146-175.mp3'}],
                           'original_url': [{'value': 'http://he3.magnatune.com/all/09--Weinen%20Klagen%20Sorgen%20Zagen%20BWV%2012_%20IV%20Aria%20-%20Kreuz%20und%20Krone%20sind%20verbunden--ABS.mp3'}],
                           'tag': [{'value': 'classical'}, {'value': 'violin'}],
                           'title': [{'value': '-Weinen Klagen Sorgen Zagen BWV 12_ IV Aria - Kreuz und Krone sind verbunden-'}],
                           'pitchmidi': [
                             {
                               'value': 67,
                               'time': 0.500004,
                               'duration': 0.26785899999999996
                             },
                             {
                               'value': 71,
                               'time': 0.500004,
                               'duration': 0.26785899999999996
                             }],
                           }
                           ]},
 'schemaversion': 1.31}
 ```

Using those, a dataset is described by only two files: a .hdf5 for the audio, a .pyjama for the annotations.

We provide a set of datasets (each with its .hdf5 and .pyjama file) for this tutorial [here](https://perso.telecom-paristech.fr/gpeeters/tuto_DL101forMIR/].



### Pytorch dataset/dataloader

From a top-down approach, the central part of the training will consist on a loop over epochs and iteration over batches of data:
```
for n_epoch in range(epochs):
  for batch in train_dataloader:
    hat_y = my_model(batch['X'])
    loss = my_loss(hat_y, batch['y'])
    loss.backward()
    ...
```
In this, `train_dataloader` is an instance of the pytorch-class `Dataloader` which goal is to encapsulate `batch_size` times the outputs (the `X` and `y`) of `train_dataset`.
```
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
```
`train_dataset` is the one responsible for providing the `X`and the `y`.
It is an instance of a class written by the user (which inherits from the pytorch-class `Dataset`).
Writing this class is probably the most complex part.

![expe](/images/expe_dataset_P.png)

The design of this class involves defining what should the `__getitem__` return (the `X` and `y` for the model) and involves providing in the `__init__`  all the necessary information so that `__getitem__` can do its job.
- (1) It involves
  - defining what is the **input representation** of the model (`X` can be waveform, Log-Mel-Spectrogram, Harmonic-CQT),
  - defining **where to compute** those
    - compute those one-the-fly in the `__getitem__` ?
    - pre-computed those in the `__init__` and read them on-the-fly from drive/memory in the `__getime__` ?

In the first notebooks, we define a set of features in the `feature.py` package (`feature.f_get_waveform`, `feature.f_get_lms`, `feature.f_get_hcqt`).
We also define the output of `__getitem__`/`X` as a **patch** (a segment/chunk of a specific ime duration extracted from a tensor (Channel, Dimension/Frequency, Time)).
To define the patch, we do a frame analysis (with a specific window lenght and hope size) over the tensor.
We pre-compute the list of all possible patches for a given audio in the `feature.f_get_patches`.

- (2) It involves **mapping the annotations** contained in the .pyjama file (such as pitch, genre or work-id annotations) to the format of `hat_y` (scalar, matrix or one-hot-encoding) and to map it to the time position and extend of the patches `X`.

- (3) It involves **defining what is the unit of `idx`** in the `__getitem__(idx)`. It can refer to the patch number, the file number (in this case `X` can be all the patches a given file) or the work-id (in this case `X` provides the features of all the files with the same work-id).


### Pytorch models

Models in pytorch are usually written as classes which inherits from the pytorch-class `nn.Module`.
Such a class should have
- a `__init__` method defining the parameters (layers) to be trained and
- a `__forward__` method describing how to do the forward with the layers defined before (for example how to go from `X` to `hat_y`).

```
class NetModel(nn.Module):
    """
    Generic class for neural-network models based on the f_parse_component of .yaml file
    """
    def __init__(self, config, current_input_dim):
        super().__init__()
          self.layer_l = []
          self.layer_l.append(nn.Sequential(...))
          ...
          self.model = nn.ModuleList(self.layer_l)

    def forward(self, X, do_verbose=False):
        hat_y = self.model(X)
        return hat_y
```

In practice, it is common to specify the hyper-parameters of the model (such as number of layers, feature-maps, activations) in a dedicated `.yaml`.

In the first notebooks, we do a step forward here by defining the whole model in a `.yaml` file.
The model then becomes much more readable.
`model_factory.NetModel` is a generic class that allows dynamically creating model classes based on this `.yaml`  file (such as exemplified below).

```python
model:
    name: AutoTagging
    block_l:
    - sequential_l:
        - layer_l:
            - [LayerNorm, {'normalized_shape': [128, 64]}]
            - [Conv2d, {'in_channels': 1, 'out_channels': 80, 'kernel_size': [128, 5], 'stride': [1,1]}]
            - [Squeeze, {'dim': 2}]
        - layer_l:
            - [LayerNorm, {'normalized_shape': -1}]
            - [Activation, LeakyReLU]
            - [Dropout, {'p': 0}]
        - layer_l:
            - [Conv1d, {'in_channels': -1, 'out_channels': 60, 'kernel_size': 5, 'stride': 1}]
            - [MaxPool1d, {'kernel_size': 3, 'stride': 3}]
            - [LayerNorm, {'normalized_shape': -1}]
            - [Activation, LeakyReLU]
            - [Dropout, {'p': 0}]
        - layer_l:
            - ['Permute', {'shape': [0, 2, 1]}]
    - sequential_l:
        - layer_l:
            - [LayerNorm, {'normalized_shape': -1}]
            - [Linear, {'in_features': -1, 'out_features': 128}]
            - [BatchNorm1dT, {'num_features': -1}]
            - [Activation, LeakyReLU]
            - [Dropout, {'p': 0}]
        - layer_l:
            - [Linear, {'in_features': -1, 'out_features': 128}]
            - [BatchNorm1dT, {'num_features': -1}]
            - [Activation, LeakyReLU]
            - [Dropout, {'p': 0}]
        - layer_l:
            - ['Permute', {'shape':[0, 2, 1]}]
        - layer_l:
            - [Mean, {'dim': 2}]
    - sequential_l:
        - layer_l:
            - [Linear, {'in_features': -1, 'out_features': 50}]
            - [Activation, Softmax]
```


### TorchLightning training

[Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/) is a high-level wrapper for PyTorch that simplifies the process of organizing, training, and scaling deep learning models.
It structures PyTorch code with best practices, making it easier to implement, debug, and accelerate models across different hardware with minimal boilerplate code.
It allows to by-pass the tedious work of writing training and validation loop over epoch and over mini-batch.

The writing of the Lightning class is very standard and almost the same for all tasks.
It involves indicating
- which model, loss and optimizer to use
- what is a forward pass for training (`training_step`) and validation (`validation_step`)

```python
class AutoTaggingLigthing(pl.LightningModule):
    def __init__(self, in_model):
        super().__init__()
        self.model = in_model
    def training_step(self, batch, batch_idx):
        hat_y = self.model(batch['X'])
        loss = self.loss(hat_y, batch['y'])
        self.log("train_loss", loss, prog_bar=True)
        return loss
    def validation_step(self, batch, batch_idx):
        hat_y = self.model(batch['X'])
        loss = self.loss(hat_y, batch['y'])
        self.log('val_loss', loss, prog_bar=True)
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), 0.001)
        return optimizer
```

The training code is then extremely simple: `trainer.fit`.
It also allows to define **CallBack** using predefined methods such as for `EarlyStopping` or for saving `ModelCheckpoint`.

```python
my_lighting = AutoTaggingLigthing( model )

early_stop_callback = EarlyStopping(monitor="val_loss", patience=10, verbose=True, mode="min")
checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath=param_lightning.dirpath, filename=param_lightning.filename, save_top_k=1, mode='min')

trainer = pl.Trainer(accelerator="gpu",  max_epochs = param_lightning.max_epochs, callbacks = [early_stop_callback, checkpoint_callback])
trainer.fit(model=my_lighting, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
```

### Evaluation metrics

In the notebooks, we will rely most of the time on
- [`mir_eval`](https://github.com/craffel/mir_eval) which provides most MIR specific evaluation metrics,
- [`scikit-learn`](https://scikit-learn.org/) which provides the standard machine-learning evaluation metrics.


### In summary

![concept2](/images/main_concept2.png)
