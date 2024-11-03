### Pytorch dataset/dataloader

To help understanding what a dataloader should provide, we use a <mark>top-down approach</mark>.\
We start from the central part of the training of a deep-learning model.\
It consists in a loop over `epochs` and for each an iteration over `batches` of data:
```
for n_epoch in range(epochs):
  for batch in train_dataloader:
    hat_y = my_model(batch['X'])
    loss = my_loss(hat_y, batch['y'])
    loss.backward()
    ...
```
In this, `train_dataloader` is an instance of the pytorch-class `Dataloader` which goal is to encapsulate a set of `batch_size` paris of input `X`/output `y` which are each provided by `train_dataset`.
```
train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                batch_size = batch_size,
                                                shuffle = True)
```

`train_dataset` is the one responsible for providing the `X` and the `y`.\
It is an instance of a class written by the user (which inherits from the pytorch-class `Dataset`).\



<hr style="border: 2px solid red; margin: 60px 0;">


Writing this class is probably the most complex part.\
It involves
- defining what should the `__getitem__` return (the `X` and `y` for the model) and
- provides in the `__init__`  all the necessary information so that `__getitem__` can do its job.

![expe](/images/expe_dataset_P.png)


1. It involves defining `X`
  - **what is the input representation** of the model ? (`X` can be waveform, Log-Mel-Spectrogram, Harmonic-CQT),
  - **where to compute it**
    - compute those one-the-fly in the `__getitem__` ?
    - pre-compute those in the `__init__` and read them on-the-fly from drive/memory in the `__getitem__` ?

In the first notebooks, we define a set of features in the `feature.py` package (`feature.f_get_waveform`, `feature.f_get_lms`, `feature.f_get_hcqt`).

We also define the output of `__getitem__`/`X` as a **patch** (a segment/achunk) of a specific time duration.\
The patches are extracted from the features which are represented as a tensor (Channel, Dimension/Frequency, Time).\
In the case of `waveform` the tensor is (1,Time), of `LMS` it is (1,128,Time), of `H-CQT` it is (6,92,Time).\
To define the patch, we do a **frame analysis** (with a specific window lenght and hope size) over the tensor.\
We pre-compute the list of all possible patches for a given audio in the `feature.f_get_patches`.

2. It involves **mapping the annotations** contained in the .pyjama file (such as pitch, genre or work-id   annotations)
    - to the format of the output of the pytorch-model, `hat_y`, (scalar, matrix or one-hot-encoding) and
    - to the time position and extent of the patches `X`.

3. It involves **defining what is the unit of `idx`** in the `__getitem__(idx)`. \
For example it can refer to
    - a patch number,
    - a file number (in this case `X` represents all the patches a given file) or
    - a work-id (in this case `X` provides the features of all the files with the same work-id).


<hr style="border: 2px solid red; margin: 60px 0;">


### Pytorch models

Models in pytorch are usually written as classes which inherits from the pytorch-class `nn.Module`.\
Such a class should have
- a `__init__` method defining the parameters (layers) to be trained and
- a `__forward__` method describing how to do the forward with the layers defined in `__init__` \
(for example how to go from `X` to `hat_y`).

```
class NetModel(nn.Module):

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


<hr style="border: 2px solid red; margin: 60px 0;">


In practice, it is common to <mark>specify the hyper-parameters</mark> of the model (such as number of layers, feature-maps, activations) in a dedicated `.yaml`.

In the first notebooks, we go one step further and <mark>specify the entire model</mark> in a `.yaml` file.
  - The code then becomes much more readable.
  - The class `model_factory.NetModel` allows to dynamically create model classes by parsing this file

Below is an example of such a `.yaml` file.

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
