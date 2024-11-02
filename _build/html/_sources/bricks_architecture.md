# Architectures

![top](/images/top.png)

We denote by `architecture` the overall design of a neural network, i.e. the way front-end and projections are used together.


(lab_unet)=
## U-Net

The U-Net was proposed in {cite}`DBLP:conf/miccai/RonnebergerFB15` in the framework of biomedical image segmentation and made popular in MIR by {cite}`DBLP:conf/ismir/JanssonHMBKW17` for singing voice separation.

The U-Net is an auto-encoder with skip-connections.
- The **encoder** (left part) downsample the spatial dimensions and increase the depth,
- The **decoder** (right part) upsample the spatial dimensions and decrease the depth.

**Skip connections** are added between equivalent layers of the encoder and decoder:
- example: the 256 channels level of the encoder is concatenated with the 256 level of the decoder to form a 512 tensor.
- The goal of the skip-connections are two-folds:
	- to bring back details of the original images to the decoder (the bottleneck being to compressed to represent detailed information)
	- to facilitate the backpropagation of the gradient.

The **upsampling** (decoder) part can be done either
- using Transposed Convolution (hence a well-known checkerboard artefact may appears)
- using Interpolation followed by Normal convolution

![brick_unet](/images/brick_unet.png)\
**Figure** U-Net architecture for biomedical image segmentation *image source: {cite}`DBLP:conf/miccai/RonnebergerFB15`*



## Many to One: reducing the time dimensions

They are many different ways to reduce a (temporal) sequence of embeddings $\{ \mathbf{x}_1, \ldots \mathbf{x}_{T_x}\}$ to a single embedding $\mathbf{x}$ (Many-to-One).

Such a mechanism can be necessary in order to map the temporal embedding provided by the last layer of a network to a single ground-truth (such as in auto-tagging, where the whole track is from a given genre, or in Acoustic Scene Classification).

The most simple way to achieve this is to use the Mean/Average value (Average Pooling) or Maximum value (Max Pooling) of the $\mathbf{x}_t$ over time $t$ (as done for example in {cite}`Dieleman2014Spotify`).

### Attention weighting

Another possibility is to compute a weighted sum of the values $\mathbf{x}_t$ where the weights $a_t$ are denoted by **attention** parameters:
$\mathbf{x} = \sum_{t=0}^{T_x-1} a_t \mathbf{x}_t$

In {cite}`DBLP:conf/ismir/GururaniSL19`, it is proposed to compute these weights $a_t$ either
1. by computing a new projection of the $\mathbf{x}_t$ and then normalize them:
	$a_t = \frac{\sigma(\mathbf{v}^T h(\mathbf{x}_t))}{\sum_{\tau} \sigma(\mathbf{v}^T h(\mathbf{x}_{\tau}))}$
	- with $h$ a learnable embedding, $\mathbf{v}$ the learned parameters of the attention layer
2. doing the same after splitting $\mathbf{x}_t$ in two (along the channel dimensions): the first part being used to compute "values", the second to compute "weights"

![brick_attention_instrument](/images/brick_attention_instrument.png)\
**Figure** Attention weighting, *image source: {cite}`DBLP:conf/ismir/GururaniSL19`*

```python
class nnSoftmaxWeight(nn.Module):
    """
    Perform attention weighing based on softmax with channel splitting
    Code from https://github.com/furkanyesiler/move
    """
    def __init__(self, nb_channel):
        super().__init__()
        self.nb_channel = nb_channel
    def forward(self, X):
        weights = torch.nn.functional.softmax(X[:, int(self.nb_channel/2):], dim=3)
        X = torch.sum(X[:, :int(self.nb_channel/2)] * weights, dim=3, keepdim=True)
        return X
```

(lab_AutoPoolWeightSplit)=
### Auto-Pool
The above attention mechanism can by combined with the auto-pool operators proposed by {cite}`DBLP:journals/taslp/McFeeSB18`.

The auto-pool operators is defined as

$$\tilde{\mathbf{x}}_t = \frac{\exp(\alpha \cdot \mathbf{x}_t)}{\sum_{\tau} \exp(\alpha \cdot \mathbf{x}_{\tau})}$$

It uses a parameter $\alpha$ which allows to range from
- $\alpha=0$ (unweighted, a.k.a. average pooling),
- $\alpha=1$ (softmax weighted mean),
- $\alpha=\infty$: (a.k.a. max pooling).

The $\alpha$ parameters is a trainable parameters (optimized using SGD).

![brick_autopool](/images/brick_autopool.png)\
**Figure** Auto-pool operator *image source: {cite}`DBLP:journals/taslp/McFeeSB18`*

```python
# Code: https://github.com/furkanyesiler/move
def f_autopool_weights(data, autopool_param):
    """
    Calculating the autopool weights for a given tensor
    :param data: tensor for calculating the softmax weights with autopool
    :return: softmax weights with autopool

    see https://arxiv.org/pdf/1804.10070
    alpha=0: unweighted mean
    alpha=1: softmax
    alpha=inf: max-pooling
    """
    # --- x: (batch, 256, 1, T)
    x = data * autopool_param
    # --- max_values: (batch, 256, 1, 1)
    max_values = torch.max(x, dim=3, keepdim=True).values
    # --- softmax (batch, 256, 1, T)
    softmax = torch.exp(x - max_values)
    # --- weights (batch, 256, 1, T)
    weights = softmax / torch.sum(softmax, dim=3, keepdim=True)
    return weights
```

### Using models

It is also possible to use a **RNN/LSTM in Many-to-One configuration** (only the last hidden state $\mathbf{x}_{T_x}$ is mapped to an output $\hat{y}$).

Finally it is possible to add an extra CLASS token to a Transformer architecture.

![brick_pooling](/images/brick_pooling_P.png)



It should be noted that the term "Attention" encapsulates a large set of different paradigms.
- In the **encode-decoder architecture** {cite}`DBLP:journals/corr/BahdanauCB14` it is used during decoding to define the correct context $\mathbf{c}_{\tau}$ to be used to generate the hidden state $\mathbf{s}_{\tau}$.
For this it compares the decoder hidden state $\mathbf{s}_{\tau-1}$ to all the encoder hidden states $\mathbf{a}_t$.
- In the **transformer architecture** {cite}`DBLP:conf/nips/VaswaniSPUJGKP17` it is used to compute a self-attention.
For this, the $\mathbf{x}_t$ are mapped (using matrix projections) to query $\mathbf{q}_t$, key $\mathbf{k}_t$ and values $\mathbf{v}_t$.
A given $\mathbf{q}_{\tau}$ is then compared to all $\mathbf{k}_t$ to compute attention weights $\mathbf{a}_{t,\tau}$ which are used in the weighting sum of the $\mathbf{v}_t$:
$\mathbf{e}_{\tau} = \sum_t \mathbf{a}_{t,\tau} \mathbf{v}_{t}$.


## Recurrent Architectures

(lab_rnn)=
### RNN

**Recurrent Neural Networks (RNNs)** are a type of neural network designed to work with <mark>sequential data</mark> (e.g., time series, text, etc.).
They "remember" information from previous inputs by using hidden states, which allows them to model dependencies across time steps.

Their generic formulation for inputs $\mathbf{x}_t$ over time is:

$$\mathbf{h}_t = \tanh (\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{hx} \mathbf{x}_t+ \mathbf{b}_h)$$

where $\mathbf{h}_t$ is the hidden state of the RNN at time $t$.

A <mark>bi-directional-RNN</mark>, read the data in both directions (left-to-right and right-to-left).
The goal is to make $\mathbf{h}_t$ both dependent on $\mathbf{h}_{t-1}$ and $\mathbf{h}_{t+1}$.

Two configurations are often used with RNNs:
- <mark>Many-to-Many</mark>: RNN can be used to model the evaluation over time of features (such as done in the past with Kalman filters or HMM).
They are often used to represent a Language model.
- <mark>Many-to-One</mark>: One can also use the last hidden state of a RNN $\mathbf{h}_{T_x}$ where $T_x$ is the length of the input sequence, to sum up the content of the input sequence (see picture below).

![brick_rnn](/images/brick_rnn.png)\
**Figure** RNN in Many-to-Many and Many-to-One configurations *image source: [Link](https://www.researchgate.net/figure/The-four-types-of-recurrent-neural-network-architectures-a-univariate-many-to-one_fig3_317192370)*



```python
torch.nn.RNN(input_size, hidden_size, num_layers=1, bidirectional=False)
```


### LSTM
**Long Short-Term Memory (LSTM)**  are a specialized type of RNN designed to handle long-term dependencies more effectively.
LSTM use a more complex architecture with
- a memory $\mathbf{c}_t$ over time $t$,
- a hidden value $\mathbf{h}_t$ and
- a set of gates (input gate, forget gate, and output gate)
	- they allow to control the flow of information between the input $\mathbf{x}_t$, the previous hidden state $\mathbf{h}_{t-1}$ and memory $\mathbf{c}_{t-1}$ and their new values.

This allows them to retain relevant information over longer sequences while "forgetting" irrelevant information.

As RNN, two configurations are often used with LSTMs:
- <mark>Many-to-Many</mark>
- <mark>Many-to-One</mark>

![brick_rnn](/images/brick_lstm.png)\
**Figure** Details of a LSTM cell  *image source: [Link](https://mlarchive.com/deep-learning/understanding-long-short-term-memory-networks/)*

```python
torch.nn.LSTM(input_size, hidden_size, num_layers=1, bidirectional=False)
```

### Transformer/ Self-Attention
