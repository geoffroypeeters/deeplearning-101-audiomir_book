# Deep Learning Bricks

## Inputs

### Waveform

### Log-Mel-Spectrogram (LMS)

### Constant-Q-Transform (HCQT)

### Harmonic-CQT (HCQT)

### Parametric front-end: SincNet

### Parametric front-end: LEAF

### Audio augmentations



## Projections

### 2D-Conv

### 1D-Conv

### Dilated-Conv

### Dephtwise Separable Convolution

```python
class depthwise_conv(nn.Module):

    def __init__(self, nin, kernels_per_layer):
        super(depthwise_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, groups=nin)

    def forward(self, x):
        out = self.depthwise(x)
        return out

class pointwise_conv(nn.Module):

    def __init__(self, nin, nout):
        super(pointwise_conv, self).__init__()
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.pointwise(x)
        return out

class depthwise_separable_conv(nn.Module):

    def __init__(self, nin, kernels_per_layer, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

model = depthwise_separable_conv(16, 1, 32)
X = torch.randn(2,16,23,23)
model(X).size()
```

### Temporal Convolution Network (TCN)

### WaveNet


## Bottleneck

### Auto-encoder

### Variational auto-encoder (VAE)

### Vector Quantised-Variational AutoEncoder (VQ-VAE)

### Residual Vector Quantizers (RVQ)





## Architectures

### U-Net

### RNN/ LSTM

### Transformer/ Self-Attention

### Conformer



## Paradigms

### Supervised

### Self-supervised

### Metric Learning

### Adversarial

### Encoder-Decoder

### Diffusion
