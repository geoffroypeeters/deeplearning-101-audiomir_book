# Musical Audio Generation

## Goal of the Task

Musical audio generation aims to create various musical content, from individual notes to full instrumental arrangements and complete songs. In the early days of audio generation research, methods often focused on producing audio directly in the time or time-frequency domain. Recent approaches, however, work with compressed representations, often using neural audio codecs.

The most widely used models today are autoregressive (Transformer) architectures and diffusion models. Autoregressive architectures are particularly effective for discrete codecs, while diffusion models are better suited for continuous representations.

## Popular Datasets

- **NSynth**: NSynth was once the go-to dataset for musical audio generation and can be regarded the "MNIST" in audio. It contains short, synthetic, single-note samples from different instrument families and detailed metadata, making it a valuable resource for early experiments.

- **GTZAN (delete?)**: The GTZAN dataset is often used for genre classification and can serve as a starting point for more complex audio generation tasks involving diverse genres.

- **MusicNet**: Contains recordings of classical music with aligned annotations, suitable for tasks involving complex musical structures.

- **MAESTRO**: The MAESTRO dataset features piano performances, providing MIDI and corresponding audio recordings. This makes it particularly useful for training models of high-quality piano music generation.

- **MagnaTagATune**: Offers a large collection of music with tags, useful for genre classification and multi-label tasks.

## Neural Musical Audio Synthesis

### Discrete vs. Continuous Generation

In generative tasks it is necessary to inject *some form of stochasticity* into the generation process.
In this regard, two general approaches can be distinguished: Generation of discrete sequences and generation of continuous-valued data.
For **discrete sequences**, models (RNNs, Causal Convolutional Networks, Transformers) are typically trained with cross-entropy loss to output a probability distribution over discrete random variables in a deterministic manner.
The stochasticity is then "injected" by sampling from that distribution. Note that in this case, we can primarily deal with one-hot encoded sequences, selecting one token per time step, as we don't have a simple way to sample N-hot vectors (where N > 1 tokens are selected simultaneously) from the model's output distribution.

In contrast, for generating **continuous-valued** data, the stochasticity usually comes from any form of noise injection into the neural network.
For example, Generative Adversarial Networks in their basic form inject noise by inputting a high-dimensional noise vector (sampled from an independent Gaussian distribution) into the generator.
That way, the task can be described as transforming an independent Gaussian distribution into the data distribution.

Similarly, in Variational Autoencoders (VAEs), the decoder receives as input a sample from an independent Gaussian prior.
The model is trained so that the encoder learns to compose this Gaussian prior by a mixture of Gaussians, each corresponding to a data point (posterior). 
The posterior for a specific data point is inferred by encoding it into a corresponding mean vector and a variance vector.

In Diffusion Models, the noise input has the same dimensionality as the data point that should be generated.
Other than that, we again sample from an independent Gaussian and transform this distribution into the data distribution.


### Early Works

Before the rise of Transformers and diffusion models, models like Causal Convolutional Networks, Recurrent Neural Networks (RNNs), and Generative Adversarial Networks (GANs) were used for musical audio generation.
At the time, it was common to generate in a low-level representation space, either directly in the signal domain (WaveNet, SampleRNN) or in the spectral domain (GANs).
Not least, due to their generation in such a high-dimensional space, CNNs/RNNs struggled with long-term dependencies, leading to repetitive or incoherent results without higher-level structure.  
GANs were used to generate audio in the signal or frequency domain but faced challenges with training instability and producing high-quality, diverse outputs.
Through the usage of neural audio codecs and the resulting reduction in dimensionality, the problem became simpler.
Nowadays, through a combination of more efficient/simpler to-train generative models with generation in a compressed space, it is possible to generate high-quality, full-length music tracks.

#### WaveNet


![wavenet_fig](./images/wavenet.png)

**Figure 1:** WaveNet architecture showing causal, dilated convolutions (image source: {cite}`DBLP:conf/ssw/OordDZSVGKSK16`).

WaveNet {cite}`DBLP:conf/ssw/OordDZSVGKSK16` can be seen as the first successful attempt to directly generate audio using a Neural Network.
Important components in WaveNet are dilated convolutions {cite}`DBLP:journals/corr/YuK15` that enable an exponentially growing receptive field with linearly increasing numbers of layers.
A big receptive field is critical in WaveNet because it operates directly in the signal domain with 16k samples/second.
In addition, causal convolutions are used to prevent the model from looking into the future during training, resulting in a generative autoregressive sequence model.

Autoregressive sequence models are typically trained with cross-entropy loss that requires one-hot encoded sequences.
As raw audio is usually 16-bit, a naive transformation into one-hot vectors would result in 65,536 dimensions per time step.
To keep the problem tractable, in WaveNet, each sample is non-linearly scaled and quantized to obtain 256-dimensional vectors.
The non-linear scaling function (ITU-T, 1988) is defined as

$$
f(x_t) = \text{sign}(x_t) \frac{\ln(1 + \mu |x_t|)}{\ln(1 + \mu)}.
$$

![wavenet_scaling](./images/wavenet_non-linearity.png)

**Figure 2:** Non-linear scaling of audio samples in WaveNet for $\mu = 255$ (in practice, $-1 < x_t < 1$).

*Usage Example*: WaveNet was used by Google for text-to-speech (TTS) applications.

#### SampleRNN

![sample_rnn](./images/sample_rnn.png)
**Figure 3:** Snapshot of the unrolled SampleRNN model at timestep $i$ with 3 tiers. As a simplification, only one RNN and up-sampling ratio $r = 4$ is used for all tiers (image source: {cite}`DBLP:conf/iclr/MehriKGKJSCB17`).

SampleRNN {cite}`DBLP:conf/iclr/MehriKGKJSCB17` was the first RNN-based neural audio synthesizer that had an impact in the community.
It can effectively learn to generate long-form audio at a sample rate of 16kHz.
While **WaveNet** builds hierarchical representations of audio by its built-in sub-sampling through dilated convolutional layers, 
SampleRNN builds such a hierarchy through multiple tiers of RNNs that operate in different "clock rates".
This approach enables representations at varying temporal resolutions, where lower tiers (faster rates) are conditioned on higher tiers. 
This encourages higher tiers to generate higher-level signal representations that help predict lower-level details.

Similarly to WaveNet, in order to keep the task tractable, the sample values are quantized to 256 bins—but without prior, non-linear scaling.
As the memory of RNNs can be updated iteratively without the need to reconsider past inputs, they tend to need less compute at inference time than non-recurrent autoregressive models (like causal convolutions or transformers).

*Usage Example*: Different artists used SampleRNN for music generation. Notably, a [livestream](https://www.youtube.com/watch?v=JF2p0Hlg_5U&ab_channel=DADABOTS) (by Dadabots) with Technical Death Metal music is ongoing with hardly any interruptions since March 2019 {cite}`DBLP:journals/corr/abs-1811-06633`.       

#### Generative Adversarial Networks

For several years, Generative Adversarial Networks (GANs) {cite}`DBLP:journals/corr/GoodfellowPMXWOCB14` were among the most influential generative models.
Their ability to implicitly model multi-dimensional *continuous-valued* distributions made them a compelling tool for image and audio generation.
This enabled the use of spectrogram (or spectrogram-like) representations in audio generation, which is a natural modality for 2D convolutional networks.
Another motivation for using image-like spectrogram representations with GANs for audio generation was the ability to leverage insights from the broader image-processing community. 


![gan_synth](./images/gansynth.png)
**Figure 4:** GANSynth rainbowgrams to showcase the influence of different audio representations (image source: {cite}`DBLP:conf/iclr/EngelACGDR19`).

While WaveGAN {cite}`DBLP:conf/iclr/DonahueMP19` was an influential work on using GANs directly for raw musical audio waveform generation, most works focussed on spectrogram-like representations.
Examples for that are GANSynth {cite}`DBLP:conf/iclr/EngelACGDR19`, SpecGAN {cite}`DBLP:conf/iclr/DonahueMP19`, DrumGAN {cite}`DBLP:conf/ismir/NistalLR20, DBLP:journals/corr/abs-2206-14723`, and DarkGAN {cite}`DBLP:conf/ismir/NistalLR21`, omitting those only applied to speech.
For simplicity reasons, the listed works can generate fixed-size outputs only. Some (later) examples of variable-size musical audio generation using GANs are VQCPC-GAN {cite}`DBLP:conf/waspaa/NistalALR21` and Musica! {cite}`DBLP:conf/ismir/PasiniS22`.    

Presently, GANs are widely replaced by Diffusion Models, which are more stable in training, less prone to mode-collapse, and have a simpler architecture, resulting in higher-quality outputs.  
A concept of GANs that could remain in the mid-term is the usage of adversarial losses from auxiliary networks, for example, for domain confusion or as additional loss in reconstruction-based training (e.g., in neural audio codecs, like DAC {cite}`DBLP:conf/nips/KumarSLKK23`). 

*Usage Example*: DrumGAN was the first commercially available neural audio synthesizer for music integrated into [Steinberg's Backbone](https://www.steinberg.net/vst-instruments/backbone/). It is now available for free as an [online app](https://drumgan.csl.sony.fr/).

### Autoregressive (Transformer) Architectures

Autoregressive models, especially those based on Transformers, are well-suited for generating sequences like musical audio. These models generate audio by predicting each subsequent token based on prior ones, effectively capturing long-term relationships, which helps produce coherent compositions.

### Diffusion Models

Diffusion models offer another approach to musical audio generation. They transform random noise into meaningful continuous audio representations.


## How is the Task Evaluated?

Evaluation of generation tasks is difficult. In other ML tasks, specific targets (e.g., labels, data points) are available in a given evaluation set, allowing precision estimation for a given model. In contrast, in audio generation, the goal is to sample from the distribution of the training set without directly reproducing any training data.

As a result, indirect, distribution-based evaluation metrics are commonly used rather than relying on one-to-one comparisons, as in autoencoders or classification tasks.

### Frechet Audio Distance (FAD)

Nowadays, the most commonly used metric in audio generation is the Frechet Audio Distance (FAD). FAD compares the statistics of generated audio to those of real audio using embeddings from a pre-trained model. This metric measures how close the generated audio is to the original data distribution, which helps assess the quality and diversity of generated samples.

### Subjective Evaluation

Objective evaluation metrics cannot capture all the details people care about when listening to audio. Therefore, it is very common (and important) in audio generation works to perform user studies.
In user studies, participants might be asked to rate the quality of generated audio samples on a Likert Scale ranging from 1 (very poor) to 5 (excellent). This helps quantify subjective perceptions of audio quality, coherence, and musicality.

### Likert Scale

...