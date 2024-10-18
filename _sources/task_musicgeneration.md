# Musical Audio Generation

## Goal of the Task

Musical audio generation aims to create a wide range of musical content, from individual notes to full instrumental arrangements and complete songs. In the early days of audio generation research, methods often focused on producing audio directly in the time or time-frequency domain. Recent approaches, however, work with compressed representations, often using neural audio codecs.

The most widely used models today are autoregressive (Transformer) architectures and diffusion models. Autoregressive architectures are particularly effective for discrete codecs, while diffusion models are better suited for continuous representations.

## Popular Datasets

- **NSynth**: NSynth was once the go-to dataset for musical audio generation. It contains short, synthetic, single-note samples from different instrument families, along with detailed metadata, making it a valuable resource for early experiments.

- **GTZAN**: The GTZAN dataset is often used for genre classification and can serve as a starting point for more complex audio generation tasks involving diverse genres.

- **MAESTRO**: The MAESTRO dataset features piano performances, providing both MIDI and corresponding audio recordings. This makes it particularly useful for training models focused on high-quality piano music generation.

## Solving Musical Audio Generation Using Deep Learning

### Previous Approaches

Before the rise of Transformers and diffusion models, models like Recurrent Neural Networks (RNNs) and Generative Adversarial Networks (GANs) were commonly used for musical audio generation. RNNs could handle sequences but often struggled with long-term dependencies, leading to repetitive or incoherent results. GANs were used to generate audio but faced challenges with training instability and producing high-quality, diverse outputs. These limitations led to a shift towards more robust architectures like Transformers and diffusion models, which can better capture the complexity of musical content.

### Autoregressive (Transformer) Architectures

Autoregressive models, especially those based on Transformers, are well-suited for generating sequences like musical audio. These models generate audio by predicting each subsequent token based on prior ones, capturing long-term relationships effectively, which helps in producing coherent compositions.

### Diffusion Models

Diffusion models offer another approach to musical audio generation. They transform random noise into meaningful continuous audio representations.


## How is the Task Evaluated?

Evaluation of generation tasks is difficult. In other ML tasks, specific targets (e.g., labels, data points) are available in a given evaluation set, allowing for the estimation of precision for a given model. In contrast, in audio generation, the goal is to sample from the distribution of the training set, without directly reproducing any training data.

As a result, indirect, distribution-based evaluation metrics are commonly used, rather than relying on one-to-one comparisons like in autoencoders or classification tasks.

### Frechet Audio Distance (FAD)

Nowadays, the most commonly used metric in audio generation is the Frechet Audio Distance (FAD). FAD compares the statistics of generated audio to those of real audio using embeddings from a pre-trained model. This metric provides a measure of how close the generated audio is to the original data distribution, which helps in assessing the quality and diversity of generated samples.

### Subjective Evaluation

Objective evaluation metrics cannot capture all details people care about when listening to audio. Therefore, it is very common (and important) in works on audio generation to perform user studies. 

Lickert Scale
