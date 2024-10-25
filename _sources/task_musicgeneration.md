# Musical Audio Generation

## Goal of the Task

Musical audio generation aims to create various musical content, from individual notes to full instrumental arrangements and complete songs. In the early days of audio generation research, methods often focused on producing audio directly in the time or time-frequency domain. Recent approaches, however, work with compressed representations, often using neural audio codecs.

The most widely used models today are autoregressive (Transformer) architectures and diffusion models. Autoregressive architectures are particularly effective for discrete codecs, while diffusion models are better suited for continuous representations.

## Popular Datasets

- **NSynth**: NSynth was once the go-to dataset for musical audio generation and can be regarded the "MNIST" in audio. It contains short, synthetic, single-note samples from different instrument families and detailed metadata, making it a valuable resource for early experiments.

- **GTZAN**: The GTZAN dataset is often used for genre classification and can serve as a starting point for more complex audio generation tasks involving diverse genres.

- **MusicNet**: Contains recordings of classical music with aligned annotations, suitable for tasks involving complex musical structures.

- **MAESTRO**: The MAESTRO dataset features piano performances, providing MIDI and corresponding audio recordings. This makes it particularly useful for training models of high-quality piano music generation.

- **MagnaTagATune**: Offers a large collection of music with tags, useful for genre classification and multi-label tasks.


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