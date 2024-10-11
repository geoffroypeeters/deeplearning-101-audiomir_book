# Deep Learning 101 for Audio-based MIR

This is a web book written for a tutorial session of the [25th International Society for Music Information Retrieval Conference](https://ismir2024.ismir.net/), Nov 10-14, 2024 in San Francisco, CA, USA. The [ISMIR conference](https://ismir.net/) is the world’s leading research forum on processing, searching, organising and accessing music-related data.

## Scope

*Audio-based MIR (MIR based on the processing of audio signals) covers a broad range of tasks, including analysis (pitch, chord, beats, tagging), similarity/cover identification, and processing/generation of samples or music fragments. A wide range of techniques can be employed for solving each of these tasks, spanning from conventional signal processing and machine learning algorithms to the whole zoo of deep learning techniques.*

*This tutorial aims to review the various elements of this deep learning zoo commonly applied in Audio-based MIR tasks. We review typical audio front-ends (such as waveform, Log-Mel-Spectrogram, HCQT, SincNet, quantization using VQ-VAE, RVQ), as well as projections (including 1D-Conv, 2D-Conv, Dilated-Conv, TCN, RNN, Transformer, U-Net, VAE), and examine the various training paradigms (such as supervised, self-supervised, metric-learning, adversarial, encoder-decoder, diffusion).*

*Rather than providing an exhaustive list of all of these elements, we illustrate their use within a subset of (commonly studied) Audio-based MIR tasks such as multi-pitch, cover-detection, auto-tagging, source separation, music-translation or music generation. This subset of Audio-based MIR tasks is designed to encompass a wide range of deep learning elements.*

*For each tack we address a) the goal of the tasks, b) how it is evaluated, c) provide some popular datasets to train a system, and d) explain (using slides and pytorch code) how we can solve it using deep learning.*

*The objective is to provide a 101 lecture (introductory lecture) on deep learning techniques for Audio-based MIR. It does not aim at being exhaustive in terms of Audio-based MIR tasks neither on deep learning techniques but to provide an overview for newcomers to Audio-Based MIR on how to solve the most common tasks using deep learning. It will provide a portfolio of codes (Colab notebooks and Jupyter book) to help newcomers achieve the various Audio-based MIR Tasks.*





## Citing this book¶

```
@book{deeplearning-101-audiomir:book,
    Author = {Peeters, Geoffroy and Meseguer-Brocal, Gabriel and Riou, Alain and Lattner, Stefan},
    Month = Nov.,
    Publisher = {https://geoffroypeeters.github.io/deeplearning-101-audiomir_book},
    Title = {Deep Learning 101 for Audio-based MIR},
    Year = 2024,
    Url = {https://geoffroypeeters.github.io/deeplearning-101-audiomir_book},
    doi = {???}
}
```
