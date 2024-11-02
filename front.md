# Deep Learning 101 for Audio-based MIR

This is a web book written for a tutorial session of the [25th International Society for Music Information Retrieval Conference](https://ismir2024.ismir.net/), Nov 10-14, 2024 in San Francisco, CA, USA. The [ISMIR conference](https://ismir.net/) is the world’s leading research forum on processing, searching, organising and accessing music-related data.

**Context.** Audio-based MIR (MIR based on the processing of audio signals) covers a broad range of tasks, including
- music content analysis (pitch, chord, beats, tagging), similarity (acoustic or cover identification), fingerprint,
- music audio processing (source separation, music translation)
- music audio generation (of samples or whole tracks).

A wide range of techniques can be employed for solving each of these tasks, spanning from conventional signal processing and machine learning algorithms to the whole zoo of deep learning techniques.

**Objective.** This tutorial aims to review the various elements of this deep learning zoo which are commonly applied in Audio-based MIR tasks.
We review typical
- audio front-ends (such as [waveform](lab_waveform), [Log-Mel-Spectrogram](lab_lms), [HCQT](lab_hcqt), [SincNet](lab_sincnet), quantization using VQ-VAE, RVQ),
- projections (including [1D-Conv](lab_conv1d), [2D-Conv](lab_conv2d), [Dilated-Conv](lab_dilated), [TCN](lab_tcn), [RNN](lab_rnn), Transformer, [U-Net](lab_unet), VAE),
- examine the various training paradigms (such as [supervised](lab_supervised), self-supervised, [metric-learning](lab_metric_learning), adversarial, encoder-decoder, diffusion).

**Method.** Rather than providing an exhaustive list of all of these elements, we illustrate their use within a subset of (commonly studied) Audio-based MIR tasks such as
- [multi-pitch](lab_multi_pitch), [cover-detection](lab_cover_detection), [auto-tagging](lab_auto_tagging),
- source separation
- music generation.

This subset of Audio-based MIR tasks is designed to encompass a wide range of deep learning elements.

*The objective is to provide a 101 lecture (introductory lecture) on deep learning techniques for Audio-based MIR. It does not aim at being exhaustive in terms of Audio-based MIR tasks neither on deep learning techniques but to provide an overview for newcomers to Audio-Based MIR on how to solve the most common tasks using deep learning. It will provide a portfolio of codes (Colab notebooks and Jupyter book) to help newcomers achieve the various Audio-based MIR Tasks.*


*This tutorial can be considered  as a follow-up of the tutorial ["Deep Learning for MIR" ](https://github.com/keunwoochoi/dl4mir) by Alexander Schindler, Thomas Lidy and Sebastian Böck, held at ISMIR-2018.*

## Citing this book

```
@book{deeplearning-101-audiomir:book,
	author = {Peeters, Geoffroy and Meseguer-Brocal, Gabriel and Riou, Alain and Lattner, Stefan},
	title = {Deep Learning 101 for Audio-based MIR, ISMIR 2024 Tutorial},
	url = {https://geoffroypeeters.github.io/deeplearning-101-audiomir_book},
	address = {San Francisco, USA},
	year = {2024},
	month = November,
	doi = {???},
```
