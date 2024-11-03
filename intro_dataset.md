### Datasets .hdf5/.pyjama

In the first part of this tutorial, each dataset will be saved as a pair of files:
- one in .hdf5 format for the audio and
- one in .pyjama format for the annotations.


<hr style="border: 2px solid red; margin: 60px 0;">


A single [.hdf5](https://docs.h5py.org/) file contains all the audio data of a dataset.
Each `key` corresponds to an entry.
An entry corresponds to a specific audiofile.
- Its array contains the audio waveform.
- Its attribute `sr_hz` provides the sampling rate of the audio waveform.

```python
with h5py.File(hdf5_audio_file, 'r') as hdf5_fid:
    audiofile_l = [key for key in hdf5_fid['/'].keys()]
    key = audiofile_l[0]
    pp.pprint(f"audio shape: {hdf5_fid[key][:].shape}")
    pp.pprint(f"audio sample-rate: {hdf5_fid[key].attrs['sr_hz']}")
```


<hr style="border: 2px solid red; margin: 60px 0;">


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


<hr style="border: 2px solid red; margin: 60px 0;">


Using those, a dataset is described by only two files: a .hdf5 for the audio, a .pyjama for the annotations.

We provide a set of datasets (each with its .hdf5 and .pyjama file) for this tutorial [here](https://perso.telecom-paristech.fr/gpeeters/tuto_DL101forMIR/).

```python
Index of /gpeeters/tuto_DL101forMIR
[ICO]	Name	Last modified	Size	Description
[PARENTDIR]	Parent Directory	 	-	 
[   ] bach10.pyjama                   2024-10-19 12:21	19M	 
[   ] bach10_audio.hdf5.zip           2024-10-02 07:51	129M	 
[   ] cover1000.pyjama                2024-10-19 12:21	1.0M	 
[   ] cover1000_feat.hdf5.zip         2024-10-02 07:52	101M	 
[   ] datacos-benchmark.pyjama        2024-10-19 12:21	6.3M	 
[   ] datacos-benchmark_feat.hdf5.zip 2024-10-14 12:31	1.5G	 
[   ] gtzan-genre.pyjama              2024-10-19 12:21	306K	 
[   ] gtzan-genre_audio.hdf5.zip      2024-10-02 09:59	1.5G	 
[   ] maps.pyjama                     2024-10-19 12:21	51M	 
[   ] maps_audio.hdf5.zip             2024-10-14 12:12	2.3G	 
[   ] mtt.pyjama                      2024-10-19 12:21	1.7M	 
[   ] mtt_audio.hdf5.zip              2024-10-14 12:15	2.3G	 
[   ] rwc-pop_chord.pyjama            2024-10-22 12:23	10M	 
[   ] rwc-pop_chord_audio.hdf5.zip    2024-10-22 12:25	1.8G	 
```
