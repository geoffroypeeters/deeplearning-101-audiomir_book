# Tools to make life easier

## Dataset

To make life easier we will store all datasets in the form of
- a *.pyjama file [DOC](https://github.com/geoffroypeeters/pyjama) for the annotations of all items of a given dataset
- a $.hdf5 file for the audio of all items of a given dataset

Example of pyjama file https://github.com/geoffroypeeters/audiodataset/blob/master/rwc-pop-structure.pyjama

### Creating a pytorch dataset class

We use simple pytorch Dataset classes that allow to access patch/chunk/slices of audio and store them in CPU/GPU memory (only works for small amount of data).
The `__getitem__` method allows to give the `X` and `y` for training/ validation/ testing.

```python
def f_get_patches(nb_frame):
    """
    description
    """
    patch_d.start_frame = 0
    patch_l = []
    while patch_d.start_frame+patch_d.L_frame < nb_frame:
        patch_d.end_frame = patch_d.start_frame+patch_d.L_frame
        patch_l.append({'start_frame': patch_d.start_frame, 'end_frame': patch_d.end_frame})
        patch_d.start_frame += patch_d.STEP_frame
    return patch_l

class WaveformDataset(Dataset):
    """
    description
    """

    def __init__(self, hdf5_audio_file, pyjama_annot_file, do_train):


        print(f'{hdf5_audio_file} {pyjama_annot_file}')
        with open(pyjama_annot_file, encoding = "utf-8") as json_fid:
            pyjama_d = json.load(json_fid)
        self.audiofile_l = [entry['filepath'][0]['value'] for entry in pyjama_d['collection']['entry']]
        self.genre_l = [entry['genre'][0]['value'] for entry in pyjama_d['collection']['entry']]

        # --- SPLIT TRAIN/TEST
        self.do_train = do_train
        if self.do_train:   self.audiofile_l = [self.audiofile_l[idx] for idx in range(len(self.audiofile_l)) if (idx % 10) != 0]
        else:               self.audiofile_l = [self.audiofile_l[idx] for idx in range(len(self.audiofile_l)) if (idx % 10) == 0]

        self.feature_d = {}
        self.patch_l = []

        audio_d = Namespace()

        with h5py.File(hdf5_audio_file, 'r') as audio_fid:
            for idx_file, audiofile in enumerate(tqdm(self.audiofile_l)):
                audio_d.file = audiofile
                audio_d.value_v = audio_fid[audio_d.file][:]
                audio_d.sr_hz = audio_fid[audio_d.file].attrs['sr_hz']
                # --- CQT_3m(6,freq,time)

                self.feature_d[audio_d.file] = {'X': torch.from_numpy(audio_d.value_v).float()}

                entry_l = pyjama_d['collection']['entry']
                genre = [entry['genre'][0]['value'] for entry in entry_l  if entry['filepath'][0]['value']==audiofile][0]

                localpatch_l = f_get_patches(self.feature_d[audio_d.file]['X'].size(0))
                for localpatch in localpatch_l:
                    self.patch_l.append({'audiofile': audio_d.file,
                                        'start_frame': localpatch['start_frame'],
                                        'end_frame': localpatch['end_frame'],
                                        'genre': genre
                                        })
        self.list_genre_l = set([patch['genre'] for patch in self.patch_l])


    def __len__(self):
        return len(self.patch_l)

    def __getitem__(self, idx_patch):
        X = self.feature_d[ self.patch_l[idx_patch]['audiofile'] ]['X']
        s = self.patch_l[idx_patch]['start_frame']
        e = self.patch_l[idx_patch]['end_frame']
        X = X[s:e]
        y = self.patch_l[idx_patch]['genre']
        return {'X':X, 'y': y }
```


## Pytorch lightning

Also to make life easier we will use pytorch-lightning for the training/validation loop since it avoids having to write tons of code.

Based on https://lightning.ai/docs/pytorch/stable/

```python
import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L

# define any number of nn.Modules (or use your current ones)
encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))


# define the LightningModule
class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# init the autoencoder
autoencoder = LitAutoEncoder(encoder, decoder)

dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
train_loader = utils.data.DataLoader(dataset)

trainer = L.Trainer(limit_train_batches=100, max_epochs=1)
trainer.fit(model=autoencoder, train_dataloaders=train_loader)
```
