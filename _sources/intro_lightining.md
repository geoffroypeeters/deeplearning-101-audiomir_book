### TorchLightning training

[Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/)
- is a high-level wrapper for PyTorch that **simplifies** the process of organizing, training, and scaling models.
- **structures** PyTorch code with best practices, making it easier to implement, debug, and accelerate models across different hardware with minimal boilerplate code.
- allows to **bypass the tedious writing** of training and validation loop over epoch and over mini-batch.

The <mark>writing of the Lightning class</mark> is very standard and almost the same for all tasks.
It involves indicating
- which `model` to use, `loss` to minimize and the `optimizer` to use
- what is a step of forward pass for training (`training_step`) and validation (`validation_step`)

```python
class AutoTaggingLigthing(pl.LightningModule):

    def __init__(self, in_model):
        super().__init__()
        self.model = in_model
        self.loss= nn.BCELoss()

    def training_step(self, batch, batch_idx):
        hat_y = self.model(batch['X'])
        loss = self.loss(hat_y, batch['y'])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        hat_y = self.model(batch['X'])
        loss = self.loss(hat_y, batch['y'])
        self.log('val_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), 0.001)
        return optimizer
```


<hr style="border: 2px solid red; margin: 60px 0;">


The training code is then extremely simple: `trainer.fit`.\
Pytorch Lightning also allows to define **CallBack** using predefined methods such as
- `EarlyStopping` to avoid over-fitting or
- `ModelCheckpoint` for saving the best model

```python
my_lighting = AutoTaggingLigthing( model )

early_stop_callback = EarlyStopping(monitor="val_loss",
                                    patience=10,
                                    verbose=True,
                                    mode="min")
checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                      dirpath=param_lightning.dirpath,
                                      filename=param_lightning.filename,
                                      save_top_k=1,
                                      mode='min')

trainer = pl.Trainer(accelerator="gpu",
                    max_epochs = param_lightning.max_epochs,
                    callbacks = [early_stop_callback, checkpoint_callback])
trainer.fit(model=my_lighting,
            train_dataloaders=train_dataloader,
            val_dataloaders=valid_dataloader)
```

<hr style="border: 2px solid red; margin: 60px 0;">
