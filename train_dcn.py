"""Train CNN2D deep neural network with input seismic patches from pstm gathers"""

import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers

from patch_dcn import PatchDCN
from patch_dset import PatchDataModule, PatchDataset

# Logger parameters
LGR_VERSION = 0
LGR_EXPERIMENT_NAME = "config_0"  # useful when testing different configurations of hyperparameters


# DataModule parameters
BATCH_SIZE = 128
PATCH_SIZE = 32  # must be less than 61 for this training data
MAX_FOLD = 61  # number of traces in each ensemble for training/testing/validating; data.shape[0]
NUM_SAMPLES = 1501  # number of samples in each trace; data.shape[1]

# Model hyper-parameters
N_CHANNELS = 1  # first dimension; number of attributes in each patch; data dependent
K_MULTIPLIER = 2  # depthwise convolution; see https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
DROPOUT = 0.1
LEARNING_RATE = 1e-3

csv_logger = pl_loggers.CSVLogger(
    save_dir="models",
    name=LGR_EXPERIMENT_NAME, 
    version=LGR_VERSION, 
    prefix="", 
    flush_logs_every_n_steps=100)
# trainer = pl.Trainer(max_epochs=1, accelerator='cpu', devices=1)
trainer = pl.Trainer(
    max_epochs=21, 
    accelerator='gpu', 
    devices=1, 
    logger=csv_logger)
dm = PatchDataModule(
    batch_size=BATCH_SIZE, 
    max_fold=MAX_FOLD,
    num_samples=NUM_SAMPLES,
    patch_size=PATCH_SIZE)
dm.setup(stage="fit")

model = PatchDCN(n_chans=N_CHANNELS, k_mult=K_MULTIPLIER, dp=DROPOUT, lr=LEARNING_RATE)
trainer.fit(model, datamodule=dm)

# dm.setup(stage="test")
# trainer.test(model, datamodule=dm)