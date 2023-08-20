"""Deep neural network consisting of conv2d layers"""

# standard libs
import math

# third party
import torch
# import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl

# local

class PatchDCN(pl.LightningModule):
    
    def __init__(self, n_chans:int, k_mult:int=2, dp:float=0.0, lr:float=0.001):
        """Deep Convolutional Neural Network for 2D patches.

        Parameters
        ----------
        n_chans : int
            number of input channels for a given patch with shape (channels, height, width); typically just one for seismic, but could be greater than one if including attributes.
        k_mult : int
            depthwise convolution multiplier
        dp : float
            dropout
        lr : float
            learning rate for optimizer
        """
        # call the constructor of the super class
        #--super().__init__()
        super(PatchDCN, self).__init__()
        self.save_hyperparameters()
        
        self._n_chans = n_chans
        self._k_mult = k_mult  # depthwise convolution
        self._kern_sz = (3,3) #(2,2)  # kernel size

        self._dp = dp  # dropout
        self._lr = lr
        self.loss_function = nn.MSELoss()

        # define network layers
        self._bnorm = nn.BatchNorm2d(num_features=n_chans, eps=1e-5, momentum=0.1, affine=True)
        self._m1 = self.layer2d(k_in=1)
        k = int(math.pow(self._k_mult,0))  # k=1
        self._m2 = self.layer2d(k_in=k*self._k_mult)
        k = int(math.pow(self._k_mult,1))  # k=2 when k_mult=2
        self._m3 = self.layer2d(k_in=k*self._k_mult)
        k = int(math.pow(self._k_mult,2))  # k=4 when k_mult=2
        self._m4 = self.layer2d(k_in=k*self._k_mult)
        
        # --- the turn
        k = int(math.pow(self._k_mult,3))  # k=8
        self._m4T = self.layer2dT(k_in=k*self._k_mult)
        k = int(math.pow(self._k_mult,2))  # k=4
        self._m3T = self.layer2dT(k_in=k*self._k_mult)
        k = int(math.pow(self._k_mult,1))  # k=2
        self._m2T = self.layer2dT(k_in=k*self._k_mult)
        k = int(math.pow(self._k_mult,0))  # k=1
        self._m1T = self.layer2dT(k_in=k*self._k_mult)

        
    def layer2d(self, k_in:int):
        k_out = k_in * self._k_mult
        lyr = nn.Sequential(
            nn.Conv2d(
                in_channels=self._n_chans*k_in, 
                out_channels=self._n_chans*k_out, 
                kernel_size=self._kern_sz, 
                stride=(1,1), 
                padding=(0,0), 
                dilation=(1,1), 
                groups=self._n_chans*k_in),
            nn.Dropout2d(p=self._dp),
            nn.Tanh()
        )
        return lyr

    def layer2dT(self, k_in:int):
        k_out = k_in // self._k_mult
        lyr = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self._n_chans*k_in, 
                out_channels=self._n_chans*k_out, 
                kernel_size=self._kern_sz, 
                stride=(1,1), 
                padding=(0,0), 
                output_padding=(0,0), 
                groups=self._n_chans*k_out),
            nn.Dropout2d(p=self._dp),
            nn.Tanh()
        )
        return lyr       

    def forward(self, x: torch.tensor):
        """forward pass
        Parameters
        ----------
        x : torch.tensor
            patch with Size([batch_size,C,H,W])
        
        Returns
        -------
        x : torch.tensor
            patch with Size([batch_size,C,H,W])
        """
        x = self._bnorm(x)
        x = self._m1(x)
        x = self._m2(x)
        x = self._m3(x)
        x = self._m4(x)
        x = self._m4T(x)
        x = self._m3T(x)
        x = self._m2T(x)
        x = self._m1T(x)
        
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)