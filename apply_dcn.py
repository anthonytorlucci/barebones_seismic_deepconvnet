import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
from torch.nn import functional as F

from patch_dcn import PatchDCN
from plotting import plot_gather

PATCH_SIZE = 32  # NOTE: must match the patch_size used in the PatchDataset

def apply_dcn_to_gather(dcn_model:torch.nn.Module, ensemble_data:torch.tensor, nn_patch_size:int, ) -> torch.tensor:
    """Predict each patch in the ensemble and coalesce into a final gather of same size as input.

    Parameters
    ----------
    dcn_model : torch.nn.Module
        Trained deep convolutional autoencoder network used for patch prediction.
    ensemble_data : torch.tensor
        Input 2D array with shape (number_of_traces, number_of_samples).
        This is opposite of what most seismic gather processing programs would use, but it
        was chosen to keep the C-style indexing and samples being the fastest dimension.
    nn_patch_size : int
        The patch size used in the training.
    """
    input_fold = ensemble_data.shape[0]  # number of traces in the input ensemble
    input_nsps = ensemble_data.shape[1]  # number of samples per trace in the input ensemble
    gather = F.pad(ensemble_data, pad=(nn_patch_size-1,nn_patch_size-1,nn_patch_size-1,nn_patch_size-1), mode='constant', value=0.0)
    print(f"padded shape: {gather.shape}")

    gather_output = torch.zeros_like(gather, device='cpu')
    # NOTE: the nested loop could be run in parallel on multiple threads as each patch is independent of the neighbors and summed at the end.
    with torch.no_grad():
        for i in range(input_fold+nn_patch_size-1):
            for j in range(input_nsps+nn_patch_size-1):
                patch = gather[i:i+nn_patch_size,j:j+nn_patch_size]
                # print(i, j, "in  ", patch.mean(), patch.std(), patch.min(), patch.max())
                sclr = torch.max(torch.abs(patch))
                if sclr < 0.001:
                    patch_predicted = patch  # likely in the mute zone
                else:
                    patch = patch / sclr  # normalizes to range (-1,1); same as what is done in the Dataset.__getitem__() call
                    patch_predicted = dcn_model(patch.reshape((1,1,nn_patch_size,nn_patch_size)))
                    # simple scaling to match amplitude range of input
                    #---patch = (patch / torch.max(torch.abs(patch))) * sclr
                    patch_predicted = patch_predicted * sclr
                patch = patch_predicted.reshape((nn_patch_size,nn_patch_size))
                
                # print(i, j, "out ", patch.mean(), patch.std(), patch.min(), patch.max())
                # print(input_fold, i, i+nn_patch_size, input_nsps, j, j+nn_patch_size)
                gather_output[i:i+nn_patch_size,j:j+nn_patch_size] = gather_output[i:i+nn_patch_size,j:j+nn_patch_size] + patch
            
    gather_output = gather_output / (nn_patch_size**2)  # normalize;
    # reshape to match input by trimming off the padding
    return gather_output[nn_patch_size-1:input_fold+nn_patch_size-1, nn_patch_size-1:input_nsps+nn_patch_size-1]


# --- applying the model prediction
model = PatchDCN.load_from_checkpoint("models/config_0/version_0/checkpoints/epoch=20-step=202587.ckpt").to('cpu')
model.eval()

# tmp = torch.rand(1,1,PATCH_SIZE,PATCH_SIZE)
# tmp_hat = model(tmp)
# print(tmp.size())

numpy_data = numpy.load("data/3D_gathers_pstm_reformat_ieee_little_endian_full_fold_inline_1360_1400.npy")
# TRIM SAMPLES FOR FASTER DISPLAY
numpy_data = numpy_data[:,:1001]
input_ensemble = torch.tensor(numpy_data, device='cpu')
print(input_ensemble.size())
output_ensemble = apply_dcn_to_gather(dcn_model=model, ensemble_data=input_ensemble, nn_patch_size=PATCH_SIZE)

# plot
fig = plt.figure(figsize=(16, 12))
plot_gather(fig_handle=fig, input_ensemble=input_ensemble, output_ensemble=output_ensemble)
# plt.show()
fig.savefig("images/example_inline_1360_crossline_1400.png")