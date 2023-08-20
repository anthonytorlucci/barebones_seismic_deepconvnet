import numpy
import matplotlib.pyplot as plt

PATCH_SIZE = 32  # NOTE: must match the patch_size used in the PatchDataset
FOLD = 61
NSAMP = 1501

# illustrating sliding window
gather = numpy.zeros(shape=(FOLD+(2*(PATCH_SIZE-1)),NSAMP+(2*(PATCH_SIZE-1))))
patch = numpy.ones(shape=(PATCH_SIZE,PATCH_SIZE))  # patch_size, patch_size

patch = numpy.ones(shape=(PATCH_SIZE,PATCH_SIZE))  # patch_size, patch_size
for i in range(FOLD+PATCH_SIZE-1):
    for j in range(NSAMP+PATCH_SIZE-1):
        gather[i:i+PATCH_SIZE,j:j+PATCH_SIZE] = gather[i:i+PATCH_SIZE,j:j+PATCH_SIZE] + patch

# print(numpy.max(gather))
gather = gather / (PATCH_SIZE**2)  # normalize;
# print(gather[:PATCH_SIZE+1,:PATCH_SIZE+1])

gather = gather[PATCH_SIZE-1:FOLD+PATCH_SIZE-1, PATCH_SIZE-1:NSAMP+PATCH_SIZE-1]
print(f"output shape: {gather.shape}")
# print(gather[:2,:2])
print(numpy.min(gather), numpy.max(gather))

