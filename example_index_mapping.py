from patch_dset import PatchDataset

# creating the index mapping, index : (ensemble_index, fold_patch_start, samples_patch_start),
idx_map = PatchDataset.create_index_mapping(mode='train', num_ensembles=475, max_fold=61, num_samples=1501, patch_size=32)

pdset = PatchDataset(max_fold=61, num_samples=1501, patch_size=32, mode='train', patch_index_mapping=idx_map)
print(f"number of ensembles in training set: {len(pdset.ensemble_files)}")
