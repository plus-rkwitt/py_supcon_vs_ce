from .config import DS_PATH_CFG , DS_SPLIT_CFG

from pytorch_utils.data.dataenv import DatasetFactory,\
    StratifiedShuffleSplitDatasetFactory


ds_factory = DatasetFactory(DS_PATH_CFG)


ds_split_factory = StratifiedShuffleSplitDatasetFactory(
    ds_factory,
    DS_SPLIT_CFG,
    'train_indices.pkl'
)
