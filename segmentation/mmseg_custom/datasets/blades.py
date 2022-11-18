# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

@DATASETS.register_module(force=True)
class BladesDataset(CustomDataset):
    """Invenergy Blades dataset.
    """
    CLASSES = ('blade', 'background')

    PALETTE = [[255, 255, 255], [32, 32, 32]]

    def __init__(self, **kwargs):
        print("***** ARGUMENTS FOR BLADES DATASET", kwargs)
        super(BladesDataset, self).__init__(
            # img_dir='/net/projects/invenergy/data/InvenergyPhotos/vit_dataset/data/my_dataset',
            #update the img_suffix to JPG
            img_suffix='.JPG',
            seg_map_suffix='_seg.png',
            # reduce_zero_label=True,
            **kwargs)