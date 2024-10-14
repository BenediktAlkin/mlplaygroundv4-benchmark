import numpy as np
from kappadata.copying.folder import copy_folder_from_global_to_local
from torchvision.datasets.folder import default_loader

from distributed.config import barrier, is_data_rank0
from utils.num_worker_heuristic import get_fair_cpu_count
from .base.dataset_base import DatasetBase


class CLEVRCounts(DatasetBase):
    def __init__(self, split, global_root=None, local_root=None, **kwargs):
        super().__init__(**kwargs)
        global_root, local_root = self._get_roots(global_root, local_root, "clevr")
        if split in ["valid", "validation"]:
            split = "val"
        assert split in ["train", "val"]
        if local_root is None:
            # load data from global_root
            self.source_root = global_root
            self.logger.info(f"data_source (global): '{self.source_root}'")
        else:
            # load data from local_root
            self.source_root = local_root / "CLEVR_v1.0"
            if is_data_rank0():
                # copy data from global to local
                self.logger.info(f"data_source (global): '{global_root}'")
                self.logger.info(f"data_source (local): '{self.source_root}'")
                # copy images
                copy_folder_from_global_to_local(
                    global_path=global_root / "images",
                    local_path=self.source_root / "images",
                    relative_path=split,
                    log_fn=self.logger.info,
                    num_workers=min(10, get_fair_cpu_count()),
                )
            barrier()
        # load annotations from global (small files)
        self.image_paths = np.load(global_root / "counts" / f"{split}_images.npy")
        self.labels = np.load(global_root / "counts" / f"{split}_labels.npy")

    def getitem_x(self, idx, ctx=None):
        # path is e.g. "./CLEVR_v1.0/images/val/CLEVR_val_000000.png" -> remove ./CLEVR_v1.0/
        path = self.source_root / self.image_paths[idx][len("./CLEVR_v1.0/"):]
        x = default_loader(path)
        return x

    # noinspection PyUnusedLocal
    def getitem_class(self, idx, ctx=None):
        return int(self.labels[idx][len("count_"):]) - 3

    @staticmethod
    def getshape_class():
        return 8,

    @property
    def class_names(self):
        return ["count_3", "count_4", "count_5", "count_6", "count_7", "count_8", "count_9", "count_10"]

    def __len__(self):
        return len(self.image_paths)
