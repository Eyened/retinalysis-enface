import random
from ast import Dict
from pathlib import Path
from typing import List

import pandas as pd
from mask_extraction import Bounds
from rtnls_enface.fundus import Fundus


class FundusLoader:
    def __init__(
        self,
        disc_paths: List[str] = None,
        fundus_paths: List[str] = None,
        metadata: List[Dict] = None,
    ):
        if disc_paths is not None and fundus_paths is not None:
            assert len(disc_paths) == len(fundus_paths)

        self.disc_paths = disc_paths
        self.fundus_paths = fundus_paths

        self.metadata = metadata
        if metadata is None:
            print("Meta not provided. Fundus will be created with scaling_factor=None")
        else:
            if disc_paths is not None:
                assert len(disc_paths) == len(metadata)
            if fundus_paths is not None:
                assert len(fundus_paths) == len(metadata)

        self.count = -1

    def _get_one(self, index):
        fundus = Fundus()
        if self.disc_paths is not None:
            fundus.load_disc(self.disc_paths[index])
        if self.fundus_paths is not None:
            fundus.load_fundus_image(self.fundus_paths[index])
        if self.metadata is not None:
            bounds = Bounds(**self.metadata[index])
            M = bounds.get_cropping_matrix(1024)
            fundus.set_bounds(bounds.warp(M, (1024, 1024)))
            # fundus.scaling_factor = (
            #     self.meta.loc[self.ids[index], "scaling_w"]
            #     * self.meta.loc[self.ids[index], "scale"]
            # )

        return fundus

    def __getitem__(self, key):
        if isinstance(key, slice):
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else len(self)
            step = key.step if key.step is not None else 1
            return [self._get_one(i) for i in range(start, stop, step)]
        else:
            return self._get_one(key)

    def by_id(self, id):
        return self[self.id_to_index[id]]

    def __iter__(self):
        return self

    def __next__(self):
        if self.count < len(self.mask_paths) - 1:
            self.count += 1
            return self[self.count]
        else:
            raise StopIteration

    def __len__(self):
        return len(self.mask_paths)

    def sample(self, n):
        indices = random.sample(range(len(self)), n)
        mask_paths = [self.mask_paths[i] for i in indices]
        disc_paths = (
            [self.disc_paths[i] for i in indices]
            if self.disc_paths is not None
            else None
        )
        fundus_paths = (
            [self.fundus_paths[i] for i in indices]
            if self.fundus_paths is not None
            else None
        )
        return FundusLoader(mask_paths, disc_paths, fundus_paths, self.meta_path)

    @classmethod
    def from_folders(
        cls,
        discs_folder: str = None,
        fundus_folder: str = None,
        meta_path: str = None,
    ):
        if discs_folder is not None:
            disc_paths = sorted(list(Path(discs_folder).glob("*.png")))
            disc_stems = [f.stem for f in disc_paths]
        else:
            disc_paths = None

        if fundus_folder is not None:
            fundus_paths = sorted(list(Path(fundus_folder).glob("*.png")))
            fundus_stems = [f.stem for f in fundus_paths]
        else:
            fundus_paths = None

        if disc_paths is not None and fundus_paths is not None:
            for dp, fp in zip(disc_paths, fundus_paths):
                assert dp.stem == fp.stem

        if meta_path is not None:
            metadata = pd.read_csv(meta_path, index_col="id")
            ids = disc_stems
            if metadata.index.dtype == int:
                ids = [int(id) for id in ids]
            metadata = metadata.loc[ids]

        return cls(disc_paths, fundus_paths, metadata.to_dict("records"))
