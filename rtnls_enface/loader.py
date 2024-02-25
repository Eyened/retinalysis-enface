import random
from ast import Dict
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from mask_extraction import Bounds
from rtnls_enface.fundus import Fundus


class FundusLoader:
    def __init__(
        self,
        disc_paths: List[str] = None,
        fundus_paths: List[str] = None,
        fovea_locations: List[Tuple[float, float]] = None,
        metadata: List[Dict] = None,
    ):
        # we check that, if provided, the lengths of the inputs are all the same.
        lengths = [
            len(el)
            for el in [disc_paths, fundus_paths, fovea_locations, metadata]
            if el is not None
        ]
        if len(lengths) > 0:
            assert all(
                l == lengths[0] for l in lengths
            ), "Input lengths are inconsistent"
            self.num_items = lengths[0]
        else:
            self.num_items = 0

        self.disc_paths = disc_paths
        self.fundus_paths = fundus_paths
        self.fovea_locations = fovea_locations
        self.metadata = metadata
        if metadata is None:
            print("Meta not provided. Fundus will be created with scaling_factor=None")

        self.count = -1

    def _get_one_dict(self, index):
        res = dict()
        if self.disc_paths is not None:
            res["disc_path"] = self.disc_paths[index]

        if self.fundus_paths is not None:
            res["fundus_path"] = self.fundus_paths[index]

        if self.fovea_locations is not None:
            res["fovea_location"] = self.fovea_locations[index]

        if self.metadata is not None:
            bounds = Bounds(**self.metadata[index])
            M = bounds.get_cropping_matrix(1024)
            res["bounds"] = bounds.warp(M, (1024, 1024))

        return res

    def _get_one(self, index):
        return Fundus.from_file(**self._get_one_dict(index))

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
        if self.count < self.num_items - 1:
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
    def _read_folders(
        cls,
        discs_folder: str | Path = None,
        fundus_folder: str | Path = None,
        meta_path: str | Path = None,
        fovea_locations_path: str | Path = None,
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
            metadata = metadata.loc[ids].to_dict("records")
        else:
            metadata = None

        if fovea_locations_path is not None:
            fovea_locations_df = pd.read_csv(fovea_locations_path, index_col=0)
            ids = disc_stems
            if fovea_locations_df.index.dtype == int:
                ids = [int(id) for id in ids]
            fovea_locations_df = fovea_locations_df.loc[ids]
            fovea_locations = list(
                zip(
                    fovea_locations_df["mean_x"].to_list(),
                    fovea_locations_df["mean_y"].to_list(),
                )
            )
        else:
            fovea_locations = None

        return disc_paths, fundus_paths, metadata, fovea_locations

    @classmethod
    def from_folders(
        cls,
        discs_folder: str | Path = None,
        fundus_folder: str | Path = None,
        path_to_fovea_locations_df: str | Path = None,
        meta_path: str = None,
    ):
        disc_paths, fundus_paths, metadata, fovea_locations = cls._read_folders(
            discs_folder, fundus_folder, meta_path, path_to_fovea_locations_df
        )
        return cls(disc_paths, fundus_paths, metadata, fovea_locations)

    @classmethod
    def from_folder(
        cls,
        base_folder: str | Path,
        discs_subfolder: str = "discs",
        fundus_subfolder: str = "rgb",
        meta_csv: str = "meta.csv",
        fovea_locations_csv: str = "fovea.csv",
    ):
        base = Path(base_folder)
        return cls.from_folders(
            base / discs_subfolder,
            base / fundus_subfolder,
            base / fovea_locations_csv,
            base / meta_csv,
        )

    def to_dict(self):
        res = [dict() for _ in range(self.num_items)]
        if self.disc_paths is not None:
            for i, fp in enumerate(self.disc_paths):
                res[i]["disc_path"] = fp

        if self.fundus_paths is not None:
            for i, fp in enumerate(self.fundus_paths):
                res[i]["fundus_path"] = fp

        if self.fovea_locations is not None:
            for i, l in enumerate(self.fovea_locations):
                res[i]["fovea_location"] = l

        return res
