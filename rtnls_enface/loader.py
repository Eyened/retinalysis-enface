import os
import random
import warnings
from ast import Dict
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd
from rtnls_fundusprep.mask_extraction import Bounds

from rtnls_enface.fundus import Fundus


class FundusLoader:
    def __init__(
        self,
        disc_paths: List[str] = None,
        fundus_paths: List[str] = None,
        fovea_locations: List[Tuple[float, float]] = None,
        metadata: List[Dict] = None,
        ids: List[int] = None,
    ):
        # we check that, if provided, the lengths of the inputs are all the same.

        self.disc_paths = disc_paths
        self.fundus_paths = fundus_paths
        self.fovea_locations = fovea_locations
        self.metadata = metadata
        self.ids = ids
        self.id_to_index = {id: i for i, id in enumerate(ids)}
        if metadata is None:
            print("Meta not provided. Fundus will be created with scaling_factor=None")

        # self.count = -1
        self.num_items = self._check_lengths(
            disc_paths, fundus_paths, fovea_locations, metadata, ids
        )

    def _check_lengths(self, *elems):
        lengths = [len(el) for el in elems if el is not None]
        if len(lengths) > 0:
            assert all(
                l == lengths[0] for l in lengths
            ), f"Input lengths {lengths} are inconsistent"
            num_items = lengths[0]
        else:
            num_items = 0
        return num_items

    def _get_one_dict(self, index):
        res = dict()
        if self.ids is not None:
            res["id"] = self.ids[index]
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
        for i in range(self.num_items):
            yield self[i]

    # def __next__(self):
    #     if self.count < self.num_items - 1:
    #         self.count += 1
    #         return self[self.count]
    #     else:
    #         raise StopIteration

    def __len__(self):
        return self.num_items

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
    def _get_filenames(cls, *file_lists):
        stem_dicts = [
            None if file_list is None else {Path(p).stem: p for p in file_list}
            for file_list in file_lists
        ]
        stems = [list(d.keys()) for d in stem_dicts if d is not None]
        intersection = set.intersection(*[set(lst) for lst in stems])
        stems = sorted(list(intersection))

        filtered_lists = [
            None
            if d is None
            else [
                d[stem] for stem in stems
            ]  # sorted([fp for fp in file_list if Path(fp).stem in intersection])
            for d in stem_dicts
        ]
        return filtered_lists, stems

    @classmethod
    def _read_meta(cls, meta_path, ids):
        if meta_path is None:
            return None
        metadata = pd.read_csv(meta_path, index_col="id")
        if metadata.index.dtype == int:
            ids = [int(id) for id in ids]
        metadata = metadata.loc[ids].to_dict("records")
        return metadata

    @classmethod
    def _read_fovea_locations(cls, fovea_locations_path, ids):
        if fovea_locations_path is None:
            return None
        fovea_locations_df = pd.read_csv(fovea_locations_path, index_col=0)
        if fovea_locations_df.index.dtype == int:
            ids = [int(id) for id in ids]
        fovea_locations_df = fovea_locations_df.loc[ids]
        fovea_locations = list(
            zip(
                fovea_locations_df["mean_x"].to_list(),
                fovea_locations_df["mean_y"].to_list(),
            )
        )
        return fovea_locations

    # @classmethod
    # def _prep_paths(cls, disc_paths, fundus_paths, fovea_locations_path, meta_path):
    #     if disc_paths is not None:
    #         stems = [Path(p).stem for p in disc_paths]
    #     elif fundus_paths is not None:
    #         stems = [Path(p).stem for p in fundus_paths]
    #     else:
    #         raise ValueError("Either disc_paths or fundus_paths must be provided")
    #     if disc_paths is not None and fundus_paths is not None:
    #         for dp, fp in zip(disc_paths, fundus_paths):
    #             assert Path(dp).stem == Path(fp).stem

    #     return disc_paths, fundus_paths, fovea_locations, metadata, stems

    @classmethod
    def from_paths(cls, disc_paths, fundus_paths, fovea_locations_path, meta_path):
        if disc_paths is None and fundus_paths is None:
            raise ValueError("Either disc_paths or fundus_paths must be provided")
        (disc_paths, fundus_paths), stems = cls._get_filenames(disc_paths, fundus_paths)
        # disc_paths, fundus_paths, fovea_locations, metadata, ids = cls._prep_paths(
        #     disc_paths, fundus_paths, fovea_locations_path, meta_path
        # )
        fovea_locations = cls._read_fovea_locations(fovea_locations_path, stems)
        metadata = cls._read_meta(meta_path, stems)
        return cls(disc_paths, fundus_paths, fovea_locations, metadata, stems)

    @classmethod
    def _read_folders(
        cls,
        discs_folder: Union[str, Path] = None,
        fundus_folder: Union[str, Path] = None,
        fovea_locations_path: Union[str, Path] = None,
        meta_path: Union[str, Path] = None,
    ):
        # either disc or fundus must be available'
        disc_paths = (
            sorted(list(Path(discs_folder).glob("*.png")))
            if discs_folder is not None
            else None
        )
        fundus_paths = (
            sorted(list(Path(fundus_folder).glob("*.png")))
            if fundus_folder is not None
            else None
        )
        (disc_paths, fundus_paths), stems = cls._get_filenames(disc_paths, fundus_paths)
        fovea_locations = cls._read_fovea_locations(fovea_locations_path, stems)
        metadata = cls._read_meta(meta_path, stems)
        return stems, disc_paths, fundus_paths, fovea_locations, metadata

    @classmethod
    def from_folders(
        cls,
        discs_folder: Union[str, Path] = None,
        fundus_folder: Union[str, Path] = None,
        path_to_fovea_locations_df: Union[str, Path] = None,
        meta_path: str = None,
    ):
        stems, disc_paths, fundus_paths, fovea_locations, metadata = cls._read_folders(
            discs_folder, fundus_folder, path_to_fovea_locations_df, meta_path
        )
        return cls(disc_paths, fundus_paths, fovea_locations, metadata, ids=stems)

    @classmethod
    def from_folder(
        cls,
        base_folder: Union[str, Path],
        discs_subfolder: str = "discs",
        fundus_subfolder: str = "rgb",
        meta_csv: str = "meta.csv",
        fovea_locations_csv: str = "fovea.csv",
    ):
        base = Path(base_folder)
        meta_csv = base / meta_csv if meta_csv is not None else None
        if meta_csv is not None and not os.path.exists(meta_csv):
            warnings.warn(f"file {meta_csv} not found")
        fovea_locations_csv = (
            base / fovea_locations_csv if fovea_locations_csv is not None else None
        )
        if fovea_locations_csv is not None and not os.path.exists(fovea_locations_csv):
            warnings.warn(f"file {fovea_locations_csv} not found")
        return cls.from_folders(
            base / discs_subfolder if discs_subfolder is not None else None,
            base / fundus_subfolder if fundus_subfolder is not None else None,
            fovea_locations_csv,
            meta_csv,
        )

    def to_dict(self):
        res = [{"id": self.ids[i]} for i in range(self.num_items)]
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
