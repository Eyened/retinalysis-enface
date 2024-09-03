import os
import random
import warnings
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd

from rtnls_enface.fundus import Fundus
from rtnls_fundusprep.mask_extraction import Bounds


class PairedFileLoader:
    def __init__(
        self,
        items=Dict[str, Dict[str, Any]],
    ):
        # we check that, if provided, the lengths of the inputs are all the same.
        self.items = items
        self.indices = list(items.keys())

    def make_object(self, index):
        raise NotImplementedError()

    def __getitem__(self, key):
        if isinstance(key, slice):
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else len(self)
            step = key.step if key.step is not None else 1
            return [self.make_object(self.indices[i]) for i in range(start, stop, step)]
        else:
            return self.make_object(self.indices[key])

    def by_id(self, id):
        return self.make_object(id)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.items)

    def sample(self, n):
        indices = random.sample(range(len(self)), n)

        items = {self.indices[i]: self.items[self.indices[i]] for i in indices}
        return self.__class__(items)

    def to_dict(self):
        return list(self.items.values())

    @classmethod
    def _get_items_from_files(
        cls, **file_lists: Dict[str, List[str | Path]]
    ) -> Dict[str, Dict[str, Any]]:
        stem_dicts = {
            key: None if file_list is None else {Path(p).stem: p for p in file_list}
            for key, file_list in file_lists.items()
        }
        for key, d in stem_dicts.items():
            if d is not None:
                print(f"Folder {key}: {len(d)} files")

        stems = [list(d.keys()) for d in stem_dicts.values() if d is not None]
        intersection = set.intersection(*[set(lst) for lst in stems])
        stems = sorted(list(intersection))

        items = {
            stem: {
                key: (d[stem] if d is not None else None)
                for key, d in stem_dicts.items()
            }
            for stem in stems
        }
        for key, item in items.items():
            item["id"] = key
        return items

    @classmethod
    def _get_items_from_folders(
        cls, **folders: Dict[str, str | Path]
    ) -> Dict[str, Dict[str, Any]]:
        file_lists = {
            key: sorted(list(Path(folder).glob("*.png")))
            if folder is not None
            else None
            for key, folder in folders.items()
        }
        return cls._get_items_from_files(**file_lists)


class FundusLoader(PairedFileLoader):
    def get_item(self, id):
        item = self.items[id]
        if "metadata" in item:
            bounds = Bounds(**item["metadata"])
            M = bounds.get_cropping_matrix(1024)
            item["bounds"] = bounds.warp(M, (1024, 1024))
        return item

    def make_object(self, id):
        return Fundus.from_file(**self.get_item(id))

    @classmethod
    def _add_meta(cls, items: Dict[str, Dict[str, Any]], meta_path):
        if meta_path is None:
            return
        metadata = pd.read_csv(meta_path, index_col="id")
        metadata.index = metadata.index.astype(str)

        for id, item in items.items():
            item["metadata"] = metadata.loc[id].to_dict()

    @classmethod
    def _add_fovea_locations(
        cls, items: Dict[str, Dict[str, Any]], fovea_locations_path
    ):
        if fovea_locations_path is None:
            return
        fovea_locations_df = pd.read_csv(fovea_locations_path, index_col=0)
        fovea_locations_df.index = fovea_locations_df.index.astype(str)

        for id, item in items.items():
            item["fovea_location"] = (
                fovea_locations_df.loc[id, "mean_x"],
                fovea_locations_df.loc[id, "mean_y"],
            )

    @classmethod
    def from_paths(cls, disc_paths, fundus_paths, fovea_locations_path, meta_path):
        if disc_paths is None and fundus_paths is None:
            raise ValueError("Either disc_paths or fundus_paths must be provided")
        (disc_paths, fundus_paths), stems = cls._get_items_from_files(
            disc_paths, fundus_paths
        )
        fovea_locations = cls._read_fovea_locations(fovea_locations_path, stems)
        metadata = cls._add_meta(meta_path, stems)
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

        (disc_paths, fundus_paths), stems = cls._get_items_from_folders(
            discs_folder, fundus_folder
        )
        fovea_locations = cls._read_fovea_locations(fovea_locations_path, stems)
        metadata = cls._add_meta(meta_path, stems)
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
