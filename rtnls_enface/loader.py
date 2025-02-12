import os
import random
import warnings
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd

from rtnls_enface.fundus import Fundus
from rtnls_fundusprep.cfi_bounds import CFIBounds


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

    def filter(self, filter_fn):
        self.items = {k: v for k, v in self.items.items() if filter_fn(v)}
        self.indices = list(self.items.keys())

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
    # load an object
    def get_item(self, id):
        item = self.items[id]
        if "bounds" in item:
            bounds = CFIBounds(**item["bounds"])
            M = bounds.get_cropping_transform(1024)
            item["bounds"] = bounds.warp(M)
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
            if id not in metadata.index:
                raise ValueError(f"ID {id} not found in metadata")
            success = metadata.loc[id, "success"]
            if not success:
                item["bounds"] = None
                warnings.warn(
                    f"ID {id} metadata has success=False. Bounds set to None."
                )
            else:
                item["bounds"] = eval(metadata.loc[id, "bounds"])

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
    def from_folders(
        cls,
        discs_folder: Union[str, Path] = None,
        fundus_folder: Union[str, Path] = None,
        fovea_locations_path: Union[str, Path] = None,
        meta_path: Union[str, Path] = None,
    ):
        items = cls._get_items_from_folders(
            disc_path=discs_folder, fundus_path=fundus_folder
        )
        cls._add_fovea_locations(items, fovea_locations_path)
        cls._add_meta(items, meta_path)
        return cls(items)

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
        meta_path = base / meta_csv if meta_csv is not None else None
        fovea_locations_path = (
            base / fovea_locations_csv if fovea_locations_csv is not None else None
        )

        for path in [meta_path, fovea_locations_path]:
            if path is not None and not os.path.exists(path):
                warnings.warn(f"File {path} not found")

        return cls.from_folders(
            discs_folder=base / discs_subfolder
            if discs_subfolder is not None
            else None,
            fundus_folder=base / fundus_subfolder
            if fundus_subfolder is not None
            else None,
            fovea_locations_path=fovea_locations_path,
            meta_path=meta_path,
        )
