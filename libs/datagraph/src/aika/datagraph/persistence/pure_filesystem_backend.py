import glob
import json
import os.path
import pickle
import re
import sys
from pathlib import Path
from retry import retry
from overrides import overrides
import portalocker
import typing as t
from frozendict import frozendict

from aika.datagraph import (
    IPersistenceEngine,
    DataSetMetadata,
    DataSetMetadataStub,
    DataSet,
)
from aika.datagraph.interface import _SerialisingBase
from aika.time import TimeRange


class FileSystemPersistenceEngine(_SerialisingBase):
    def __init__(self, root_file_path: str, compression="zip"):
        self._path = Path(root_file_path)
        self._compression = compression

    def __hash__(self):
        return hash(self._path.absolute())

    def __eq__(self, other):
        if isinstance(other, FileSystemPersistenceEngine):
            return self._path == other._path
        else:
            return NotImplemented  # pragma: no cover

    @overrides
    def set_state(self) -> t.Dict[str, t.Any]:
        return {"type": "pure_filesystem", "root_file_path": str(self._path.absolute())}

    def _base_file_name(self, name, hash):
        return f"{name}_h{str(hash)}"

    def _metadata_file_name(self, metadata: DataSetMetadata) -> str:
        return f"{self._base_file_name(metadata.name, metadata.__hash__())}.meta"

    def _metadata_file_path_from_hash(self, name, version, hash) -> Path:
        parent = self._path / version / "metadata"
        parent.mkdir(parents=True, exist_ok=True)
        return parent / (self._base_file_name(name, hash) + ".meta")

    def _metadata_file_path(self, metadata: DataSetMetadata) -> Path:
        parent = self._path / metadata.version / "metadata"
        parent.mkdir(parents=True, exist_ok=True)
        return parent / self._metadata_file_name(metadata)

    def _datafile_name(self, metadata: DataSetMetadata) -> str:
        return f"{self._base_file_name(metadata.name, metadata.__hash__())}.data"

    def _datafile_path(self, metadata: DataSetMetadata) -> Path:
        parent = self._path / metadata.version / "datafiles"
        parent.mkdir(parents=True, exist_ok=True)
        return parent / self._datafile_name(metadata)

    def _datafile_path_from_hash(self, name, version, hash) -> Path:
        parent = self._path / version / "datafiles"
        parent.mkdir(parents=True, exist_ok=True)
        return parent / (self._base_file_name(name, hash) + ".data")

    @overrides
    def _find_record(self, metadata: DataSetMetadata, include_data=False) -> t.Dict:
        metadata_path = self._metadata_file_path(metadata)
        if not os.path.exists(metadata_path):
            return None
        with portalocker.Lock(metadata_path, "rt") as metadata_file:
            record = json.load(metadata_file)
            if include_data:
                with open(self._datafile_path(metadata), "rb") as datafile:
                    record["data"] = datafile.read()
            return record

    @overrides
    def _find_record_from_hash(self, name, version, hash, include_data=False):
        metadata_path = self._metadata_file_path_from_hash(name, version, hash)
        if not os.path.exists(metadata_path):
            return None
        with portalocker.Lock(metadata_path, "rt") as metadata_file:
            record = json.load(metadata_file)
            if include_data:
                with open(
                    self._datafile_path_from_hash(name, version, hash), "rb"
                ) as datafile:
                    record["data"] = datafile.read()
            return record

    @overrides
    def replace(
        self,
        dataset: DataSet,
    ) -> bool:
        metadata_path = self._metadata_file_path(dataset.metadata)
        data_path = self._datafile_path(dataset.metadata)
        record = self._make_record(dataset)
        already_exists = metadata_path.exists()
        with portalocker.Lock(metadata_path, "wt") as metadata_file:
            with open(data_path, "wb") as data_file:
                pickle.dump(dataset.data, data_file)
                json.dump(record, metadata_file)
        return already_exists

    @overrides
    def exists(self, metadata: DataSetMetadata) -> bool:
        return self._metadata_file_path(metadata).exists()

    def _read_all_cached_metadata(self, version) -> t.Set:
        results = set()
        target_dir = self._path / version / "metadata"
        for file in os.listdir(target_dir):
            with open(target_dir / file, "rt") as f:
                results.add(self._deserialise_meta_data(json.load(f)))
        return results

    @overrides
    def find_successors(self, metadata: DataSetMetadata) -> t.Set[DataSetMetadataStub]:
        all_metadata = self._read_all_cached_metadata(metadata.version)
        successors = set()
        for candidate in all_metadata:
            if metadata in candidate.predecessors.values():
                successors.add(candidate)
        return successors

    def _delete_leaf(self, metadata: DataSetMetadata):
        if not self.exists(metadata):
            return False
        else:
            successors = self.find_successors(metadata)
            if len(successors) > 0:
                raise ValueError("Cannot delete a dataset that still has successors")
            else:
                metadata_path = self._metadata_file_path(metadata)
                data_path = self._datafile_path(metadata)
                os.unlink(data_path)
                os.unlink(metadata_path)
                return True

    @overrides()
    def delete(self, metadata: DataSetMetadata, recursive=False):
        if recursive:
            for successor in self.find_successors(metadata):
                self.delete(successor, recursive=True)
        return self._delete_leaf(metadata)

    @overrides
    def find(self, match: str, version: t.Optional[str] = None) -> t.List[str]:
        glob_pattern = os.path.join(
            self._path.absolute(), version if version else "**", "metadata", f"*.meta"
        )
        candidates = list(
            sorted(
                [
                    "_".join(Path(p).name.split("_")[:-1])
                    for p in glob.glob(glob_pattern, recursive=True)
                ]
            )
        )
        results = []
        for candidate in candidates:
            if re.match(match, candidate):
                results.append(candidate)

        return results

    @overrides
    def scan(
        self, dataset_name: str, params: t.Optional[t.Dict] = None
    ) -> t.Set[DataSetMetadataStub]:
        glob_pattern = os.path.join(
            self._path.absolute(), "**", "metadata", f"{dataset_name}_*.meta"
        )
        results = set()
        for file in glob.glob(glob_pattern):
            with open(file, "rt") as fh:
                record = json.load(fh)
            metadata = self._deserialise_meta_data(record)
            if metadata.name == dataset_name and (
                not params
                or all(
                    [
                        metadata.recursively_get_parameter_value(target) == value
                        for target, value in params.items()
                    ]
                )
            ):
                results.add(metadata)
        return results
