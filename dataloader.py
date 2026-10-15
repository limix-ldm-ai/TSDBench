from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset

from dataset import MultiModalDataset
from dataset import read_all_configs
import ipdb


def _time_series_input_specs(task_config: Dict[str, Any]) -> list[Dict[str, Any]]:
    """
    Get time series field specifications, including source, window, history_len, sampling, and field.
    Args:
        - task_config: Dict[str, Any], task configuration.
    Returns:
        - list[Dict[str, Any]]: Time series field specifications, including source, window, history_len, sampling, and field.
    """
    specs: list[Dict[str, Any]] = []
    for item in (task_config.get("inputs") or {}).get("time_series", []):
        source = item.get("source")
        window = item.get("window") or {}
        spec_base = {
            "source": str(source) if source is not None else None,
            "window": window,
            "history_len": _window_length(window),
            "sampling": item.get("sampling") or {},
        }
        for field in _as_list(item.get("fields")):
            specs.append({**spec_base, "field": str(field)})

    if not specs:
        raise ValueError("inputs.time_series must define at least one field")
    return specs


def _meta_time_series_input_specs(task_config: Dict[str, Any]) -> list[Dict[str, Any]]:
    """
    Get time series field specifications for metadata tasks, including source, window, history_len, sampling, and field.
    Args:
        - task_config: Dict[str, Any], task configuration.
    Returns:
        - list[Dict[str, Any]]: Time series field specifications for metadata tasks, including source, window, history_len, sampling, and field.
    """
    specs: list[Dict[str, Any]] = []
    for item in (task_config.get("inputs") or {}).get("time_series", []):
        source = item.get("source")
        window = item.get("window") or {}
        spec_base = {
            "source": str(source) if source is not None else None,
            "window": window,
            "history_len": _window_length(window, default=0),
            "sampling": item.get("sampling") or {},
        }
        for field in _as_list(item.get("fields")):
            specs.append({**spec_base, "field": str(field)})
    return specs


def _partition_time_series_fields(task_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Partition time series fields into historical and future fields.
    Args:
        - task_config: Dict[str, Any], task configuration.
    Returns:
        - Dict[str, Any]: Time series field specifications.
            - target_field: str, prediction target field.
            - target_hist_spec: Dict[str, Any], input specification for the prediction target field.
            - hist_specs: list[Dict[str, Any]], historical field specifications.
            - future_specs: list[Dict[str, Any]], future field specifications.
    """
    specs = _time_series_input_specs(task_config)
    spec_by_field: dict[str, Dict[str, Any]] = {}
    for spec in specs:
        field = spec["field"]
        if field in spec_by_field: # Raise an error for duplicate time series fields.
            raise ValueError(f"Duplicate time series input field: {field!r}")
        spec_by_field[field] = spec

    prediction = task_config.get("prediction") or {}
    target_field = str(prediction.get("field", "traffic_flow"))
    if target_field not in spec_by_field:
        raise ValueError(f"Prediction target field {target_field!r} must also be in inputs.time_series")

    leakage = task_config.get("leakage") or {}
    observed_past_only = {str(field) for field in _as_list(leakage.get("observed_past_only_columns"))} # Get fields whose future values cannot be used as inputs.
    known_future = {str(field) for field in _as_list(leakage.get("known_future_columns"))} # Get fields whose future values can be used as inputs.
    ordered_fields = [spec["field"] for spec in specs]

    hist_specs = [
        spec_by_field[field]
        for field in ordered_fields
        if field in observed_past_only and field != target_field
    ]
    future_specs = [spec_by_field[field] for field in ordered_fields if field in known_future]

    return {
        "target_field": target_field,
        "target_hist_spec": spec_by_field[target_field], # Get the input specification for the prediction target field.
        "hist_specs": hist_specs, # Get historical field specifications.
        "future_specs": future_specs, # Get future field specifications.
    }


def _loaded_time_series_field(
    series_data: Dict[str, Any], field: str, source: str | None) -> Dict[str, Any]:
    """
    Get time series data by source and field; use field as the data source when source is None.
    Args:
        - series_data: Dict[str, Any], time series data.
        - field: str, field name.
        - source: str | None, data source.
    Returns:
        - Dict[str, Any]: Time series data.
    """
    timestamps = pd.DatetimeIndex(pd.to_datetime(series_data["timestamp"]))
    mode = series_data.get("variables_mode")
    if mode == "single":
        values = series_data.get("value") or {}
        if not values:
            raise ValueError(f"Loaded time series source={source!r} has no values")
        return {
            "field": field,
            "source": source,
            "timestamps": timestamps,
            "time_to_pos": {timestamp: idx for idx, timestamp in enumerate(timestamps)},
            "node_scoped": True, # Whether the time series is scoped to individual nodes.
            "values": values,
        }

    values_by_field = series_data.get("value") or {}
    values = values_by_field.get(field)
    if values is None:
        available = sorted(values_by_field)
        raise KeyError(f"Loaded time series source={source!r} is missing field={field!r}; available={available}")

    return {
        "field": field,
        "source": source,
        "timestamps": timestamps,
        "time_to_pos": {timestamp: idx for idx, timestamp in enumerate(timestamps)},
        "node_scoped": True, # Whether the time series is scoped to individual nodes.
        "values": values,
    }


def _time_series_data_ref(task_config: Dict[str, Any], source: str | None) -> Dict[str, Any] | None:
    """
    Get the time series data source configuration by source.
    Args:
        - task_config: Dict[str, Any], task configuration.
        - source: str | None, data source.
    Returns:
        - Dict[str, Any] | None: Time series data source configuration.
            - file: str, file path.
            - value_field: str, value field.
            - native_grid_minutes: int, native time grid interval in minutes.
    """
    if source is None:
        return None

    refs = ((task_config.get("data_refs") or {}).get("time_series") or {})
    if isinstance(refs, dict):
        ref = refs.get(source)
        return ref if isinstance(ref, dict) else None

    for item in _as_list(refs):
        if not isinstance(item, dict) or source not in item:
            continue
        nested = item[source]
        if isinstance(nested, dict):
            return nested
        return {key: value for key, value in item.items() if key != source}
    return None


def _data_ref_path_candidates(dataset_dir: Path, file_name: str) -> list[Path]:
    """
    Get candidate paths for a time series source file by file_name.
    Args:
        - dataset_dir: Path, dataset directory.
        - file_name: str, file name.
    Returns:
        - list[Path]: Candidate paths for the time series source file.
    """
    path = Path(file_name).expanduser()
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([dataset_dir / path, dataset_dir.parent / path])

    parts = path.parts
    if dataset_dir.name in parts:
        protocol_idx = parts.index(dataset_dir.name)
        remainder = Path(*parts[protocol_idx + 1 :])
        candidates.append(dataset_dir / remainder)

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def _resolve_data_ref_path(dataset_dir: Path, file_name: str) -> Path:
    """
    Returns:
        - Path: Path to the time series source file.
    """
    candidates = _data_ref_path_candidates(dataset_dir, file_name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    shown = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Cannot resolve time series data_ref file {file_name!r}; tried: {shown}")


def _load_time_series_frame(path: Path) -> pd.DataFrame:
    """
    Load a time series source file as a pandas DataFrame.
    Returns:
        - pd.DataFrame: Contents of the time series source file.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".tsv", ".txt"):
        return pd.read_csv(path, sep="\t")
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suffix in (".h5", ".hdf5", ".hdf"):
        return pd.read_hdf(path)
    supported = ".csv, .tsv, .txt, .parquet, .pq, .h5, .hdf5, .hdf"
    raise ValueError(
        f"Unsupported time series data_ref file {path}; expected a pandas-readable single-table format "
        f"with one of: {supported}"
    )


def _data_ref_time_series_field(
    dataset: MultiModalDataset,
    task_config: Dict[str, Any],
    *,
    source: str,
    field: str,
) -> Dict[str, Any]:
    """
    Load time series data from the file referenced by data_ref and return the data.
    Args:
        - dataset: MultiModalDataset, containing time series data.
        - task_config: Dict[str, Any], task configuration.
        - source: str, data source.
        - field: str, field name.
    Returns:
        - Dict[str, Any]: Time series data.
    """
    data_ref = _time_series_data_ref(task_config, source)
    if data_ref is None:
        raise KeyError(f"Cannot resolve time series source={source!r} for field={field!r}")

    file_name = data_ref.get("file")
    if not file_name:
        raise ValueError(f"time_series data_ref {source!r} is missing file")

    path = _resolve_data_ref_path(Path(dataset.dataset_dir), str(file_name))
    frame = _load_time_series_frame(path)
    timestamp_col = "timestamp" if "timestamp" in frame.columns else None
    if timestamp_col is None and not isinstance(frame.index, pd.RangeIndex):
        timestamps = pd.DatetimeIndex(pd.to_datetime(frame.index))
    elif timestamp_col is not None:
        timestamps = pd.DatetimeIndex(pd.to_datetime(frame[timestamp_col]))
    else:
        raise ValueError(f"Time series source={source!r} must include a timestamp column or timestamp index")

    value_col = field if field in frame.columns else data_ref.get("value_field")
    if value_col not in frame.columns:
        raise ValueError(f"Time series source={source!r} is missing field={field!r}")

    values = pd.to_numeric(frame[value_col], errors="raise").to_numpy(dtype=np.float32)
    return {
        "field": field,
        "source": source,
        "timestamps": timestamps,
        "time_to_pos": {timestamp: idx for idx, timestamp in enumerate(timestamps)},
        "node_scoped": False, # Whether the time series is scoped to individual nodes; False indicates a global time series.
        "values": values,
    }


def _resolve_or_load_series(
    dataset: MultiModalDataset,
    task_config: Dict[str, Any],
    *,
    source: str | None,
    field: str,
) -> Dict[str, Any]:
    """
    Get time series data by source and field. If field is in dataset.time_series, use dataset.time_series[field]; if source is in dataset.time_series, use dataset.time_series[source]; otherwise, raise an error.
    Args:
        - dataset: MultiModalDataset, containing time series data.
        - task_config: Dict[str, Any], task configuration.
        - source: str | None, data source.
        - field: str, field name.
    Returns:
        - Dict[str, Any]: Time series data.
    """
    if field in dataset.time_series:
        return _loaded_time_series_field(dataset.time_series[field], field, field)
    if source and _time_series_data_ref(task_config, source) is not None:
        return _data_ref_time_series_field(dataset, task_config, source=source, field=field)
    if source and source in dataset.time_series:
        return _loaded_time_series_field(dataset.time_series[source], field, source)
    if len(dataset.time_series) == 1: # If source is not provided, use the only key in dataset.time_series as the data source.
        key = next(iter(dataset.time_series))
        return _loaded_time_series_field(dataset.time_series[key], field, key)

    available = sorted(dataset.time_series)
    raise KeyError(f"Cannot resolve time series field={field!r}, source={source!r}; available={available}")


class TimeSeriesForecastingDataset(Dataset):
    """
    Build a time series forecasting dataset organized around timestamp and sample_id, with one sample for each (timestamp, sample_id) pair.
        1. Get time series data for the prediction target field using target_field. 
        2. Get time series data for historical and future fields using hist_specs and future_specs.
        3. Get static tabular features.
        4. Get graph data.
        5. Filter out invalid timestamps.
        6. Sample time series forecasting windows using decision timestamps.
        7. Return the time series forecasting dataset.
    Sample time series forecasting windows using decision timestamps.
    Args:
        - dataset: MultiModalDataset, containing time series data.
        - task_config: Dict[str, Any], task configuration.
        - timestamps: Iterable[Any], timestamps.
        - split: str, dataset split.
        - graph: Dict[str, Any] | None, graph data.
    Returns:
        - TimeSeriesForecastingDataset: Time series forecasting dataset.
    """

    def __init__(
        self,
        dataset: MultiModalDataset,
        task_config: Dict[str, Any],
        timestamps: Iterable[Any],
        *,
        split: str,
        graph: Dict[str, Any] | None = None,
    ) -> None:

        # 1. Get the input specification for the prediction target field.
        prediction = task_config.get("prediction") or {}
        partition = _partition_time_series_fields(task_config) # Get the time series field partition, including target_hist_spec, hist_specs, and future_specs.
        target_hist_spec = partition["target_hist_spec"] # Get the input specification for the prediction target field.

        self.dataset = dataset
        self.task_config = task_config
        self.split = split
        self.target_field = partition["target_field"]
        self.target_hist_spec = target_hist_spec
        self.hist_specs = partition["hist_specs"]
        self.future_specs = partition["future_specs"]
        self.hist_fields = [spec["field"] for spec in self.hist_specs]
        self.future_fields = [spec["field"] for spec in self.future_specs]
        self.input_len = int(target_hist_spec["history_len"])
        self.horizon = _window_length((prediction.get("horizon") or {}))

        target_source = prediction.get("source") or target_hist_spec.get("source") # Get the data source for the prediction target field.

        # 2. Get time series data for the prediction target field.
        self._target_series = _resolve_or_load_series(
            dataset,
            task_config,
            source=str(target_source) if target_source is not None else None,
            field=self.target_field,
        )
        if not self._target_series["node_scoped"]:
            raise ValueError("Forecasting target must be node-scoped, rather than global time series")

        self.timestamps = self._target_series["timestamps"] # Get timestamps for the prediction target field.
        self._time_to_pos = {timestamp: idx for idx, timestamp in enumerate(self.timestamps)}
        # Map target-field timestamps to indices to look up their offsets later.
        self.node_ids = list((self._target_series.get("values") or {}).keys()) # Get all node IDs for the prediction target field.
        if not self.node_ids:
            raise ValueError("Forecasting target source has no node values")

        # 3. Get time series data for covariates.
        self._series_by_field: dict[str, Dict[str, Any]] = {} # Store time series data keyed by field name.
        for spec in [*self.hist_specs, *self.future_specs]:
            self._series_by_field[spec["field"]] = _resolve_or_load_series(
                dataset,
                task_config,
                source=spec.get("source"),
                field=spec["field"],
            )
        self._series_available_len_cache: dict[int, int] = {} # Cache time series lengths, keyed by the ID of each time series data object.
        self._series_matrix_cache: dict[int, np.ndarray] = {}
        self._series_alignment_cache: dict[int, bool] = {}

        # 4. Get static tabular features.
        self.static_specs = _static_input_specs(task_config)
        self.static_fields = [spec["field"] for spec in self.static_specs]
        self.static_features, self.missing_static_fields = _build_static_features(
            dataset,
            self.node_ids,
            self.static_specs,
        )

        # 5. Get graph data.
        self.graph = graph

        # 6. Filter out invalid timestamps.
        self.samples = self._filter_valid_timestamps(timestamps)
    
    def _series_uses_target_positions(self, series: Dict[str, Any]) -> bool:
        cache_key = id(series)
        cached = self._series_alignment_cache.get(cache_key)
        if cached is not None:
            return cached

        series_timestamps = series.get("timestamps")
        aligned = bool(
            series_timestamps is self.timestamps
            or (
                isinstance(series_timestamps, pd.DatetimeIndex)
                and len(series_timestamps) == len(self.timestamps)
                and series_timestamps.equals(self.timestamps)
            )
        )
        self._series_alignment_cache[cache_key] = aligned
        return aligned

    def _series_available_len(self, series: Dict[str, Any]) -> int:
        """
        Get the available length of the time series data.
        Returns:
            - int: Available length.
        """
        cache_key = id(series)
        cached = self._series_available_len_cache.get(cache_key)
        if cached is not None:
            return cached

        values = series["values"]
        if series["node_scoped"]:
            lengths = [len(values[node_id]) for node_id in self.node_ids if node_id in values]
            available_len = min(lengths) if len(lengths) == len(self.node_ids) else 0
        else:
            available_len = len(values)

        self._series_available_len_cache[cache_key] = available_len
        return available_len

    def _has_series_window(self, series: Dict[str, Any], start: int, end: int) -> bool:
        """
        Check whether a window is available in the time series data.
        Returns:
            - bool: Whether the window is available.
        """
        if start < 0 or end > len(self.timestamps):
            return False
        if self._series_uses_target_positions(series):
            return end <= self._series_available_len(series)

        positions = self._series_positions(series, start, end)
        if positions is None:
            return False
        if not positions:
            return True

        max_pos = max(positions)
        return max_pos < self._series_available_len(series)
    
    def _has_field_window(self, spec: Dict[str, Any], start: int, end: int) -> bool:
        return self._has_series_window(self._series_by_field[spec["field"]], start, end)

    def _window_bounds(self, pos: int) -> tuple[int, int, int, int]:
        """Return left-closed, right-open history and future window bounds."""
        return (
            pos - self.input_len + 1,
            pos + 1,
            pos + 1,
            pos + 1 + self.horizon,
        )

    def _filter_valid_timestamps(self, timestamps: Iterable[Any]) -> list[tuple[pd.Timestamp, int]]:
        """
        Filter out invalid timestamps.
        Returns:
            - list[tuple[pd.Timestamp, int]]: Valid timestamps and their corresponding indices.
        """
        valid: list[tuple[pd.Timestamp, int]] = []
        for raw_timestamp in timestamps:
            timestamp = pd.Timestamp(raw_timestamp)
            pos = self._time_to_pos.get(timestamp)
            if pos is None:
                continue
            hist_start, hist_end, future_start, future_end = self._window_bounds(pos)

            if not self._has_series_window(self._target_series, hist_start, hist_end): # Check whether historical time series data is available for the prediction target field.
                continue
            if not self._has_series_window(self._target_series, future_start, future_end): # Check whether future time series data is available for the prediction target field.
                continue
            if any(not self._has_field_window(spec, hist_start, hist_end) for spec in self.hist_specs): # Check whether historical fields are available.
                continue
            if any(not self._has_field_window(spec, future_start, future_end) for spec in self.future_specs): # Check whether future fields are available.
                continue
            valid.append((timestamp, pos))
        return valid

    def __len__(self) -> int:
        return len(self.samples)

    def _series_positions(self, series: Dict[str, Any], start: int, end: int) -> list[int] | None:
        if self._series_uses_target_positions(series):
            return list(range(start, end))
        positions: list[int] = []
        for timestamp in self.timestamps[start:end]:
            pos = series["time_to_pos"].get(timestamp)
            if pos is None:
                return None
            positions.append(pos)
        return positions

    def _series_matrix(self, series: Dict[str, Any]) -> np.ndarray:
        cache_key = id(series)
        cached = self._series_matrix_cache.get(cache_key)
        if cached is not None:
            return cached

        if series["node_scoped"]:
            values = series["values"]
            matrix = np.stack(
                [np.asarray(values[node_id], dtype=np.float32) for node_id in self.node_ids],
                axis=0,
            )
        else:
            matrix = np.asarray(series["values"], dtype=np.float32)

        self._series_matrix_cache[cache_key] = matrix
        return matrix

    def _series_window(self, series: Dict[str, Any], start: int, end: int) -> np.ndarray:
        """
        Get a time series window from series and return it as an np.ndarray.
        Args:
            - series: Dict[str, Any], time series data.
            - start: int, window start index.
            - end: int, window end index.
        Returns:
            - np.ndarray: Time series window data, with shape (len(node_ids), window_len, 1).
        """
        matrix = self._series_matrix(series)
        if self._series_uses_target_positions(series):
            if series["node_scoped"]:
                array = matrix[:, start:end]
            else:
                values = matrix[start:end]
                array = np.broadcast_to(values, (len(self.node_ids), end - start)).copy()
            return array[:, None, :]

        positions = self._series_positions(series, start, end)
        if positions is None:
            raise IndexError(f"Time series source={series.get('source')!r} is missing requested timestamps")

        positions_array = np.asarray(positions, dtype=np.int64)
        is_contiguous = bool(
            len(positions) > 0
            and positions[-1] - positions[0] == len(positions) - 1
            and np.all(np.diff(positions_array) == 1)
        )
        if series["node_scoped"]:
            if is_contiguous:
                array = matrix[:, positions[0] : positions[0] + len(positions)]
            else:
                array = matrix[:, positions_array]
        else:
            values = matrix[positions[0] : positions[0] + len(positions)] if is_contiguous else matrix[positions_array]
            array = np.broadcast_to(values, (len(self.node_ids), len(positions))).copy()

        return array[:, None, :]

    def _window_for_specs(self, specs: list[Dict[str, Any]], start: int, end: int) -> torch.Tensor:
        """
        Get time series window data for specs and return it as a torch.Tensor.
        Args:
            - specs: list[Dict[str, Any]], time series field specifications.
            - start: int, window start index.
            - end: int, window end index.
        Returns:
            - torch.Tensor: Time series window data, with shape (len(node_ids), len(specs), window_len, 1).
                
        """
        window_len = end - start
        if not specs:
            return torch.empty((len(self.node_ids), 0, window_len), dtype=torch.float32)

        arrays = [
            self._series_window(self._series_by_field[spec["field"]], start, end)
            for spec in specs
        ]
        return torch.from_numpy(np.concatenate(arrays, axis=1).astype(np.float32, copy=False))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get sample data.
        Args:
            - idx: int, sample index.
        Returns:
            - Dict[str, Any]: Sample data.
                - x_hist: torch.Tensor, historical time series data, with shape (len(node_ids), input_len, 1).
                - x_target_hist: torch.Tensor, historical time series data for the prediction target field, with shape (len(node_ids), input_len, 1).
                - x_future_known: torch.Tensor, known future covariates, with shape (len(known_future_fields), horizon).
                - static_features: torch.Tensor, static feature data, with shape (len(node_ids), len(static_fields)).
                - graph: Dict[str, Any], graph data.
                - y: torch.Tensor, target time series data, with shape (len(node_ids), horizon, 1).
                - timestamp: str, timestamp.
                - hist_fields: list[str], historical field names.
                - future_fields: list[str], future field names.
                - target_field: str, prediction target field name.
        """
        timestamp, pos = self.samples[idx] # Map the sample index to a timestamp and its time-axis index.
        hist_start, hist_end, future_start, future_end = self._window_bounds(pos)

        return {
            "x_hist": self._window_for_specs(self.hist_specs, hist_start, hist_end), # Historical time series data.
            "x_target_hist": torch.from_numpy(
                self._series_window(self._target_series, hist_start, hist_end)
            ), # Historical time series data for the prediction target field.
            "x_future_known": self._window_for_specs(self.future_specs, future_start, future_end), # Known future covariates.
            "static_features": self.static_features,
            "graph": self.graph,
            "y": torch.from_numpy(self._series_window(self._target_series, future_start, future_end)),
            "timestamp": timestamp.isoformat(),
            "hist_fields": self.hist_fields,
            "future_fields": self.future_fields,
            "target_field": self.target_field,
        }


def _static_input_specs(task_config: Dict[str, Any]) -> list[Dict[str, str]]:
    """
    Get static input field specifications.

    Two forms are supported:
      - "Length": both the logical field and the source column are Length.
      - {"field": "Latitude", "source_field": "Lat"}: output the field as Latitude and read the source column Lat.

    Args:
        - task_config: Dict[str, Any], task configuration.
    Returns:
        - list[Dict[str, str]]: Static input field specifications.
    """
    inputs = task_config.get("inputs") or {}
    target_field = str((task_config.get("prediction") or {}).get("field", ""))
    forbidden = {str(field) for field in _as_list((task_config.get("leakage") or {}).get("forbidden_input_columns"))}
    forbidden.add(target_field)

    specs: list[Dict[str, str]] = []
    for raw_field in _as_list((inputs.get("tabular_static") or {}).get("fields")):
        if isinstance(raw_field, dict):
            if "field" not in raw_field:
                raise ValueError(f"tabular_static field mapping must include 'field': {raw_field!r}")
            field_name = str(raw_field["field"])
            raw_source_field = raw_field.get("source_field", field_name)
            source_field = field_name if raw_source_field is None else str(raw_source_field)
        else:
            field_name = str(raw_field)
            source_field = field_name

        if field_name not in forbidden and source_field not in forbidden:
            specs.append({"field": field_name, "source_field": source_field})
    return specs


class MetaFeatureTaskDataset(Dataset):
    """
    Build a metadata feature dataset organized around sample_id, with one sample for each sample_id.
        1. Get sample data by sample_id, then build the metadata feature dataset from meta_features.
    Args:
        - dataset: MultiModalDataset, containing meta_features.
        - task_config: Dict[str, Any], task configuration.
        - split_keys: dict[str, list[Any]], split keys.
        - split: str, dataset split.
        - graph: Dict[str, Any] | None, graph data.
    Returns:
        - MetaFeatureTaskDataset: Metadata feature dataset.
    """
    def __init__(
        self,
        dataset: MultiModalDataset,
        task_config: Dict[str, Any],
        sample_ids: Iterable[Any],
        *,
        split: str,
        graph: Dict[str, Any] | None = None,
    ) -> None:
        features = dataset.meta_features.get("features", {}) 
        primary_key = _sample_primary_key(task_config)
        prediction = task_config.get("prediction") or {}
        target_field = str(prediction.get("field", "")) #
        task_type = _task_type(task_config) # Get the task type.

        if primary_key not in features:
            raise ValueError(f"Meta features are missing sample primary key column {primary_key!r}")
        if target_field not in features:
            raise ValueError(f"Meta features are missing prediction target column {target_field!r}")
        
        self.dataset = dataset
        self.task_config = task_config
        self.split = split
        self.primary_key = primary_key
        self.target_field = target_field
        self.graph = graph
        self.is_classification = "classification" in task_type #
        
        self._row_by_sample_id = {str(value): idx for idx, value in enumerate(features[primary_key])}
        self.samples = [sample_id for sample_id in sample_ids if str(sample_id) in self._row_by_sample_id]
        self.static_specs = _static_input_specs(task_config)
        self.static_fields = [spec["field"] for spec in self.static_specs]
        self.x_static, self.missing_static_fields = self._build_static_matrix(features)
        self.y_values, self.class_to_idx = self._build_targets(features[target_field])
        self.time_series_specs = _meta_time_series_input_specs(task_config)
        self.hist_fields = [spec["field"] for spec in self.time_series_specs]
        self._series_for_specs = [
            _resolve_or_load_series(
                dataset,
                task_config,
                source=spec.get("source"),
                field=spec["field"],
            )
            for spec in self.time_series_specs
        ]
        self._history = self._build_time_series_lookup()

    def _build_static_matrix(self, features: Dict[str, Any]) -> tuple[torch.Tensor, list[str]]:
        """
        Build the static feature matrix and return it along with the names of missing static fields.
        Args:
            - features: Dict[str, Any], data containing static features.
        Returns:
            - tuple[torch.Tensor, list[str]]: Static feature matrix and names of missing static fields.
        """
        columns: list[np.ndarray] = []
        missing_fields: list[str] = []

        for spec in self.static_specs:
            actual_field = _resolve_static_field(features, spec)
            if actual_field is None:
                missing_fields.append(spec["field"])
                continue
            columns.append(_coerce_static_column(features[actual_field]))

        if not columns:
            return torch.empty((len(features[self.primary_key]), 0), dtype=torch.float32), missing_fields
        return torch.from_numpy(np.stack(columns, axis=1).astype(np.float32)), missing_fields
    
    def _build_targets(self, target_values: Iterable[Any]) -> tuple[torch.Tensor, dict[str, int] | None]:
        if self.is_classification:
            series = pd.Series(list(target_values)).astype(str)
            class_to_idx = {value: idx for idx, value in enumerate(sorted(series.dropna().unique()))}
            encoded = series.map(class_to_idx).to_numpy(dtype=np.int64).copy()
            return torch.from_numpy(encoded), class_to_idx

        numeric = pd.to_numeric(pd.Series(list(target_values)), errors="coerce").to_numpy(dtype=np.float32).copy()
        return torch.from_numpy(numeric), None

    def _build_time_series_lookup(self) -> Dict[str, Any] | None:
        """
        Build a time series lookup that maps sample_id to time series data.
        Args:
            - None
        Returns:
            - Dict[str, Any] | None: Time series lookup.
                - input_len: int, historical time series length.
                - lookups: list[Dict[str, Any]]: Time series lookups, including spec, series, and sample_to_series_key.
                    - spec: Dict[str, Any], time series configuration keyed by field, including source, window, history_len, and sampling.
                    - series: Dict[str, Any], time series data keyed by time series data ID.
                    - sample_to_series_key: dict[str, Any], mapping from sample IDs to time series data IDs.
                    - start_pos: int, position corresponding to window.start_time; 0 when not configured.
        window.start_time applies to tabular classification/regression tasks, includes the specified time, and must exactly match a data timestamp.
        """

        if not self.time_series_specs:
            return None

        history_lens = {int(spec["history_len"]) for spec in self.time_series_specs}
        if len(history_lens) > 1:
            by_field = {spec["field"]: int(spec["history_len"]) for spec in self.time_series_specs}
            raise ValueError(
                "Meta time_series inputs must share one window length to build x_hist; "
                f"got {by_field}"
            )

        features = self.dataset.meta_features.get("features", {})
        time_id_col = self.dataset.meta_features.get("sample_id_time_series")
        graph_id_col = self.dataset.meta_features.get("sample_id_graph")
        lookups: list[Dict[str, Any]] = []

        # Iterate over time series specifications and data to build the time series lookup.
        for spec, series in zip(self.time_series_specs, self._series_for_specs):
            if not series["node_scoped"]:
                raise ValueError(
                    "Meta time_series inputs must be node-scoped because meta samples do not "
                    f"provide a timestamp anchor; field={spec['field']!r}, source={spec.get('source')!r}"
                )

            # Optional absolute start for a fixed history window in a tabular task.
            start_time = spec["window"].get("start_time")
            start_pos = 0
            if start_time is not None:
                context = f"field={spec['field']!r}, source={spec.get('source')!r}"
                try:
                    start_timestamp = pd.Timestamp(start_time)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(f"Invalid window.start_time={start_time!r}; {context}") from exc
                if pd.isna(start_timestamp):
                    raise ValueError(f"Invalid window.start_time={start_time!r}; {context}")

                timestamps = series["timestamps"]
                if not timestamps.is_unique or not timestamps.is_monotonic_increasing:
                    raise ValueError(f"window.start_time requires unique, increasing timestamps; {context}")
                start_pos = series["time_to_pos"].get(start_timestamp)
                if start_pos is None:
                    raise ValueError(
                        f"window.start_time={start_time!r} must exactly match a series timestamp; {context}"
                    )
                history_len = int(spec["history_len"])
                if history_len <= 0 or start_pos + history_len > len(timestamps):
                    raise ValueError(
                        f"Cannot read window.length={history_len} from window.start_time={start_time!r}; "
                        f"available time steps={len(timestamps) - start_pos}; {context}"
                    )

            values = series.get("values") or {}
            key_lookup = {str(key): key for key in values}
            sample_to_series_key: dict[str, Any] = {}

            for row_idx, primary_value in enumerate(features[self.primary_key]):
                candidates = [primary_value]
                if time_id_col and time_id_col in features:
                    candidates.append(features[time_id_col][row_idx])
                if graph_id_col and graph_id_col in features:
                    candidates.append(features[graph_id_col][row_idx])

                for candidate in candidates:
                    series_key = key_lookup.get(str(candidate))
                    if series_key is not None:
                        sample_to_series_key[str(primary_value)] = series_key
                        break

            lookups.append(
                {
                    "spec": spec,
                    "series": series,
                    "sample_to_series_key": sample_to_series_key,
                    "start_pos": start_pos,
                }
            )

        input_len = next(iter(history_lens), 0)
        if any(spec["window"].get("start_time") is not None for spec in self.time_series_specs):
            windows = [
                lookup["series"]["timestamps"][lookup["start_pos"] : lookup["start_pos"] + input_len]
                for lookup in lookups
            ]
            if any(not window.equals(windows[0]) for window in windows[1:]):
                raise ValueError(
                    "Meta time_series windows must cover the same timestamps to build x_hist; "
                    "set window.start_time consistently for all input fields"
                )

        return {
            "input_len": input_len,
            "lookups": lookups,
        }

    def _history_window_for_sample(self, sample_id: Any) -> torch.Tensor | None:
        """
        Get historical time series data for a sample.
        Args:
            - sample_id: Any, sample ID.
        Returns:
            - torch.Tensor | None: Historical time series data, with shape (input_len, len(hist_fields)).
        """
        if self._history is None: # Return None if the time series lookup is empty.
            return None

        input_len = int(self._history["input_len"])
        if input_len <= 0:
            return None

        columns: list[np.ndarray] = []
        for lookup in self._history["lookups"]: # Iterate over time series lookups to get historical data for the sample.
            series_key = lookup["sample_to_series_key"].get(str(sample_id)) 
            if series_key is None:
                return None

            values = np.asarray(lookup["series"]["values"][series_key], dtype=np.float32)
            start_pos = lookup["start_pos"]
            end_pos = start_pos + input_len
            if len(values) < end_pos:
                if lookup["spec"]["window"].get("start_time") is not None:
                    raise ValueError(
                        f"Insufficient history for sample_id={sample_id!r}, "
                        f"field={lookup['spec']['field']!r}: need {end_pos} values, got {len(values)}"
                    )
                return None
            columns.append(values[start_pos:end_pos])

        if not columns:
            return None
        return torch.from_numpy(np.stack(columns, axis=1).astype(np.float32, copy=False))

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_id = self.samples[idx]
        row_idx = self._row_by_sample_id[str(sample_id)]
        item: Dict[str, Any] = {
            "x_static": self.x_static[row_idx], # Static features, with shape (len(static_fields),).
            "y": self.y_values[row_idx], # Target value, with shape (1,).
            "sample_id": sample_id, # Sample ID.
            "graph": self.graph, # Graph data.
        }
        # Get historical time series data.
        x_hist = self._history_window_for_sample(sample_id) 
        if x_hist is not None:
            item["x_hist"] = x_hist # Historical time series data, with shape (input_len, len(hist_fields)).
            item["hist_fields"] = self.hist_fields # Historical time series field names, with shape (len(hist_fields),).
        return item

def _window_length(window_config: Dict[str, Any], default: int | None = None) -> int:
    """Get the window length."""
    length = window_config.get("length", default)
    if length is None:
        raise ValueError("window length is required for traffic forecasting")
    return int(length)


def _future_known_fields(task_config: Dict[str, Any]) -> list[str]:
    """Get the names of known future covariates."""
    known = set(_as_list((task_config.get("leakage") or {}).get("known_future_columns")))
    fields: list[str] = []
    for item in task_config.get("inputs", {}).get("time_series", []):
        for field in _as_list(item.get("fields")):
            if field in known:
                fields.append(str(field))
    return fields


def _build_static_features(
    dataset: MultiModalDataset,
    node_ids: list[Any],
    static_specs: list[Dict[str, str]],
) -> tuple[torch.Tensor, list[str]]:
    """
    Args:
        - dataset: MultiModalDataset, containing meta_features.
        - node_ids: list[Any], list of node IDs.
        - static_fields: list[str], list of static feature field names.
    Build static features and return: tuple[torch.Tensor, list[str]]
        - torch.Tensor: Static feature tensor, with shape (len(node_ids), len(static_fields)).
        - list[str]: Names of static fields required by task_config but missing from the dataset's meta_features.
    """
    features = dataset.meta_features.get("features", {})
    id_column = dataset.meta_features.get("sample_id_graph") # The dataset is aligned using sample_id_graph by default, so id_column is sample_id_graph.
    missing_fields: list[str] = []

    if not features or not id_column or id_column not in features:
        # If meta_features has no features or id_column, or id_column is absent from features, return an empty tensor and the missing static field names.
        return torch.empty((len(node_ids), 0), dtype=torch.float32), [spec["field"] for spec in static_specs]
    # Build id_to_row to map id_column values to row indices in features.
    id_to_row = {str(value): idx for idx, value in enumerate(features[id_column])}
    columns: list[np.ndarray] = []

    for spec in static_specs:
        actual_field = _resolve_static_field(features, spec)
        if actual_field is None:
            missing_fields.append(spec["field"])
            continue
        # Convert a static feature field to numeric values if it exists in features.
        full_column = _coerce_static_column(features[actual_field])
        aligned = np.full(len(node_ids), np.nan, dtype=np.float32)
        # Align static feature values with node_ids; full_column follows the order of features[id_column].
        for idx, node_id in enumerate(node_ids):
            row_idx = id_to_row.get(str(node_id))
            if row_idx is not None:
                aligned[idx] = full_column[row_idx]
        columns.append(aligned)

    if not columns:
        return torch.empty((len(node_ids), 0), dtype=torch.float32), missing_fields
    return torch.from_numpy(np.stack(columns, axis=1).astype(np.float32)), missing_fields


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _load_task_config(task_config: str | Path | Dict[str, Any]) -> tuple[Dict[str, Any], Path | None]:
    """
    Load task_config and return task_config and task_config_path.
    task_config: Path to task.yaml or a dict.
    return: tuple[Dict[str, Any], Path | None]
        - task_config: Contents of task.yaml.
        - task_config_path: Path to task.yaml.
    """
    if isinstance(task_config, dict):
        return task_config, None

    path = Path(task_config).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f), path


def _task_type(task_config: Dict[str, Any]) -> str:
    """
    Get task_type from task_config to identify the task type:
    - episode_classification: Classification task.
    - episode_regression: Regression task.
    - time_series_forecasting: Time series forecasting task.
    """
    task_block = task_config.get("task") or {}
    return str(task_block.get("type", task_config.get("type", "")))


def _task_id(task_config: Dict[str, Any]) -> str | None:
    task_block = task_config.get("task") or {}
    task_id = task_block.get("id", task_config.get("id"))
    return str(task_id) if task_id else None


def _split_path_and_column(
    task_config: Dict[str, Any],
    task_config_path: Path | None,
    dataset_dir: Path,
    split_name: str | Path | None,
) -> tuple[Path, str]:
    '''Return the path and the field name used to split the dataset.'''
    task_dir = task_config_path.parent if task_config_path else dataset_dir / "tasks" / str(_task_id(task_config))
    split_column = "split"

    if split_name is None or str(split_name) == "split":
        return task_dir / "splits.csv", split_column

    split_candidate = Path(split_name).expanduser()
    if split_candidate.suffix == ".csv" or split_candidate.exists():
        split_path = split_candidate if split_candidate.is_absolute() else task_dir / split_candidate
        return split_path.resolve(), split_column

    return task_dir / "splits.csv", str(split_name)


def _sample_primary_key(task_config: Dict[str, Any]) -> str:
    '''Return the sample primary-key field name.'''
    sample_block = task_config.get("sample") or {}
    primary_key = sample_block.get("primary_key", "timestamp")
    return str(primary_key)


def _normalize_split_name(split: Any) -> str:
    value = str(split).strip().lower()
    return {
        "training": "train",
        "validation": "valid",
        "val": "valid",
        "dev": "valid",
        "testing": "test",
    }.get(value, value)


def _read_split_keys(split_path: Path, split_column: str, primary_key: str) -> dict[str, list[Any]]:
    '''Read the split file and return split keys as dict[str, list[Any]].'''
    splits = pd.read_csv(split_path)
    required = {primary_key, split_column}
    missing = required - set(splits.columns)
    if missing:
        raise ValueError(f"Split file {split_path} is missing columns: {sorted(missing)}")

    grouped = {"train": [], "valid": [], "test": []}
    for _, row in splits.iterrows():
        split = _normalize_split_name(row[split_column])
        if split in grouped:
            grouped[split].append(row[primary_key])
    return grouped


def _resolve_static_field(features: Dict[str, Any], spec: Dict[str, str]) -> str | None:
    source_field = spec["source_field"]
    return source_field if source_field in features else None


def _coerce_static_column(values: Iterable[Any]) -> np.ndarray:
    """
    Convert static feature values to numeric values.
    Args:
        - values: Iterable[Any], static feature values.
    Returns:
        - np.ndarray: Numeric static feature values.
    """
    series = pd.Series(list(values))
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.to_numpy(dtype=np.float32)

    categories = {value: idx for idx, value in enumerate(sorted(series.dropna().astype(str).unique()))}
    encoded = series.astype(str).map(categories)
    return encoded.to_numpy(dtype=np.float32)


def _resolve_graph(dataset: MultiModalDataset, source: str | None) -> Dict[str, Any] | None:
    if not dataset.graphs:
        return None
    if source:
        if source in dataset.graphs:
            return dataset.graphs[source]
        available = sorted(dataset.graphs)
        raise KeyError(f"Cannot resolve graph source={source!r}; available={available}")
    if len(dataset.graphs) == 1:
        return next(iter(dataset.graphs.values()))
    available = sorted(dataset.graphs)
    raise KeyError(f"Cannot resolve graph source={source!r}; available={available}")


def _build_forecasting_datasets(
    dataset: MultiModalDataset,
    task_config: Dict[str, Any],
    split_keys: dict[str, list[Any]],
    graph: Dict[str, Any] | None,
) -> list[Dataset]:
    '''Build time series forecasting datasets.'''
    return [
        TimeSeriesForecastingDataset(dataset, task_config, split_keys["train"], split="train", graph=graph),
        TimeSeriesForecastingDataset(dataset, task_config, split_keys["valid"], split="valid", graph=graph),
        TimeSeriesForecastingDataset(dataset, task_config, split_keys["test"], split="test", graph=graph),
    ]


def _build_meta_feature_datasets(
    dataset: MultiModalDataset,
    task_config: Dict[str, Any],
    split_keys: dict[str, list[Any]],
    graph: Dict[str, Any] | None,
) -> list[Dataset]:
    return [
        MetaFeatureTaskDataset(dataset, task_config, split_keys["train"], split="train", graph=graph),
        MetaFeatureTaskDataset(dataset, task_config, split_keys["valid"], split="valid", graph=graph),
        MetaFeatureTaskDataset(dataset, task_config, split_keys["test"], split="test", graph=graph),
    ]


def get_multi_modal_datalset(
    dataset: MultiModalDataset, 
    task_config: str | Path | Dict[str, Any],
    split_name: str | Path | None = None,
) -> List[Dataset]:
    """
    Build downstream train/valid/test datasets from MultiModalDataset using task_config and split_name.
    """
    # 1. Load task_config.
    task_cfg, task_config_path = _load_task_config(task_config)
    task_type = _task_type(task_cfg)
    # 2. Get split_path and split_column to identify the split file and column used to build the datasets.
    split_path, split_column = _split_path_and_column(
        task_cfg,
        task_config_path,
        Path(dataset.dataset_dir),
        split_name,
    )
    # 3. Read the split file to get split keys.
    split_keys = _read_split_keys(split_path, split_column, _sample_primary_key(task_cfg))
    # 4. Get graph data.
    graph_source = ((task_cfg.get("inputs") or {}).get("graph") or {}).get("source")
    graph = _resolve_graph(dataset, graph_source)
    # 5. Build the datasets.
    if "forecasting" in task_type:
        # Time series forecasting task: build time series forecasting datasets.
        return _build_forecasting_datasets(dataset, task_cfg, split_keys, graph)
    if "classification" in task_type or "regression" in task_type:
        # Classification or regression task: build classification or regression datasets.
        return _build_meta_feature_datasets(dataset, task_cfg, split_keys, graph)
    raise NotImplementedError(f"Unsupported task type: {task_type}")

if __name__ == "__main__":
    protocol_dir = Path(__file__).resolve().parent
    dataset_yaml_path = protocol_dir / "dataset.yaml"
    configs = read_all_configs(str(dataset_yaml_path))
    dataset = MultiModalDataset(configs, protocol_dir)
    task_config = "tasks/highway_category_classification/task.yaml"
    split_name = "split"
    datasets = get_multi_modal_datalset(dataset, task_config, split_name)
    ipdb.set_trace()
    
    task_config = "tasks/traffic_flow_forecasting/task.yaml"
    split_name = "split"
    datasets = get_multi_modal_datalset(dataset, task_config, split_name)
    ipdb.set_trace()