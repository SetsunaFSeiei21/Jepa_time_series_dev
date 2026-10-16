"""Convert raw datasets into the unified on-disk format.

Unified layout, one directory per dataset (<out>/<NAME>/):
    data.npy            (T, N, C) float32, missing entries filled with 0
    mask.npy            (T, N, C) bool, True = missing
    time_features.npy   (T, F) float32 in [0, 1); absent when timestamps are unknown
    graph.npy           (N, N) float32 raw adjacency; absent for datasets without a graph
    nodes.csv           per-node metadata (ids, coordinates, ...); only for sources that provide it
    meta.json           facts about the data (shape, freq, time feature sizes, ...)

Experiment protocol (split ratios, normalization, metric space) is not stored here;
it lives in configs/dataset/<name>.yaml.

Supported sources:
    stfm     <root>/<NAME>/{data,mask,adj_mtx}.npy + dataset_config.json, data shaped (T, N)
    basicts  <root>/<NAME>/{train,val,test}_{data,timestamps}.npy + meta.json, data shaped (T, N)
    webtris  <root>/<NAME>/{fetch_config.json, sites.json, reports/*.json.gz} written by
             tools/fetch_webtris.py; 15-minute total volume of every site, no graph. Also writes
             <NAME>-Link, which keeps one site per link (road segment between junctions)
    coops    <root>/<NAME>/{fetch_config.json, stations.json, data/*.csv} written by
             tools/fetch_coops.py; hourly non-tidal residual (observed water level minus tide
             prediction), no graph

Usage:
    python tools/convert_datasets.py --stfm-root ~/STFM-datasets \
        --basicts-root ~/time-series-dataset --webtris-root ~/webtris-raw \
        --coops-root ~/coops-raw --out ~/datasets
"""

import argparse
import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

MINUTES_PER_DAY = 24 * 60
DAYS_PER_WEEK = 7
MAX_DAYS_PER_MONTH = 31
MAX_DAYS_PER_YEAR = 366

# Calendar features scaled to [0, 1); identical to the BasicTS encoding.
TIME_FEATURES = {
    "time_of_day": lambda ts: (ts.hour * 60 + ts.minute) / MINUTES_PER_DAY,
    "day_of_week": lambda ts: ts.dayofweek / DAYS_PER_WEEK,
    "day_of_month": lambda ts: (ts.day - 1) / MAX_DAYS_PER_MONTH,
    "day_of_year": lambda ts: (ts.dayofyear - 1) / MAX_DAYS_PER_YEAR,
}
BASICTS_TIME_FEATURES = ["time of day", "day of week", "day of month", "day of year"]
# Start times documented by BasicTS; time features of these datasets are recomputed because the
# shipped ETTm2 timestamps repeat day_of_month in the day_of_year column.
BASICTS_STARTS = {name: "2016-07-01 00:00" for name in ("ETTh1", "ETTh2", "ETTm1", "ETTm2")}
# Tolerance for `feature * size` to count as an integer index (float32 rounding).
INDEX_TOLERANCE = 1e-3
SPLITS = ("train", "val", "test")
# Sites/stations of the API sources with more missing steps than this are dropped (mostly
# decommissioned loop detectors or gauges that were out of service).
MAX_SITE_MISSING = 0.3
WEBTRIS_FREQ = "15min"
COOPS_FREQ = "1h"


def time_feature_sizes(freq: str) -> dict[str, int]:
    """Number of distinct values of every time feature; `feature * size` is an integer index."""
    minutes = pd.Timedelta(freq).total_seconds() / 60
    return {
        "time_of_day": max(1, round(MINUTES_PER_DAY / minutes)),
        "day_of_week": DAYS_PER_WEEK,
        "day_of_month": MAX_DAYS_PER_MONTH,
        "day_of_year": MAX_DAYS_PER_YEAR,
    }


def time_features(timestamps: pd.DatetimeIndex) -> np.ndarray:
    return np.stack(
        [np.asarray(fn(timestamps), dtype=np.float32) for fn in TIME_FEATURES.values()],
        axis=-1,
    )


def save(
    out_dir: Path,
    name: str,
    source: str,
    data: np.ndarray,
    mask: np.ndarray,
    time_feat: np.ndarray | None,
    graph: np.ndarray | None,
    freq: str | None,
    start: str | None,
    nodes: pd.DataFrame | None = None,
) -> None:
    """Write one dataset; `data` and `mask` are (T, N) or (T, N, C)."""
    if data.ndim == 2:
        data, mask = data[..., None], mask[..., None]
    data = np.where(mask, 0.0, data).astype(np.float32)
    num_steps, num_nodes, num_channels = data.shape
    assert mask.shape == data.shape, (mask.shape, data.shape)
    assert time_feat is None or time_feat.shape[0] == num_steps
    assert graph is None or graph.shape == (num_nodes, num_nodes)
    assert nodes is None or len(nodes) == num_nodes

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "data.npy", data)
    np.save(out_dir / "mask.npy", mask.astype(bool))
    if time_feat is not None:
        index = time_feat * np.array(list(time_feature_sizes(freq).values()))
        assert np.abs(index - np.round(index)).max() < INDEX_TOLERANCE, (
            f"{name}: off-grid time features"
        )
        np.save(out_dir / "time_features.npy", time_feat.astype(np.float32))
    if graph is not None:
        np.save(out_dir / "graph.npy", graph.astype(np.float32))
    if nodes is not None:
        nodes.to_csv(out_dir / "nodes.csv", index=False)

    meta = {
        "name": name,
        "source": source,
        "num_steps": num_steps,
        "num_nodes": num_nodes,
        "num_channels": num_channels,
        "freq": freq,
        "start": start,
        # name -> number of distinct values, in the column order of time_features.npy
        "time_features": time_feature_sizes(freq) if time_feat is not None else {},
        "has_graph": graph is not None,
        "missing_rate": float(mask.mean()),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(
        f"[{source}] {name}: T={num_steps} N={num_nodes} C={num_channels} "
        f"missing={meta['missing_rate']:.4f} graph={graph is not None}"
    )


def convert_stfm(src: Path, out_root: Path) -> None:
    with open(src / "dataset_config.json") as f:
        config = json.load(f)
    data = np.load(src / "data.npy").astype(np.float32)

    mask = np.isnan(data)
    if (src / "mask.npy").exists():
        mask |= np.load(src / "mask.npy").astype(bool)
    null_val = config["null_val"]
    if isinstance(null_val, (int, float)) and not np.isnan(null_val):
        mask |= data == null_val

    timestamps = pd.date_range(config["start"], config["end"], freq=config["freq"])
    assert len(timestamps) == len(data), (src.name, len(timestamps), len(data))
    graph_path = src / "adj_mtx.npy"
    save(
        out_root / src.name,
        name=src.name,
        source="stfm",
        data=data,
        mask=mask,
        time_feat=time_features(timestamps),
        graph=np.load(graph_path) if graph_path.exists() else None,
        freq=config["freq"],
        start=str(timestamps[0]),
    )


def convert_basicts(src: Path, out_root: Path) -> None:
    with open(src / "meta.json") as f:
        meta = json.load(f)
    data = np.concatenate([np.load(src / f"{s}_data.npy") for s in SPLITS])

    time_feat = None
    if meta["timestamps_description"] is not None:
        assert meta["timestamps_description"] == BASICTS_TIME_FEATURES, meta
        time_feat = np.concatenate([np.load(src / f"{s}_timestamps.npy") for s in SPLITS])

    minutes = meta["frequency (minutes)"]
    start = BASICTS_STARTS.get(src.name)
    if start is not None:
        timestamps = pd.date_range(start, periods=len(data), freq=f"{int(minutes)}min")
        recomputed = time_features(timestamps)
        # The documented start must reproduce the shipped time of day / week / month exactly.
        assert np.array_equal(recomputed[:, :3], time_feat[:, :3]), src.name
        time_feat, start = recomputed, str(timestamps[0])
    save(
        out_root / src.name,
        name=src.name,
        source="basicts",
        data=data,
        mask=np.isnan(data),
        time_feat=time_feat,
        graph=None,  # none of the BasicTS long-term datasets ship a graph
        freq=f"{int(minutes)}min" if minutes is not None else None,
        start=start,
    )


def convert_webtris(src: Path, out_root: Path) -> None:
    """Total volume of every mainline site on a wall-clock 15-minute grid (UK local time).

    Rows are keyed by their local wall-clock slot, so the hour skipped when clocks go forward stays
    missing; date ranges spanning the autumn change keep the first of the repeated hour.
    """
    with open(src / "fetch_config.json") as f:
        config = json.load(f)
    with open(src / "sites.json") as f:
        sites = {s["Id"]: s for s in json.load(f)["sites"]}
    timestamps = pd.date_range(
        config["start"],
        pd.Timestamp(config["end"]) + pd.Timedelta(days=1),
        freq=WEBTRIS_FREQ,
        inclusive="left",
    )
    step = pd.Timedelta(WEBTRIS_FREQ)

    columns, nodes = [], []
    for path in sorted((src / "reports").glob("*.json.gz")):
        with gzip.open(path, "rt") as f:
            rows = json.load(f)["Rows"]
        if not rows:
            continue
        df = pd.DataFrame(rows)
        # "Time Interval" is the wall-clock slot of the day (it skips 4..7 when clocks go forward)
        t = pd.to_datetime(df["Report Date"]) + df["Time Interval"].astype(int) * step
        volume = pd.Series(pd.to_numeric(df["Total Volume"], errors="coerce").values, index=t)
        volume = volume[~volume.index.duplicated()].reindex(timestamps)
        if volume.isna().mean() > MAX_SITE_MISSING:
            continue
        site = sites[path.name.removesuffix(".json.gz")]
        road, marker = site["Name"].split("/")
        # sites on one link (road segment between junctions) count the same vehicles
        link = re.search(r"on link (\d+)", site["Description"])
        columns.append(volume.to_numpy(np.float32))
        nodes.append(
            {
                "site_id": site["Id"],
                "name": site["Name"],
                "road": road,
                "carriageway": marker[-1],
                "marker": int(marker[:-1]),
                "direction": site["Description"].rsplit(";", 1)[-1].strip(),
                "link": link.group(1) if link else None,
                "latitude": site["Latitude"],
                "longitude": site["Longitude"],
            }
        )

    # node order: carriageway, then location marker along the road
    nodes = pd.DataFrame(nodes).sort_values(["carriageway", "marker"], kind="stable")
    data = np.stack(columns, axis=1)[:, nodes.index.to_numpy()]
    nodes = nodes.reset_index(drop=True)
    # <NAME>-Link: one site per link, the one with the fewest missing steps (first along the road
    # on ties); sites on one link count the same vehicles and are near-duplicates (corr > 0.999)
    nodes["missing"] = np.isnan(data).mean(0)
    per_link = nodes.dropna(subset=["link"]).sort_values("missing", kind="stable")
    keep = np.sort(
        np.concatenate(
            [
                per_link.drop_duplicates("link").index.to_numpy(),
                nodes.index[nodes["link"].isna()].to_numpy(),
            ]
        )
    )
    nodes = nodes.drop(columns="missing")
    for name, cols in ((src.name, nodes.index.to_numpy()), (f"{src.name}-Link", keep)):
        save(
            out_root / name,
            name=name,
            source="webtris",
            data=data[:, cols],
            mask=np.isnan(data[:, cols]),
            time_feat=time_features(timestamps),
            graph=None,
            freq=WEBTRIS_FREQ,
            start=str(timestamps[0]),
            nodes=nodes.iloc[cols].reset_index(drop=True),
        )


def convert_coops(src: Path, out_root: Path) -> None:
    """Hourly non-tidal residual (observed water level minus tide prediction, m) of every gauge.

    The residual is what is left after removing the astronomical tide: storm surge, wind setup,
    river discharge and seasonal mean sea level. Timestamps are UTC.
    """
    with open(src / "fetch_config.json") as f:
        config = json.load(f)
    with open(src / "stations.json") as f:
        stations = {s["id"]: s for s in json.load(f)["stations"]}
    first, last = config["years"]
    timestamps = pd.date_range(
        f"{first}-01-01", f"{last + 1}-01-01", freq=COOPS_FREQ, inclusive="left"
    )

    def read(station: str, product: str) -> pd.Series:
        parts = []
        for path in sorted((src / "data").glob(f"{station}_*_{product}.csv")):
            df = pd.read_csv(path, skipinitialspace=True)
            parts.append(
                pd.Series(df.iloc[:, 1].to_numpy(float), index=pd.to_datetime(df["Date Time"]))
            )
        series = pd.concat(parts) if parts else pd.Series(dtype=float)
        return series[~series.index.duplicated()].reindex(timestamps)

    columns, nodes = [], []
    for station in sorted({p.name.split("_")[0] for p in (src / "data").glob("*.csv")}):
        residual = read(station, "hourly_height") - read(station, "predictions")
        if residual.isna().mean() > MAX_SITE_MISSING:
            continue
        s = stations[station]
        columns.append(residual.to_numpy(np.float32))
        nodes.append(
            {
                "station_id": station,
                "name": s["name"],
                "state": s["state"],
                "latitude": s["lat"],
                "longitude": s["lng"],
            }
        )

    nodes = pd.DataFrame(nodes).sort_values("latitude", kind="stable")  # south to north
    data = np.stack(columns, axis=1)[:, nodes.index.to_numpy()]
    save(
        out_root / src.name,
        name=src.name,
        source="coops",
        data=data,
        mask=np.isnan(data),
        time_feat=time_features(timestamps),
        graph=None,
        freq=COOPS_FREQ,
        start=str(timestamps[0]),
        nodes=nodes.reset_index(drop=True),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stfm-root", type=Path)
    parser.add_argument("--basicts-root", type=Path)
    parser.add_argument("--webtris-root", type=Path)
    parser.add_argument("--coops-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", nargs="*", help="dataset names to convert (default: all)")
    args = parser.parse_args()

    sources = [
        (args.stfm_root, "dataset_config.json", convert_stfm),
        (args.basicts_root, "meta.json", convert_basicts),
        (args.webtris_root, "fetch_config.json", convert_webtris),
        (args.coops_root, "fetch_config.json", convert_coops),
    ]
    for root, marker, convert in sources:
        if root is None:
            continue
        for src in sorted(p.parent for p in root.expanduser().glob(f"*/{marker}")):
            if args.only is None or src.name in args.only:
                convert(src, args.out.expanduser())


if __name__ == "__main__":
    main()
