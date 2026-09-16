"""Episode-level statistics, bootstrap aggregation, and precursor analysis."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

import numpy as np
import pandas as pd


CORE_AGGREGATE_METRICS = [
    "target_mean_similarity",
    "similarity_margin",
    "target_mass",
    "peak_localization_error",
    "centroid_localization_error",
    "temporal_drift",
    "heatmap_entropy",
]
GT_DEPENDENT_METRICS = {
    "target_mean_similarity",
    "background_mean_similarity",
    "similarity_margin",
    "target_mass",
    "peak_localization_error",
    "centroid_localization_error",
}


def _finite(values: Iterable[Any]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float, int]:
    """Mean and episode-level bootstrap interval, ignoring missing values."""

    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(values.mean())
    if len(values) < 2 or int(n_bootstrap) <= 0:
        return mean, float("nan"), float("nan"), int(len(values))
    rng = np.random.default_rng(int(seed))
    samples = rng.choice(values, size=(int(n_bootstrap), len(values)), replace=True).mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return mean, float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha)), int(len(values))


def interpolate_episode_metric(
    rows: pd.DataFrame,
    *,
    metric: str,
    progress_grid: np.ndarray,
) -> np.ndarray:
    progress = rows["progress"].to_numpy(dtype=np.float64)
    values = rows[metric].to_numpy(dtype=np.float64)
    valid = np.isfinite(progress) & np.isfinite(values)
    if valid.sum() == 0:
        return np.full_like(progress_grid, np.nan, dtype=np.float64)
    progress = progress[valid]
    values = values[valid]
    order = np.argsort(progress)
    progress, values = progress[order], values[order]
    unique_progress, inverse = np.unique(progress, return_inverse=True)
    if len(unique_progress) != len(progress):
        values = np.asarray(
            [values[inverse == index].mean() for index in range(len(unique_progress))],
            dtype=np.float64,
        )
        progress = unique_progress
    if len(progress) == 1:
        return np.full_like(progress_grid, values[0], dtype=np.float64)
    return np.interp(progress_grid, progress, values, left=values[0], right=values[-1])


def aggregate_trajectory(
    frame_df: pd.DataFrame,
    *,
    metrics: Iterable[str] = CORE_AGGREGATE_METRICS,
    progress_points: int = 100,
    n_bootstrap: int = 2000,
    seed: int = 0,
    camera: Optional[str] = None,
    task_id: Optional[str] = None,
    shift_condition: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Interpolate each episode first, then bootstrap across episodes."""

    selected = frame_df
    if camera is not None:
        selected = selected[selected["camera"] == camera]
    if task_id is not None:
        selected = selected[selected["task_id"].astype(str) == str(task_id)]
    if shift_condition is not None:
        selected = selected[selected["shift_condition"] == shift_condition]
    selected = selected.copy()
    if "has_evaluation_gt_mask" in selected and "is_reference_frame" in selected:
        valid_future_gt = selected["has_evaluation_gt_mask"].astype(bool) & ~selected["is_reference_frame"].astype(bool)
        for metric in GT_DEPENDENT_METRICS:
            if metric in selected:
                selected.loc[~valid_future_gt, metric] = np.nan
    metrics = list(metrics)
    progress_grid = np.linspace(0.0, 1.0, int(progress_points), dtype=np.float64)
    result: list[dict[str, Any]] = []
    for outcome, outcome_df in selected.groupby("outcome", dropna=False):
        episode_groups = list(outcome_df.groupby("episode_id", sort=True))
        if not episode_groups:
            continue
        matrices: dict[str, np.ndarray] = {}
        for metric in metrics:
            matrices[metric] = np.vstack(
                [interpolate_episode_metric(rows, metric=metric, progress_grid=progress_grid) for _, rows in episode_groups]
            )
        for index, progress in enumerate(progress_grid):
            row: dict[str, Any] = {
                "outcome": str(outcome),
                "progress": float(progress),
                "episode_count": len(episode_groups),
                "camera": camera or "all",
                "task_id": task_id or "all",
                "shift_condition": shift_condition or "all",
            }
            for metric_index, metric in enumerate(metrics):
                mean, low, high, count = bootstrap_mean_ci(
                    matrices[metric][:, index],
                    n_bootstrap=n_bootstrap,
                    seed=int(seed) + index * 31 + metric_index,
                )
                row[f"{metric}_mean"] = mean
                row[f"{metric}_ci_low"] = low
                row[f"{metric}_ci_high"] = high
                row[f"{metric}_n"] = count
            result.append(row)
    return result


def cohen_d(success_values: Iterable[Any], failure_values: Iterable[Any]) -> float:
    success = _finite(success_values)
    failure = _finite(failure_values)
    if len(success) < 2 or len(failure) < 2:
        return float("nan")
    pooled = np.sqrt(((len(success) - 1) * np.var(success, ddof=1) + (len(failure) - 1) * np.var(failure, ddof=1)) / (len(success) + len(failure) - 2))
    return float((np.mean(success) - np.mean(failure)) / pooled) if pooled > 0 else float("nan")


def roc_auc_episode_level(labels: Iterable[Any], scores: Iterable[Any]) -> float:
    """A dependency-free ROC-AUC using positive=success and episode scores."""

    labels_array = np.asarray(list(labels), dtype=np.int64)
    scores_array = np.asarray(list(scores), dtype=np.float64)
    valid = np.isfinite(scores_array) & np.isin(labels_array, [0, 1])
    labels_array, scores_array = labels_array[valid], scores_array[valid]
    positives = scores_array[labels_array == 1]
    negatives = scores_array[labels_array == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")
    # Pairwise form handles ties as 0.5 and is adequate for episode-level pilot sizes.
    comparisons = positives[:, None] - negatives[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def episode_summaries(
    frame_df: pd.DataFrame,
    *,
    camera: str,
    pregrasp_only: bool = False,
) -> pd.DataFrame:
    """Reduce frame metrics to one row per episode without frame-level leakage."""

    selected = frame_df[frame_df["camera"] == camera].copy()
    if "has_evaluation_gt_mask" in selected and "is_reference_frame" in selected:
        valid_future_gt = selected["has_evaluation_gt_mask"].astype(bool) & ~selected["is_reference_frame"].astype(bool)
        for metric in GT_DEPENDENT_METRICS:
            if metric in selected:
                selected.loc[~valid_future_gt, metric] = np.nan
    if pregrasp_only and "grasp_close_frame" in selected:
        selected = selected[
            selected["grasp_close_frame"].isna() | (selected["frame_index"] <= selected["grasp_close_frame"])
        ]
    rows: list[dict[str, Any]] = []
    for episode_id, group in selected.groupby("episode_id", sort=True):
        first = group.iloc[0]
        row: dict[str, Any] = {
            "episode_id": episode_id,
            "task_id": str(first["task_id"]),
            "task_name": first["task_name"],
            "outcome": first["outcome"],
            "success": int(first["outcome"] == "success"),
            "shift_condition": first["shift_condition"],
        }
        for metric, operation in {
            "target_mass": "mean",
            "similarity_margin": "min",
            "centroid_localization_error": "max",
            "peak_localization_error": "max",
            "temporal_drift": "max",
            "tracking_residual_drift": "max",
            "heatmap_entropy": "mean",
            "peak_similarity": "mean",
        }.items():
            values = _finite(group[metric]) if metric in group else np.asarray([])
            row[f"{operation}_{metric}"] = float(getattr(np, operation)(values)) if len(values) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def compute_effects_and_auc(summary_df: pd.DataFrame) -> dict[str, Any]:
    """Report descriptive episode-level effect sizes and prediction AUCs."""

    result: dict[str, Any] = {"episode_count": int(len(summary_df))}
    if summary_df.empty:
        return result
    successes = summary_df[summary_df["success"] == 1]
    failures = summary_df[summary_df["success"] == 0]
    result["success_count"] = int(len(successes))
    result["failure_count"] = int(len(failures))
    metrics = [
        "mean_target_mass",
        "min_similarity_margin",
        "max_centroid_localization_error",
        "max_peak_localization_error",
        "max_temporal_drift",
        "mean_heatmap_entropy",
    ]
    effect_sizes: dict[str, float] = {}
    aucs: dict[str, float] = {}
    for metric in metrics:
        if metric not in summary_df:
            continue
        effect_sizes[metric] = cohen_d(successes[metric], failures[metric])
        score = summary_df[metric].to_numpy(dtype=np.float64)
        if metric.startswith("max_") and "localization" in metric or metric == "max_temporal_drift":
            score = -score
        aucs[metric] = roc_auc_episode_level(summary_df["success"], score)
    result["cohen_d_success_minus_failure"] = effect_sizes
    result["roc_auc_success_episode_level"] = aucs
    result["split_unit"] = "episode"
    return result


def failure_precursors(
    frame_df: pd.DataFrame,
    *,
    camera: str,
    low_metrics: Iterable[str] = ("target_mass", "similarity_margin"),
    high_metrics: Iterable[str] = ("centroid_localization_error", "temporal_drift"),
    success_percentile: float = 5.0,
    persistence_frames: int = 1,
    failure_frame_by_episode: Optional[Mapping[str, int]] = None,
) -> dict[str, Any]:
    """Learn thresholds from successes and measure first pre-failure anomalies."""

    selected = frame_df[frame_df["camera"] == camera].copy()
    if "has_evaluation_gt_mask" in selected and "is_reference_frame" in selected:
        valid_future_gt = selected["has_evaluation_gt_mask"].astype(bool) & ~selected["is_reference_frame"].astype(bool)
        for metric in GT_DEPENDENT_METRICS:
            if metric in selected:
                selected.loc[~valid_future_gt, metric] = np.nan
    successes = selected[selected["outcome"] == "success"]
    failures = selected[selected["outcome"] == "failure"]
    thresholds: dict[str, dict[str, float]] = {}
    for metric in low_metrics:
        values = _finite(successes[metric]) if metric in successes else np.asarray([])
        thresholds[metric] = {"direction": "low", "value": float(np.percentile(values, success_percentile)) if len(values) else float("nan")}
    for metric in high_metrics:
        values = _finite(successes[metric]) if metric in successes else np.asarray([])
        thresholds[metric] = {"direction": "high", "value": float(np.percentile(values, 100.0 - success_percentile)) if len(values) else float("nan")}

    def anomaly_series(group: pd.DataFrame) -> np.ndarray:
        anomaly = np.zeros(len(group), dtype=bool)
        for metric, rule in thresholds.items():
            if not np.isfinite(rule["value"]) or metric not in group:
                continue
            values = group[metric].to_numpy(dtype=np.float64)
            if rule["direction"] == "low":
                anomaly |= np.isfinite(values) & (values < rule["value"])
            else:
                anomaly |= np.isfinite(values) & (values > rule["value"])
        if persistence_frames <= 1:
            return anomaly
        persistent = np.zeros_like(anomaly)
        for index in range(len(anomaly)):
            start = max(0, index - persistence_frames + 1)
            persistent[index] = bool(anomaly[start : index + 1].all()) and index - start + 1 >= persistence_frames
        return persistent

    records: list[dict[str, Any]] = []
    failure_leads: list[float] = []
    failure_leads_seconds: list[float] = []
    failure_leads_progress: list[float] = []
    failure_detected = 0
    success_false_positives = 0
    for episode_id, group in selected.groupby("episode_id", sort=True):
        group = group.sort_values("frame_index").reset_index(drop=True)
        anomaly = anomaly_series(group)
        first_anomaly_index = np.flatnonzero(anomaly)
        first_anomaly = int(first_anomaly_index[0]) if len(first_anomaly_index) else None
        is_failure = str(group.iloc[0]["outcome"]) == "failure"
        if is_failure:
            default_failure_frame = int(group["frame_index"].max())
            failure_frame = int((failure_frame_by_episode or {}).get(str(episode_id), default_failure_frame))
            before_failure = group["frame_index"].to_numpy(dtype=np.int64) <= failure_frame
            valid_anomaly = first_anomaly is not None and bool(before_failure[first_anomaly]) and int(group.iloc[first_anomaly]["frame_index"]) < failure_frame
            if valid_anomaly:
                failure_detected += 1
                lead = failure_frame - int(group.iloc[first_anomaly]["frame_index"])
                failure_leads.append(float(lead))
                failure_timestamp = float(group.iloc[(group["frame_index"] - failure_frame).abs().argmin()]["timestamp"]) if "timestamp" in group else float(failure_frame)
                anomaly_timestamp = float(group.iloc[first_anomaly]["timestamp"]) if "timestamp" in group else float(group.iloc[first_anomaly]["frame_index"])
                lead_seconds = failure_timestamp - anomaly_timestamp
                lead_progress = float(group.iloc[(group["frame_index"] - failure_frame).abs().argmin()]["progress"] - group.iloc[first_anomaly]["progress"])
                failure_leads_seconds.append(float(lead_seconds))
                failure_leads_progress.append(lead_progress)
            else:
                lead = float("nan")
                lead_seconds = float("nan")
                lead_progress = float("nan")
            records.append(
                {
                    "episode_id": str(episode_id),
                    "outcome": "failure",
                    "failure_frame": failure_frame,
                    "anomaly_frame": None if not valid_anomaly else int(group.iloc[first_anomaly]["frame_index"]),
                    "lead_time_frames": lead,
                    "lead_time_seconds": lead_seconds,
                    "lead_time_progress": lead_progress,
                    "failure_event_source": str(group.iloc[0].get("failure_frame_source", "metadata_or_terminal_proxy")),
                }
            )
        elif first_anomaly is not None:
            success_false_positives += 1
            records.append(
                {
                    "episode_id": str(episode_id),
                    "outcome": "success",
                    "failure_frame": None,
                    "anomaly_frame": int(group.iloc[first_anomaly]["frame_index"]),
                    "lead_time_frames": float("nan"),
                    "lead_time_seconds": float("nan"),
                    "lead_time_progress": float("nan"),
                    "failure_event_source": "not_applicable",
                }
            )
    return {
        "camera": camera,
        "thresholds_fit_on": "success_episodes_only",
        "success_percentile": float(success_percentile),
        "persistence_frames": int(persistence_frames),
        "thresholds": thresholds,
        "episodes": records,
        "failure_count": int(len(failures.groupby("episode_id"))),
        "failures_with_pre_failure_anomaly": int(failure_detected),
        "failure_detection_rate": float(failure_detected / len(failures.groupby("episode_id"))) if len(failures.groupby("episode_id")) else float("nan"),
        "mean_lead_time_frames": float(np.mean(failure_leads)) if failure_leads else float("nan"),
        "median_lead_time_frames": float(np.median(failure_leads)) if failure_leads else float("nan"),
        "mean_lead_time_seconds": float(np.mean(failure_leads_seconds)) if failure_leads_seconds else float("nan"),
        "median_lead_time_seconds": float(np.median(failure_leads_seconds)) if failure_leads_seconds else float("nan"),
        "mean_lead_time_progress": float(np.mean(failure_leads_progress)) if failure_leads_progress else float("nan"),
        "success_false_positive_rate": float(success_false_positives / len(successes.groupby("episode_id"))) if len(successes.groupby("episode_id")) else float("nan"),
    }
