#!/usr/bin/env python3
"""Generate evaluation figures for ego-coalition DFL experiments.

This script reads a tidy experiment table and produces the following plots:

1. Accuracy vs rounds
2. Loss vs rounds
3. Coalition stability vs rounds
4. Final accuracy boxplot
5. Convergence round boxplot
6. In-coalition vs out-of-coalition score distribution
7. Coalition size distribution
8. Heatmap of pairwise wins

The script is designed for experiment tables where each row describes one
metric, one seed, one coalition percentage, and one round. Some figures
operate on per-round rows, while others operate on per-run summary values.

The implementation uses type hints compatible with ``mypy``, follows a
``pylint``-friendly style, and uses Sphinx-compatible docstrings.

Examples
--------
Run from the command line:

.. code-block:: bash

    python plot_dfl_metrics.py \
        --input results.csv \
        --output-dir figures \
        --format png \
        --ci 95

The input file may be CSV or Parquet.

Notes
-----
Expected columns for per-round plots:

- ``dataset``
- ``topology``
- ``metric_name``
- ``coalition_percent``
- ``seed``
- ``round``
- ``mean_test_accuracy``
- ``mean_test_loss``
- ``mean_coalition_stability``
- ``mean_coalition_size``
- ``mean_degree``
- ``convergence_round``

Expected columns for the score-distribution plot:

- ``score_group`` with values such as ``in_coalition`` and ``out_coalition``
- ``score_value``

Optional columns:

- ``setting_id``

If ``setting_id`` is absent, it is automatically built from dataset, topology,
and coalition percentage.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_DPI: Final[int] = 180
DEFAULT_FORMAT: Final[str] = "png"
SUPPORTED_FORMATS: Final[set[str]] = {"png", "pdf", "svg"}
REQUIRED_BASE_COLUMNS: Final[set[str]] = {
    "dataset",
    "topology",
    "metric_name",
    "coalition_percent",
    "seed",
    "round",
    "mean_test_accuracy",
    "mean_test_loss",
    "mean_coalition_stability",
    "mean_coalition_size",
    "mean_degree",
    "convergence_round",
}
REQUIRED_SCORE_COLUMNS: Final[set[str]] = {"score_group", "score_value"}
IN_GROUP: Final[str] = "in_coalition"
OUT_GROUP: Final[str] = "out_coalition"


@dataclass(frozen=True)
class PlotConfig:
    """Configuration for figure generation.

    Parameters
    ----------
    output_dir : Path
        Directory where figures will be saved.
    file_format : str
        Output image format. Supported values are ``png``, ``pdf``, and ``svg``.
    dpi : int
        Dots per inch used for raster output.
    ci : int
        Confidence interval level expressed as a percentage. Common values
        are ``90``, ``95``, and ``99``.
    """

    output_dir: Path
    file_format: str = DEFAULT_FORMAT
    dpi: int = DEFAULT_DPI
    ci: int = 95


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Generate plots for ego-coalition DFL experiments."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input results table in CSV or Parquet format.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the generated figures will be stored.",
    )
    parser.add_argument(
        "--format",
        default=DEFAULT_FORMAT,
        choices=sorted(SUPPORTED_FORMATS),
        help="Output file format for figures.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="Figure DPI for raster formats.",
    )
    parser.add_argument(
        "--ci",
        type=int,
        default=95,
        help="Confidence interval level, for example 95.",
    )
    return parser.parse_args()


def load_results_table(path: Path) -> pd.DataFrame:
    """Load the experiment table from CSV or Parquet.

    Parameters
    ----------
    path : Path
        Path to the input file.

    Returns
    -------
    pandas.DataFrame
        Loaded table.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    FileNotFoundError
        If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported input format '{suffix}'. Use CSV or Parquet.")


def validate_columns(dataframe: pd.DataFrame) -> None:
    """Validate that the required columns are present.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input experiment table.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    missing = sorted(REQUIRED_BASE_COLUMNS - set(dataframe.columns))
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required columns: {missing_str}")


def ensure_setting_id(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Ensure a ``setting_id`` column exists.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input experiment table.

    Returns
    -------
    pandas.DataFrame
        Copy of the input table with a ``setting_id`` column.
    """
    result = dataframe.copy()
    if "setting_id" not in result.columns:
        result["setting_id"] = (
            result["dataset"].astype(str)
            + "|"
            + result["topology"].astype(str)
            + "|"
            + result["coalition_percent"].astype(str)
        )
    return result


def normal_z_value(ci: int) -> float:
    """Return an approximate normal critical value for common confidence levels.

    Parameters
    ----------
    ci : int
        Confidence interval level.

    Returns
    -------
    float
        Approximate z critical value.

    Notes
    -----
    This function intentionally avoids adding an external dependency on
    SciPy. For common confidence levels, these values are standard.
    """
    lookup: dict[int, float] = {
        80: 1.282,
        85: 1.440,
        90: 1.645,
        95: 1.960,
        98: 2.326,
        99: 2.576,
    }
    return lookup.get(ci, 1.960)


def compute_error_band(
    values_by_round: pd.Series,
    ci: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean and error band from grouped round values.

    Parameters
    ----------
    values_by_round : pandas.Series
        Series indexed by round, where each element is a sequence of values.
    ci : int
        Confidence interval level.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Arrays containing mean, lower bound, and upper bound.
    """
    z_value = normal_z_value(ci)
    means: list[float] = []
    lowers: list[float] = []
    uppers: list[float] = []

    for values in values_by_round:
        array = np.asarray(values, dtype=float)
        mean_value = float(np.mean(array))
        if array.size <= 1:
            margin = 0.0
        else:
            std_value = float(np.std(array, ddof=1))
            margin = z_value * std_value / math.sqrt(array.size)
        means.append(mean_value)
        lowers.append(mean_value - margin)
        uppers.append(mean_value + margin)

    return (
        np.asarray(means, dtype=float),
        np.asarray(lowers, dtype=float),
        np.asarray(uppers, dtype=float),
    )


def save_figure(
    figure: plt.Figure,
    config: PlotConfig,
    filename_stem: str,
) -> None:
    """Save a figure and close it.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure object to save.
    config : PlotConfig
        Plot configuration.
    filename_stem : str
        Output file name without extension.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / f"{filename_stem}.{config.file_format}"
    figure.tight_layout()
    figure.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(figure)


def sorted_metrics(dataframe: pd.DataFrame) -> list[str]:
    """Return metric names in sorted order.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input experiment table.

    Returns
    -------
    list[str]
        Sorted metric names.
    """
    return sorted(dataframe["metric_name"].dropna().astype(str).unique().tolist())


def sorted_percentages(dataframe: pd.DataFrame) -> list[int]:
    """Return coalition percentages in sorted order.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input experiment table.

    Returns
    -------
    list[int]
        Sorted coalition percentages.
    """
    unique_values = dataframe["coalition_percent"].dropna().astype(int).unique()
    return sorted(int(value) for value in unique_values.tolist())


def group_round_statistics(
    dataframe: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    """Aggregate per-round values by metric and round.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Filtered experiment table.
    value_column : str
        Name of the numeric column to aggregate.

    Returns
    -------
    pandas.DataFrame
        Dataframe with columns ``metric_name``, ``round``, and ``values``.
    """
    grouped = (
        dataframe.groupby(["metric_name", "round"], as_index=False)[value_column]
        .apply(list)
        .rename(columns={value_column: "values"})
    )
    return grouped


def plot_line_with_band_per_percentage(
    dataframe: pd.DataFrame,
    value_column: str,
    y_label: str,
    title_prefix: str,
    file_prefix: str,
    config: PlotConfig,
) -> None:
    """Plot one line chart with uncertainty band per coalition percentage.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input experiment table.
    value_column : str
        Column to plot on the y-axis.
    y_label : str
        Label for the y-axis.
    title_prefix : str
        Prefix used in the figure title.
    file_prefix : str
        Prefix used in the output file name.
    config : PlotConfig
        Plot configuration.
    """
    for coalition_percent in sorted_percentages(dataframe):
        subset = dataframe[dataframe["coalition_percent"] == coalition_percent].copy()
        figure, axis = plt.subplots(figsize=(9, 5))

        for metric_name in sorted_metrics(subset):
            metric_subset = subset[subset["metric_name"] == metric_name].copy()
            grouped = (
                metric_subset.groupby("round")[value_column].apply(list).sort_index()
            )
            rounds = grouped.index.to_numpy(dtype=int)
            mean_values, lower_values, upper_values = compute_error_band(
                grouped,
                config.ci,
            )
            axis.plot(rounds, mean_values, label=metric_name)
            axis.fill_between(rounds, lower_values, upper_values, alpha=0.20)

        axis.set_title(f"{title_prefix} (top-{coalition_percent}%)")
        axis.set_xlabel("Round")
        axis.set_ylabel(y_label)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best", fontsize=8)

        filename = f"{file_prefix}_top_{coalition_percent}"
        save_figure(figure, config, filename)


def plot_accuracy_vs_rounds(
    dataframe: pd.DataFrame,
    config: PlotConfig,
) -> None:
    """Generate mean accuracy vs rounds plots.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input experiment table.
    config : PlotConfig
        Plot configuration.
    """
    plot_line_with_band_per_percentage(
        dataframe=dataframe,
        value_column="mean_test_accuracy",
        y_label="Mean test accuracy",
        title_prefix="Accuracy vs rounds",
        file_prefix="accuracy_vs_rounds",
        config=config,
    )


def plot_loss_vs_rounds(
    dataframe: pd.DataFrame,
    config: PlotConfig,
) -> None:
    """Generate mean loss vs rounds plots.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input experiment table.
    config : PlotConfig
        Plot configuration.
    """
    plot_line_with_band_per_percentage(
        dataframe=dataframe,
        value_column="mean_test_loss",
        y_label="Mean test loss",
        title_prefix="Loss vs rounds",
        file_prefix="loss_vs_rounds",
        config=config,
    )


def plot_coalition_stability_vs_rounds(
    dataframe: pd.DataFrame,
    config: PlotConfig,
) -> None:
    """Generate coalition stability vs rounds plots.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input experiment table.
    config : PlotConfig
        Plot configuration.
    """
    plot_line_with_band_per_percentage(
        dataframe=dataframe,
        value_column="mean_coalition_stability",
        y_label="Mean coalition stability (Jaccard overlap)",
        title_prefix="Coalition stability vs rounds",
        file_prefix="coalition_stability_vs_rounds",
        config=config,
    )


def final_round_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return one row per metric, percentage, setting, and seed at the final round.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input experiment table.

    Returns
    -------
    pandas.DataFrame
        Rows corresponding to the final round of each run.
    """
    result = dataframe.copy()
    group_columns = [
        "dataset",
        "topology",
        "metric_name",
        "coalition_percent",
        "seed",
        "setting_id",
    ]
    final_rounds = result.groupby(group_columns)["round"].transform("max")
    return result[result["round"] == final_rounds].copy()


def boxplot_by_metric_per_percentage(
    dataframe: pd.DataFrame,
    value_column: str,
    y_label: str,
    title_prefix: str,
    file_prefix: str,
    config: PlotConfig,
) -> None:
    """Generate one boxplot per coalition percentage.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input summary table.
    value_column : str
        Column to boxplot.
    y_label : str
        Label for the y-axis.
    title_prefix : str
        Prefix used in the figure title.
    file_prefix : str
        Prefix used in the output file name.
    config : PlotConfig
        Plot configuration.
    """
    for coalition_percent in sorted_percentages(dataframe):
        subset = dataframe[dataframe["coalition_percent"] == coalition_percent].copy()
        metrics = sorted_metrics(subset)
        data = [
            subset.loc[subset["metric_name"] == metric_name, value_column]
            .dropna()
            .astype(float)
            .to_numpy()
            for metric_name in metrics
        ]

        figure, axis = plt.subplots(figsize=(10, 5))
        axis.boxplot(data, tick_labels=metrics, vert=True)
        axis.set_title(f"{title_prefix} (top-{coalition_percent}%)")
        axis.set_xlabel("Metric")
        axis.set_ylabel(y_label)
        axis.grid(True, axis="y", alpha=0.3)
        axis.tick_params(axis="x", rotation=30)

        filename = f"{file_prefix}_top_{coalition_percent}"
        save_figure(figure, config, filename)


def plot_final_accuracy_boxplot(
    dataframe: pd.DataFrame,
    config: PlotConfig,
) -> None:
    """Generate final accuracy boxplots.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input experiment table.
    config : PlotConfig
        Plot configuration.
    """
    summary = final_round_rows(dataframe)
    boxplot_by_metric_per_percentage(
        dataframe=summary,
        value_column="mean_test_accuracy",
        y_label="Final mean test accuracy",
        title_prefix="Final accuracy distribution",
        file_prefix="final_accuracy_boxplot",
        config=config,
    )


def plot_convergence_round_boxplot(
    dataframe: pd.DataFrame,
    config: PlotConfig,
) -> None:
    """Generate convergence round boxplots.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input experiment table.
    config : PlotConfig
        Plot configuration.
    """
    summary = final_round_rows(dataframe)
    boxplot_by_metric_per_percentage(
        dataframe=summary,
        value_column="convergence_round",
        y_label="Convergence round",
        title_prefix="Convergence round distribution",
        file_prefix="convergence_round_boxplot",
        config=config,
    )


def histogram_density(
    values: np.ndarray,
    bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a normalized histogram density.

    Parameters
    ----------
    values : numpy.ndarray
        Input numeric values.
    bins : int
        Number of bins.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Bin centers and density values.
    """
    density, edges = np.histogram(values, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers.astype(float), density.astype(float)


def plot_score_distribution(
    dataframe: pd.DataFrame,
    config: PlotConfig,
) -> None:
    """Generate in-coalition vs out-of-coalition score distributions.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input experiment table.
    config : PlotConfig
        Plot configuration.

    Notes
    -----
    This function requires the columns ``score_group`` and ``score_value``.
    The expected values of ``score_group`` are ``in_coalition`` and
    ``out_coalition``.
    """
    if not REQUIRED_SCORE_COLUMNS.issubset(dataframe.columns):
        return

    score_df = dataframe.dropna(subset=["score_group", "score_value"]).copy()

    for coalition_percent in sorted_percentages(score_df):
        subset_percent = score_df[
            score_df["coalition_percent"] == coalition_percent
        ].copy()

        for metric_name in sorted_metrics(subset_percent):
            subset = subset_percent[subset_percent["metric_name"] == metric_name].copy()

            in_values = (
                subset.loc[subset["score_group"] == IN_GROUP, "score_value"]
                .astype(float)
                .to_numpy()
            )
            out_values = (
                subset.loc[subset["score_group"] == OUT_GROUP, "score_value"]
                .astype(float)
                .to_numpy()
            )

            if in_values.size == 0 or out_values.size == 0:
                continue

            figure, axis = plt.subplots(figsize=(9, 5))
            combined = np.concatenate([in_values, out_values])
            bins = min(40, max(10, int(np.sqrt(combined.size))))

            axis.hist(
                in_values,
                bins=bins,
                density=True,
                alpha=0.45,
                label="In coalition",
            )
            axis.hist(
                out_values,
                bins=bins,
                density=True,
                alpha=0.45,
                label="Out of coalition",
            )

            in_centers, in_density = histogram_density(in_values, bins)
            out_centers, out_density = histogram_density(out_values, bins)

            axis.plot(in_centers, in_density, linewidth=2.0)
            axis.plot(out_centers, out_density, linewidth=2.0)

            axis.set_title(
                f"Score distribution for {metric_name} (top-{coalition_percent}%)"
            )
            axis.set_xlabel("Score / distance value")
            axis.set_ylabel("Density")
            axis.grid(True, alpha=0.3)
            axis.legend(loc="best")

            filename = f"score_distribution_{metric_name}_top_{coalition_percent}"
            save_figure(figure, config, filename)


def plot_coalition_size_distribution(
    dataframe: pd.DataFrame,
    config: PlotConfig,
) -> None:
    """Generate coalition size distribution plots.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input experiment table.
    config : PlotConfig
        Plot configuration.
    """
    summary = final_round_rows(dataframe)

    for coalition_percent in sorted_percentages(summary):
        subset = summary[summary["coalition_percent"] == coalition_percent].copy()
        metrics = sorted_metrics(subset)
        data = [
            subset.loc[subset["metric_name"] == metric_name, "mean_coalition_size"]
            .dropna()
            .astype(float)
            .to_numpy()
            for metric_name in metrics
        ]

        figure, axis = plt.subplots(figsize=(10, 5))
        axis.violinplot(data, showmeans=True, showmedians=True)
        axis.set_xticks(np.arange(1, len(metrics) + 1))
        axis.set_xticklabels(metrics, rotation=30)
        axis.set_title(f"Coalition size distribution (top-{coalition_percent}%)")
        axis.set_xlabel("Metric")
        axis.set_ylabel("Mean coalition size")
        axis.grid(True, axis="y", alpha=0.3)

        filename = f"coalition_size_distribution_top_{coalition_percent}"
        save_figure(figure, config, filename)


def pairwise_win_matrix(
    dataframe: pd.DataFrame,
) -> tuple[list[str], np.ndarray]:
    """Compute the pairwise win matrix using final accuracy.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input experiment table.

    Returns
    -------
    tuple[list[str], numpy.ndarray]
        Metric names and a matrix where entry ``(i, j)`` is the percentage of
        settings in which metric ``i`` outperforms metric ``j`` on final
        accuracy.

    Notes
    -----
    Each setting is defined by ``setting_id`` and ``seed``. Final accuracy is
    extracted from the last round of each run.
    """
    summary = final_round_rows(dataframe)
    metrics = sorted_metrics(summary)

    pivot = summary.pivot_table(
        index=["setting_id", "seed"],
        columns="metric_name",
        values="mean_test_accuracy",
        aggfunc="mean",
    )

    matrix = np.full((len(metrics), len(metrics)), np.nan, dtype=float)

    for row_index, metric_a in enumerate(metrics):
        for col_index, metric_b in enumerate(metrics):
            if metric_a == metric_b:
                matrix[row_index, col_index] = 50.0
                continue

            valid = pivot[[metric_a, metric_b]].dropna()
            if valid.empty:
                continue

            wins = float((valid[metric_a] > valid[metric_b]).sum())
            total = float(valid.shape[0])
            matrix[row_index, col_index] = 100.0 * wins / total

    return metrics, matrix


def plot_pairwise_wins_heatmap(
    dataframe: pd.DataFrame,
    config: PlotConfig,
) -> None:
    """Generate a heatmap of pairwise metric wins.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input experiment table.
    config : PlotConfig
        Plot configuration.
    """
    metrics, matrix = pairwise_win_matrix(dataframe)

    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, aspect="auto")

    axis.set_xticks(np.arange(len(metrics)))
    axis.set_yticks(np.arange(len(metrics)))
    axis.set_xticklabels(metrics, rotation=45, ha="right")
    axis.set_yticklabels(metrics)
    axis.set_title("Pairwise wins by final accuracy (%)")

    for row_index in range(len(metrics)):
        for col_index in range(len(metrics)):
            value = matrix[row_index, col_index]
            if np.isnan(value):
                text = "NA"
            else:
                text = f"{value:.1f}"
            axis.text(
                col_index,
                row_index,
                text,
                ha="center",
                va="center",
                fontsize=8,
            )

    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Win percentage")
    save_figure(figure, config, "pairwise_wins_heatmap")


def summarize_input(dataframe: pd.DataFrame) -> str:
    """Build a short textual summary of the loaded dataset.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input experiment table.

    Returns
    -------
    str
        Human-readable summary.
    """
    metrics = sorted_metrics(dataframe)
    percentages = sorted_percentages(dataframe)
    seeds = sorted(dataframe["seed"].dropna().astype(int).unique().tolist())
    datasets = sorted(dataframe["dataset"].dropna().astype(str).unique().tolist())
    topologies = sorted(dataframe["topology"].dropna().astype(str).unique().tolist())

    return (
        f"Loaded {len(dataframe)} rows | "
        f"metrics={metrics} | "
        f"coalition_percentages={percentages} | "
        f"seeds={seeds} | "
        f"datasets={datasets} | "
        f"topologies={topologies}"
    )


def main() -> None:
    """Run the plotting pipeline."""
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    config = PlotConfig(
        output_dir=output_dir,
        file_format=args.format,
        dpi=args.dpi,
        ci=args.ci,
    )

    dataframe = load_results_table(input_path)
    validate_columns(dataframe)
    dataframe = ensure_setting_id(dataframe)

    print(summarize_input(dataframe))

    plot_accuracy_vs_rounds(dataframe, config)
    plot_loss_vs_rounds(dataframe, config)
    plot_coalition_stability_vs_rounds(dataframe, config)
    plot_final_accuracy_boxplot(dataframe, config)
    plot_convergence_round_boxplot(dataframe, config)
    plot_score_distribution(dataframe, config)
    plot_coalition_size_distribution(dataframe, config)
    plot_pairwise_wins_heatmap(dataframe, config)

    print(f"Figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
