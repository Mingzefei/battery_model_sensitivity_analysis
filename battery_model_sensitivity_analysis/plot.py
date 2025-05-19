import matplotlib.pyplot as plt
import numpy as np

from .utils import save_plot_data
from pathlib import Path


def plot_bar_chart(
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    labels: list,
    ylabel: str,
    save_path: Path,
    legend_labels: list,
    title: str = None,
) -> None:
    """
    Plots a bar chart with error bars and saves it to a specified path.

    Parameters:
    x (np.ndarray): The x locations for the groups.
    y (np.ndarray): The heights of the bars.
    yerr (np.ndarray): The error bars for the heights of the bars.
    labels (list): The labels for the x ticks.
    ylabel (str): The label for the y-axis.
    save_path (Path): The path where the plot will be saved.
    legend_labels (list): The labels for the legend.
    title (str, optional): The title of the plot. Defaults to None.

    Returns:
    None
    """
    # Change the save_path suffix to .txt
    pkl_save_path = save_path.with_suffix(".pkl")
    save_plot_data(
        pkl_save_path,
        x=x,
        y=y,
        yerr=yerr,
        labels=labels,
        ylabel=ylabel,
        legend_labels=legend_labels,
        title=title,
    )

    width = 0.35
    plt.figure(figsize=(10, 5))

    for i, (y_data, yerr_data, label) in enumerate(zip(y, yerr, legend_labels)):
        plt.bar(
            x + i * width - width / 2,
            y_data,
            width,
            yerr=yerr_data,
            label=label,
            capsize=5,
        )

    if title:
        plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_heatmap(
    data: np.ndarray, labels: list, save_path: Path, title: str = None
) -> None:
    """
    Plots a heatmap of the given data and saves it to the specified path.

    Parameters:
    data (np.ndarray): The data to be plotted as a heatmap.
    labels (list): The labels for the x and y axes.
    save_path (Path): The file path where the heatmap image will be saved.
    title (str, optional): The title of the heatmap. Defaults to None.

    Returns:
    None
    """
    # Change the save_path suffix to .txt
    pkl_save_path = save_path.with_suffix(".pkl")
    save_plot_data(pkl_save_path, data=data, labels=labels, title=title)

    plt.figure(figsize=(10, 8))
    plt.imshow(data, cmap="coolwarm", interpolation="none")
    plt.colorbar(label="S2 Value")
    if title:
        plt.title(title)
    plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(len(labels)), labels=labels)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_current_voltage_curve(
    time: np.ndarray,
    voltage: np.ndarray,
    current: np.ndarray,
    save_path: Path,
    title: str = None,
    baseline_time: np.ndarray = None,
    baseline_voltage: np.ndarray = None,
    baseline_current: np.ndarray = None,
) -> None:
    """
    Plots a current-voltage curve with dual y-axes and saves it to a specified path.
    Optionally plots baseline voltage and current.

    Parameters:
    time (np.ndarray): The time data for the plot.
    voltage (np.ndarray): The voltage data for the plot.
    current (np.ndarray): The current data for the plot.
    save_path (Path): The path where the plot will be saved.
    title (str, optional): The title of the plot. Defaults to None.
    baseline_time (np.ndarray, optional): The baseline time data for the plot. Defaults to None.
    baseline_voltage (np.ndarray, optional): The baseline voltage data for the plot. Defaults to None.
    baseline_current (np.ndarray, optional): The baseline current data for the plot. Defaults to None.

    Returns:
    None
    """
    # Ensure all arrays have the same length
    min_length = min(len(time), len(voltage), len(current))
    time = time[:min_length]
    voltage = voltage[:min_length]
    current = current[:min_length]

    if baseline_time is not None and baseline_voltage is not None and baseline_current is not None:
        min_baseline_length = min(len(baseline_time), len(baseline_voltage), len(baseline_current))
        baseline_time = baseline_time[:min_baseline_length]
        baseline_voltage = baseline_voltage[:min_baseline_length]
        baseline_current = baseline_current[:min_baseline_length]

    # Change the save_path suffix to .txt
    pkl_save_path = save_path.with_suffix(".pkl")
    save_plot_data(
        pkl_save_path,
        time=time,
        voltage=voltage,
        current=current,
        baseline_time=baseline_time,
        baseline_voltage=baseline_voltage,
        baseline_current=baseline_current,
        title=title,
    )
    
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Voltage [V]", color="tab:blue")
    ax1.plot(time, voltage, label="Voltage [V]", color="tab:blue")
    if baseline_voltage is not None and baseline_time is not None:
        ax1.plot(
            baseline_time,
            baseline_voltage,
            label="Baseline Voltage [V]",
            linestyle="--",
            color="tab:cyan",
        )
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Current [A]", color="tab:red")
    ax2.plot(time, current, label="Current [A]", color="tab:red")
    if baseline_current is not None and baseline_time is not None:
        ax2.plot(
            baseline_time,
            baseline_current,
            label="Baseline Current [A]",
            linestyle="--",
            color="tab:pink",
        )
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.legend(loc="upper right")

    if title:
        plt.title(title)
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_temperature_heating_curve(
    time: np.ndarray,
    temperature: np.ndarray,
    heating_power: np.ndarray,
    save_path: Path,
    title: str = None,
    baseline_time: np.ndarray = None,
    baseline_temperature: np.ndarray = None,
    baseline_heating_power: np.ndarray = None,
) -> None:
    """
    Plots a temperature-heating power curve with dual y-axes and saves it to a specified path.
    Optionally plots baseline temperature and heating power.

    Parameters:
    time (np.ndarray): The time data for the plot.
    temperature (np.ndarray): The temperature data for the plot.
    heating_power (np.ndarray): The heating power data for the plot.
    baseline_time (np.ndarray, optional): The baseline time data for the plot. Defaults to None.
    baseline_temperature (np.ndarray, optional): The baseline temperature data for the plot. Defaults to None.
    baseline_heating_power (np.ndarray, optional): The baseline heating power data for the plot. Defaults to None.
    save_path (Path): The path where the plot will be saved.
    title (str, optional): The title of the plot. Defaults to None.

    Returns:
    None
    """
    # Change the save_path suffix to .txt
    pkl_save_path = save_path.with_suffix(".pkl")
    save_plot_data(
        pkl_save_path,
        time=time,
        temperature=temperature,
        heating_power=heating_power,
        baseline_time=baseline_time,
        baseline_temperature=baseline_temperature,
        baseline_heating_power=baseline_heating_power,
        title=title,
    )
    
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Temperature [K]", color="tab:blue")
    ax1.plot(time, temperature, label="Temperature [K]", color="tab:blue")
    if baseline_temperature is not None and baseline_time is not None:
        ax1.plot(
            baseline_time,
            baseline_temperature,
            label="Baseline Temperature [K]",
            linestyle="--",
            color="tab:cyan",
        )
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Heating Power [W]", color="tab:red")
    ax2.plot(time, heating_power, label="Heating Power [W]", color="tab:red")
    if baseline_heating_power is not None and baseline_time is not None:
        ax2.plot(
            baseline_time,
            baseline_heating_power,
            label="Baseline Heating Power [W]",
            linestyle="--",
            color="tab:pink",
        )
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.legend(loc="upper right")

    if title:
        plt.title(title)
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
def plot_capacity_degradation(
    cycle: np.ndarray,
    capacity: np.ndarray,
    discharge_capacity: np.ndarray,
    save_path: Path,
    title: str = None,
    baseline_cycle: np.ndarray = None,
    baseline_capacity: np.ndarray = None,
    baseline_discharge_capacity: np.ndarray = None,
) -> None:
    """
    Plots a capacity degeneration curve with dual y-axes and saves it to a specified path.
    Optionally plots baseline capacity and discharge capacity.

    Parameters:
    cycle (np.ndarray): The cycle number data for the plot.
    capacity (np.ndarray): The capacity data for the plot.
    discharge_capacity (np.ndarray): The discharge capacity data for the plot.
    save_path (Path): The path where the plot will be saved.
    title (str, optional): The title of the plot. Defaults to None.
    baseline_cycle (np.ndarray, optional): The baseline cycle number data for the plot. Defaults to None.
    baseline_capacity (np.ndarray, optional): The baseline capacity data for the plot. Defaults to None.
    baseline_discharge_capacity (np.ndarray, optional): The baseline discharge capacity data for the plot. Defaults to None.

    Returns:
    None
    """
    # Change the save_path suffix to .txt
    pkl_save_path = save_path.with_suffix(".pkl")
    save_plot_data(
        pkl_save_path,
        cycle=cycle,
        capacity=capacity,
        discharge_capacity=discharge_capacity,
        baseline_cycle=baseline_cycle,
        baseline_capacity=baseline_capacity,
        baseline_discharge_capacity=baseline_discharge_capacity,
        title=title,
    )
    
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel("Cycle Number")
    ax1.set_ylabel("Capacity [A.h]", color="tab:blue")
    ax1.plot(cycle, capacity, label="Capacity [A.h]", color="tab:blue")
    if baseline_capacity is not None and baseline_cycle is not None:
        ax1.plot(
            baseline_cycle,
            baseline_capacity,
            label="Baseline Capacity [A.h]",
            linestyle="--",
            color="tab:cyan",
        )
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Discharge Capacity [A.h]", color="tab:red")
    ax2.plot(cycle, discharge_capacity, label="Discharge Capacity [A.h]", color="tab:red")
    if baseline_discharge_capacity is not None and baseline_cycle is not None:
        ax2.plot(
            baseline_cycle,
            baseline_discharge_capacity,
            label="Baseline Discharge Capacity [A.h]",
            linestyle="--",
            color="tab:pink",
        )
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.legend(loc="upper right")

    if title:
        plt.title(title)
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()