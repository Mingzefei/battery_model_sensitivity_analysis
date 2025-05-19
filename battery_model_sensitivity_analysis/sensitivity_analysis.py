import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pybamm
from SALib import ProblemSpec
from tqdm import tqdm

from ..models.a123_88param import map_params

from ..utils.logger import get_logger
from .plot import (
    plot_bar_chart,
    plot_capacity_degradation,
    plot_current_voltage_curve,
    plot_heatmap,
    plot_temperature_heating_curve,
)
from .simulation import AgingSimulation, Simulation
from .utils import (
    BASED_SIMULATION_FILENAME,
    SIMULATION_FILENAME,
    SP_RESULTS_FILENAME,
    SP_SAMPLES_FILENAME,
    prepare_save_dir,
    redirect_stdout_stderr,
    tar_simulations,
)

logger = get_logger(__name__)


class SensitivityAnalysis:
    def __init__(
        self,
        param: pybamm.ParameterValues,
        sa_param: dict,
        sa_param_bounds: dict[str : [float, float]],
        save_dir: Path,
        experiment_type: str,
        category: str,
        sample_n: int,
        use_current_results=True,
        use_mpi=False,
        aging_cycle_number: int = 20,
        aging_group_number: int = 5,
    ):
        """
        Initialize the SensitivityAnalysis class.

        Parameters:
        -----------
        param : pybamm.ParameterValues
            The parameter values for the PyBaMM model.
        sa_param : dict
            Dictionary containing sensitivity analysis parameters.
        save_dir : Path
            Directory where results will be saved.
        experiment_type : str
            Type of experiment to be conducted.
            Options: ['cccv-cc', 'pulse-pulse', 'cccv-pulse', 'xftg']
        category : str
            Category of the sensitivity analysis.
            Options: ['geometry', 'electrochemical', 'thermal', 'degradation', 'all', 'test']
        sample_n : int
            Number of samples for the sensitivity analysis.
        use_current_results : bool, optional
            Whether to use current results if available (default is True).
        use_mpi : bool, optional
            Whether to use MPI for parallel processing (default is False).

        Attributes:
        -----------
        model_options_type : str
            The type of model options based on the category.
        based_lines : None or other type
            Placeholder attribute, initially set to None.
        """
        self.param = param
        self.sa_param = sa_param
        self.sa_param_bounds = sa_param_bounds
        self.save_dir = save_dir
        self.experiment_type = experiment_type
        self.category = category
        self.sample_n = sample_n
        self.use_current_results = use_current_results
        self.use_mpi = use_mpi
        self.model_options_type = self.get_model_options_type(category)
        self.based_lines = {}
        self.based_aging_lines = {}

        if "degradation" in self.model_options_type:
            self.use_aging = True
            self.aging_cycle_number = aging_cycle_number
            self.aging_group_number = aging_group_number
        else:
            self.use_aging = False

    def run(self):
        """
        Executes the sensitivity analysis workflow.

        This method performs the following steps:
        1. Prepares the save directory and sets the baseline simulation results.
        2. Sets up the sensitivity analysis problem (sp).
        3. If `use_current_results` is False:
            - Generates Sobol samples.
            - Saves the samples to a CSV file.
            - Runs simulations in parallel using the generated samples.
            - Saves the simulation results to a CSV file.
            - Plots a subset of the simulations.
            - Archives the simulation results.
        4. If `use_current_results` is True:
            - Loads existing samples and results.
            - Sets the samples in the sensitivity analysis problem (sp).
        5. Sets the results in the sensitivity analysis problem (sp).
        6. Analyzes the results using Sobol analysis.
        7. Plots the analysis results.
        8. Analyzes and plots cleaned results.

        Logs the start and completion of the sensitivity analysis and the cleaned sensitivity analysis.

        Raises:
            Exception: If any step in the workflow fails.
        """
        logger.info("Starting sensitivity analysis")

        # prepare save_dir and set based_lines
        if not self.use_current_results:
            self.save_dir = prepare_save_dir(self.save_dir)
            if not self.use_aging:
                self.based_lines = self.run_based_simulation()
            elif self.use_aging:
                self.based_lines, self.based_aging_lines = (
                    self.run_based_aging_simulation()
                )
        else:
            if not self.use_aging:
                sim = Simulation.load(self.save_dir / BASED_SIMULATION_FILENAME)
                self.based_lines = sim.outputs
            elif self.use_aging:
                sim = AgingSimulation.load(self.save_dir / BASED_SIMULATION_FILENAME)
                self.based_lines = sim.outputs
                self.based_aging_lines = sim.aging_outputs
        # setup sp
        sp = self.setup_sp()

        if not self.use_current_results:
            # set samples
            sp.sample_sobol(self.sample_n)
            np.savetxt(
                self.save_dir / SP_SAMPLES_FILENAME,
                sp.samples,
                delimiter=",",
                header=",".join(sp.get("names")),
                comments="",
            )
            samples = [
                dict(zip(sp.get("names"), param_values)) for param_values in sp.samples
            ]
            # run sims
            results = self.parallel_run_simulations(samples)
            results_df = pd.DataFrame(results)
            results_df.to_csv(self.save_dir / SP_RESULTS_FILENAME, index=False)
            # plot some sims
            random_indices = self.select_random_simulations(samples, results_df)
            self.plot_simulations(random_indices)
            # tar sims
            tar_simulations(self.save_dir)
        else:
            samples, results_df = self.load_existing_samples_and_results()
            sp.set_samples(samples)
        # set results
        results_np = results_df[sp.get("outputs")].to_numpy()
        sp.set_results(results_np)
        # analyze
        sp.analyze_sobol()
        self.plot_sp_results(sp)
        logger.info("Sensitivity analysis completed")
        # analyze clean
        clean_sp = self.analyze_clean_results(sp, results_df)
        self.plot_sp_results(clean_sp, tag="clean")
        logger.info("Cleaned sensitivity analysis completed")

    def run_based_simulation(self):
        """
        Runs a simulation based on the provided parameters and experiment type.

        This method initializes a Simulation object with the specified parameters,
        experiment type, and model options type. It then solves the simulation,
        retrieves the output lines, and saves the simulation results to a file.

        Returns:
            list: The output lines from the simulation.
        """
        logger.info("Starting baseline simulation")
        with redirect_stdout_stderr():
            based_sim = Simulation(
                self.param,
                experiment_type=self.experiment_type,
                model_options_type=self.model_options_type,
            )
            based_sim.solve()
        based_lines = based_sim.get_outputs()
        based_sim.save(self.save_dir / BASED_SIMULATION_FILENAME)
        logger.info("Baseline simulation completed and results saved")
        return based_lines

    def run_based_aging_simulation(self):
        """
        Runs a simulation based on the provided parameters and experiment type.

        This method initializes a Simulation object with the specified parameters,
        experiment type, and model options type. It then solves the simulation,
        retrieves the output lines, and saves the simulation results to a file.

        Returns:
            list: The output lines from the simulation.
        """
        logger.info("Starting baseline aging simulation")
        with redirect_stdout_stderr():
            based_sim = AgingSimulation(
                self.param,
                experiment_type=self.experiment_type,
                model_options_type=self.model_options_type,
                aging_cycle_number=self.aging_cycle_number,
                aging_group_number=self.aging_group_number,
            )
            based_sim.solve()
        based_lines = based_sim.get_outputs()
        based_aging_lines = based_sim.get_aging_outputs()
        based_sim.save(self.save_dir / BASED_SIMULATION_FILENAME)
        logger.info("Baseline aging simulation completed and results saved")
        return based_lines, based_aging_lines

    def setup_sp(self):
        """
        Sets up the sensitivity analysis problem specification.

        This method initializes the problem specification for sensitivity analysis
        by defining the parameter names, their bounds, and the output metrics to be
        evaluated (RMSE and MAE for each key in `self.based_lines` except "Time [s]"
        and "Current [A]").

        Returns:
            ProblemSpec: An instance of the ProblemSpec class containing the
            sensitivity analysis configuration.
        """
        outputs_of_sp = []
        for key in self.based_lines:
            if key not in ["Time [s]", "Current [A]"]:
                outputs_of_sp.append(f"delta_p({key})")
                outputs_of_sp.append(f"delta_s_x({key})")
                outputs_of_sp.append(f"delta_s_t({key})")
                outputs_of_sp.append(f"delta_r({key})")
                outputs_of_sp.append(f"delta_all({key})")

        for key in self.based_aging_lines:
            if key not in ["Cycle number"]:
                outputs_of_sp.append(f"delta_p({key})")
                outputs_of_sp.append(f"delta_s_x({key})")
                outputs_of_sp.append(f"delta_s_t({key})")
                outputs_of_sp.append(f"delta_r({key})")
                outputs_of_sp.append(f"delta_all({key})")

        sp = ProblemSpec(
            {
                "names": list(self.sa_param.keys()),
                "groups": None,
                # "bounds": [self.get_bound(value) for value in self.sa_param.values()],
                "bounds": [self.sa_param_bounds[key] for key in self.sa_param.keys()],
                "outputs": outputs_of_sp,
            }
        )
        return sp

    @staticmethod
    def evaluate_place_difference(
        x: np.ndarray, x_t: np.ndarray, xb: np.ndarray, xb_t: np.ndarray
    ) -> float:
        """
        Evaluate the differences between two time series curves in terms of their positions.

        This method calculates the mean difference between two time series curves to evaluate
        the overall position shift. The formula used for the calculation is:

        .. math::
            \delta_p[X(t), X_b(t)] = \frac{1}{N} \sum_{t=1}^{N} X(t) - \frac{1}{N_b} \sum_{t=1}^{N_b} X_b(t)

        where:
        - \( X(t) \) is the first time series curve.
        - \( X_b(t) \) is the second (baseline) time series curve.
        - \( N \) is the number of points in the first time series.
        - \( N_b \) is the number of points in the second (baseline) time series.

        Args:
            x (np.narray): A time series curve for the first set of data.
            x_t (np.narray): The time values corresponding to the first set of data.
            xb (np.narray): A time series curve for the second set of data.
            xb_t (np.narray): The time values corresponding to the second set of data.

        Returns:
            float: The difference between the two curves in terms of their positions.
        """
        delta_p = np.mean(x) - np.mean(xb)
        return delta_p

    @staticmethod
    def evaluate_size_difference(
        x: np.ndarray, x_t: np.ndarray, xb: np.ndarray, xb_t: np.ndarray
    ) -> tuple[float, float]:
        """
        Evaluate the differences between two time series curves in terms of their sizes.

        This method calculates the differences in the time span and variable amplitude
        between two time series curves to evaluate the overall size shift. The formulas
        used for the calculations are:

        .. math::
            \delta_{s,x}[X(t), X_b(t)] = \max(X(t)) - \min(X(t)) - (\max(X_b(t)) - \min(X_b(t)))
            \delta_{s,t}[X(t), X_b(t)] = \max(t) - \min(t) - (\max(t_b) - \min(t_b))

        where:
        - \( X(t) \) is the first time series curve.
        - \( X_b(t) \) is the second (baseline) time series curve.
        - \( t \) is the time values corresponding to the first set of data.
        - \( t_b \) is the time values corresponding to the second (baseline) set of data.

        Args:
            x (np.narray): A time series curve for the first set of data.
            x_t (np.narray): The time values corresponding to the first set of data.
            xb (np.narray): A time series curve for the second set of data.
            xb_t (np.narray): The time values corresponding to the second set of data.

        Returns:
            tuple[float, float]: The differences between the two curves in terms of their sizes.
                - delta_s_x: The difference in variable amplitude.
                - delta_s_t: The difference in time span.
        """
        delta_s_x = (np.max(x) - np.min(x)) - (np.max(xb) - np.min(xb))
        delta_s_t = (np.max(x_t) - np.min(x_t)) - (np.max(xb_t) - np.min(xb_t))
        return delta_s_x, delta_s_t

    @staticmethod
    def evaluate_shape_difference(
        x: np.ndarray, x_t: np.ndarray, xb: np.ndarray, xb_t: np.ndarray
    ) -> float:
        """
        Evaluate the differences between two time series curves in terms of their shapes.

        This method calculates the shape difference between two time series curves by
        correcting the position and size of the first curve to match the second curve,
        and then computing the Root Mean Square Error (RMSE) between the corrected first
        curve and the second curve. The formulas used for the calculations are:

        1. Position correction:
        .. math::
            \delta_p[X(t), X_b(t)] = \frac{1}{N} \sum_{t=1}^{N} X(t) - \frac{1}{N_b} \sum_{t=1}^{N_b} X_b(t)

        2. Size correction:
        .. math::
            X'(t) = X(t) - \delta_p[X(t), X_b(t)]
            X''(t) = \frac{X'(t) - \min(X'(t))}{\max(X'(t)) - \min(X'(t))} + \min(X_b(t))
            X'''(t) = X''(\frac{N_b}{N} t)

        3. Shape difference (RMSE):
        .. math::
            \delta_r[X(t), X_b(t)] = \sqrt{\frac{1}{N_b} \sum_{t=1}^{N_b} (X'''(t) - X_b(t))^2}

        Args:
            x (np.narray): A time series curve for the first set of data.
            x_t (np.narray): The time values corresponding to the first set of data.
            xb (np.narray): A time series curve for the second set of data.
            xb_t (np.narray): The time values corresponding to the second set of data.

        Returns:
            float: The difference between the two curves in terms of their shapes.
        """
        delta_p = np.mean(x) - np.mean(xb)
        x_prime = x - delta_p
        x_double_prime = (x_prime - np.min(x_prime)) / (
            np.max(x_prime) - np.min(x_prime)
        ) + np.min(xb)
        x_triple_prime = np.interp(xb_t, x_t, x_double_prime)
        delta_r = np.sqrt(np.mean((x_triple_prime - xb) ** 2))
        return delta_r

    @staticmethod
    def evaluate_differences(sol_outputs: dict, based_lines: dict) -> dict:
        """
        Evaluate the differences between simulation results and baseline data.

        This method calculates the Root Mean Square Error (RMSE) and Mean Absolute Error (MAE)
        for each key in the provided dictionaries, except for the "Time [s]" key. It interpolates
        the values of the simulation results and baseline data to a common time frame before
        computing the differences.

        Args:
            sol_outputs (dict): A dictionary containing the simulation results with keys as
                    variable names and values as lists of corresponding values.
            based_lines (dict): A dictionary containing the baseline data with keys as
                    variable names and values as lists of corresponding values.

        Returns:
            dict: A dictionary containing the RMSE and MAE for each variable, with keys in the
              format "RMSE(variable_name)" and "MAE(variable_name)".
        """
        logger.debug("Calculating metrics for simulation results")
        evaluate_results = {}
        for key in based_lines.keys():
            if key != "Time [s]":
                delta_p = SensitivityAnalysis.evaluate_place_difference(
                    sol_outputs[key],
                    sol_outputs["Time [s]"],
                    based_lines[key],
                    based_lines["Time [s]"],
                )
                delta_s_x, delta_s_t = SensitivityAnalysis.evaluate_size_difference(
                    sol_outputs[key],
                    sol_outputs["Time [s]"],
                    based_lines[key],
                    based_lines["Time [s]"],
                )
                delta_r = SensitivityAnalysis.evaluate_shape_difference(
                    sol_outputs[key],
                    sol_outputs["Time [s]"],
                    based_lines[key],
                    based_lines["Time [s]"],
                )
                delta_all = delta_p + delta_s_x + delta_r + delta_s_t
                logger.result(
                    f"Calculated {key}'s delta_p: {delta_p}, delta_s_x: {delta_s_x}, delta_s_t: {delta_s_t}, delta_r: {delta_r}"
                )
                evaluate_results.update(
                    {
                        f"delta_p({key})": delta_p,
                        f"delta_s_x({key})": delta_s_x,
                        f"delta_s_t({key})": delta_s_t,
                        f"delta_r({key})": delta_r,
                        f"delta_all({key})": delta_all,
                    }
                )
        return evaluate_results

    @staticmethod
    def evaluate_aging_differences(
        sol_aging_outputs: dict, based_aging_lines: dict
    ) -> dict:
        logger.debug("Calculating metrics for aging simulation results")
        evaluate_results = {}
        for key in based_aging_lines.keys():
            if key != "Cycle number":
                delta_p = SensitivityAnalysis.evaluate_place_difference(
                    sol_aging_outputs[key],
                    sol_aging_outputs["Cycle number"],
                    based_aging_lines[key],
                    based_aging_lines["Cycle number"],
                )
                delta_s_x, delta_s_t = SensitivityAnalysis.evaluate_size_difference(
                    sol_aging_outputs[key],
                    sol_aging_outputs["Cycle number"],
                    based_aging_lines[key],
                    based_aging_lines["Cycle number"],
                )
                delta_r = SensitivityAnalysis.evaluate_shape_difference(
                    sol_aging_outputs[key],
                    sol_aging_outputs["Cycle number"],
                    based_aging_lines[key],
                    based_aging_lines["Cycle number"],
                )
                delta_all = delta_p + delta_s_x + delta_r + delta_s_t
                logger.result(
                    f"Calculated {key}'s delta_p: {delta_p}, delta_s_x: {delta_s_x}, delta_s_t: {delta_s_t}, delta_r: {delta_r}"
                )
                evaluate_results.update(
                    {
                        f"delta_p({key})": delta_p,
                        f"delta_s_x({key})": delta_s_x,
                        f"delta_s_t({key})": delta_s_t,
                        f"delta_r({key})": delta_r,
                        f"delta_all({key})": delta_all,
                    }
                )
        return evaluate_results

    def run_simulation(self, updated_param: dict[str, float], index: int) -> dict:
        """
        Runs a simulation with the given updated parameters and evaluates the results.

        Args:
            updated_param (dict[str, float]): A dictionary containing the parameters to update for the simulation.
            index (int): The index of the simulation run, used for saving the results.

        Returns:
            dict: A dictionary containing the evaluation results and a success flag.
                - "success_simulate" (bool): Indicates whether the simulation was successful.
                - Additional keys for evaluation metrics (e.g., "RMSE(key)", "MAE(key)") if the simulation was successful.
                - If an error occurs, "success_simulate" will be False and evaluation metrics will be NaN.
        """
        try:
            if not self.use_aging:
                sim = Simulation(
                    param=self.param,
                    updated_param=updated_param,
                    experiment_type=self.experiment_type,
                    model_options_type=self.model_options_type,
                )
            elif self.use_aging:
                sim = AgingSimulation(
                    param=self.param,
                    updated_param=updated_param,
                    experiment_type=self.experiment_type,
                    model_options_type=self.model_options_type,
                    aging_cycle_number=self.aging_cycle_number,
                    aging_group_number=self.aging_group_number,
                )

            with redirect_stdout_stderr():
                sim.solve()

            sim.save(self.save_dir / f"{index}_sim.pkl")

            if np.all(sim.outputs["Current [A]"] == 0):
                logger.warning(f"Simulation {index} failed with zero current")
                result = self.create_failure_result()
            else:
                result = {"success_simulate": True}
                result.update(self.evaluate_differences(sim.outputs, self.based_lines))
                if self.use_aging:
                    result.update(
                        self.evaluate_aging_differences(
                            sim.aging_outputs, self.based_aging_lines
                        )
                    )
        except Exception as e:
            logger.error(f"Simulation {index} encountered an error: {e}")
            result = self.create_failure_result()

        return result

    def create_failure_result(self) -> dict:
        result = {"success_simulate": False}
        for key in self.based_lines.keys():
            result.update(
                {
                    f"delta_p({key})": np.nan,
                    f"delta_s_x({key})": np.nan,
                    f"delta_s_t({key})": np.nan,
                    f"delta_r({key})": np.nan,
                    f"delta_all({key})": np.nan,
                }
            )
        for key in self.based_aging_lines.keys():
            result.update(
                {
                    f"delta_p({key})": np.nan,
                    f"delta_s_x({key})": np.nan,
                    f"delta_s_t({key})": np.nan,
                    f"delta_r({key})": np.nan,
                    f"delta_all({key})": np.nan,
                }
            )
        return result

    def parallel_run_simulations(
        self, updated_params: list[dict], max_workers: int = os.cpu_count()
    ) -> list[dict]:
        """
        Run simulations in parallel using ProcessPoolExecutor.

        This method executes the `run_simulation` method for each set of parameters
        in `updated_params` in parallel. The number of parallel workers is determined
        by `max_workers`, which defaults to the number of available CPU cores.

        If `self.use_mpi` is True, the method is currently a no-op (placeholder for MPI implementation).

        Args:
            updated_params (list[dict]): A list of dictionaries, each containing parameters
                         for a single simulation run.
            max_workers (int, optional): The maximum number of worker processes to use. Defaults
                         to the number of CPU cores available.

        Returns:
            list[dict]: A list of dictionaries containing the results of each simulation.
                If a simulation encounters an error, the corresponding result will
                contain {"success_simulate": False}.
        """
        if self.use_mpi:
            # from mpi4py import MPI
            pass
        else:
            logger.info("Starting parallel simulations with ProcessPoolExecutor")
            results = [None] * len(updated_params)

            with tqdm(total=len(updated_params), desc="Running simulations") as pbar:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            self.run_simulation, map_params(updated_param), index
                        ): index
                        for index, updated_param in enumerate(updated_params)
                    }
                    for future in as_completed(futures):
                        index = futures[future]
                        try:
                            results[index] = future.result()
                        except Exception as e:
                            logger.error(
                                f"Simulation {index} encountered an error: {e}"
                            )
                            results[index] = {"success_simulate": False}
                        pbar.update(1)
                        logger.debug(f"Simulation {index} completed")

            logger.info("All parallel simulations completed with ProcessPoolExecutor")
            return results

    def plot_sp_results(self, sp: ProblemSpec, tag: str = ""):
        """
        Plots the sensitivity analysis results for the given problem specification.

        Args:
            sp (ProblemSpec): The problem specification containing the analysis results.
            tag (str, optional): A tag to prepend to the filenames of the saved plots. Defaults to "".

        Returns:
            None

        This method generates and saves bar charts for the main and total effects (S1 and ST) and heatmaps for the
        second-order interaction effects (S2) for each output specified in the problem specification.
        """
        logger.info("Plotting sensitivity analysis results")
        for output in sp.get("outputs"):
            S1ST_data = sp.analysis[output]
            save_path = (
                self.save_dir / f"{tag}_{output}_STS1.png"
                if tag
                else self.save_dir / f"{output}_STS1.png"
            )
            plot_bar_chart(
                x=np.arange(len(sp.get("names"))),
                y=[S1ST_data["S1"], S1ST_data["ST"]],
                yerr=[S1ST_data["S1_conf"], S1ST_data["ST_conf"]],
                labels=sp.get("names"),
                ylabel="Effect Value",
                save_path=save_path,
                legend_labels=["Main Effects (S1)", "Total Effects (ST)"],
            )

            S2_data = sp.analysis[output]["S2"]
            save_path = (
                self.save_dir / f"{tag}_{output}_S2.png"
                if tag
                else self.save_dir / f"{output}_S2.png"
            )
            plot_heatmap(
                data=S2_data,
                labels=sp.get("names"),
                title="Second-order Interaction Effects (S2)",
                save_path=save_path,
            )
            logger.info("Plotting completed")

    def select_random_simulations(self, samples, results_df):
        """
        Selects a random subset of simulation indices that have valid results.

        This method randomly selects up to 5 indices from the given samples that correspond
        to valid simulation results. A valid result is determined by checking if the result
        file exists and if the simulation was successful.

        Parameters:
        -----------
        samples : list
            A list of sample data from which to select indices.
        results_df : pandas.DataFrame
            A DataFrame containing the results of simulations, with a column 'success_simulate'
            indicating whether each simulation was successful.

        Returns:
        --------
        list
            A list of indices corresponding to valid simulation results.
        """
        indices = range(len(samples))
        random_indices = np.random.choice(
            indices, size=min(5, len(indices)), replace=False
        )

        def check_vaild_index(idx):
            exist_flag = (self.save_dir / (str(idx) + SIMULATION_FILENAME)).exists()
            success_flag = results_df.loc[idx, "success_simulate"]
            return exist_flag and success_flag

        exist_indices = [idx for idx in random_indices if check_vaild_index(idx)]

        while len(exist_indices) < 5:
            random_idx = np.random.choice(indices)
            if random_idx not in random_indices and check_vaild_index(random_idx):
                exist_indices.append(random_idx)

        logger.debug(f"Randomly selected simulations for plotting: {exist_indices}")
        return exist_indices

    def plot_simulations(self, simulations_idx_list):
        """
        Plots the simulation results for the given list of simulation indices.

        Parameters:
        -----------
        simulations_idx_list : list of int
            List of indices corresponding to the simulations to be plotted.

        This method generates and saves the following plots for each simulation index:
        - Current vs. Voltage curve
        - Temperature vs. Heating curve (if the model includes thermal options)

        The plots are saved in the directory specified by `self.save_dir`.

        Notes:
        ------
        - The method assumes that the simulation results are stored in files named
          according to the pattern defined by `SIMULATION_FILENAME`.
        - The baseline data for comparison is taken from `self.based_lines`.
        - The method checks if the model includes thermal options by inspecting
          `self.model_options_type`.
        """
        for idx in simulations_idx_list:
            if not self.use_aging:
                sim = Simulation.load(self.save_dir / (str(idx) + SIMULATION_FILENAME))
            elif self.use_aging:
                sim = AgingSimulation.load(
                    self.save_dir / (str(idx) + SIMULATION_FILENAME)
                )

            plot_current_voltage_curve(
                time=sim.outputs["Time [s]"],
                voltage=sim.outputs["Voltage [V]"],
                current=sim.outputs["Current [A]"],
                baseline_voltage=self.based_lines["Voltage [V]"],
                baseline_current=self.based_lines["Current [A]"],
                baseline_time=self.based_lines["Time [s]"],
                save_path=self.save_dir / f"{idx}_current_voltage.png",
            )
            if "thermal" in self.model_options_type:
                plot_temperature_heating_curve(
                    time=sim.outputs["Time [s]"],
                    temperature=sim.outputs["Volume-averaged cell temperature [K]"],
                    heating_power=sim.outputs["Volume-averaged total heating [W.m-3]"],
                    baseline_temperature=self.based_lines[
                        "Volume-averaged cell temperature [K]"
                    ],
                    baseline_heating_power=self.based_lines[
                        "Volume-averaged total heating [W.m-3]"
                    ],
                    baseline_time=self.based_lines["Time [s]"],
                    save_path=self.save_dir / f"{idx}_temperature_heating.png",
                )
            if "degradation" in self.model_options_type:
                plot_capacity_degradation(
                    cycle=sim.aging_outputs["Cycle number"],
                    capacity=sim.aging_outputs["Capacity [A.h]"],
                    discharge_capacity=sim.aging_outputs["Discharge capacity [A.h]"],
                    baseline_cycle=self.based_aging_lines["Cycle number"],
                    baseline_capacity=self.based_aging_lines["Capacity [A.h]"],
                    baseline_discharge_capacity=self.based_aging_lines[
                        "Discharge capacity [A.h]"
                    ],
                    save_path=self.save_dir / f"{idx}_capacity_degeneration.png",
                )

    def load_existing_samples_and_results(self):
        """
        Load existing samples and results from saved files.

        This method reads sample data from a CSV file and results data from another CSV file.
        The sample data is expected to be in a file with a specific delimiter and skipping the first row.
        The results data is read into a pandas DataFrame.

        Returns:
            tuple: A tuple containing:
                - samples (numpy.ndarray): The loaded sample data.
                - results_df (pandas.DataFrame): The loaded results data.
        """
        samples = np.loadtxt(
            self.save_dir / SP_SAMPLES_FILENAME, delimiter=",", skiprows=1
        )
        results_df = pd.read_csv(self.save_dir / SP_RESULTS_FILENAME)
        return samples, results_df

    def analyze_clean_results(self, sp, results_df):
        """
        Analyzes and cleans the results of a sensitivity analysis.

        This method filters out unsuccessful simulation results and performs a Sobol analysis
        on the cleaned data.

        Args:
            sp (ProblemSpec): The problem specification containing the parameters and their bounds.
            results_df (pd.DataFrame): DataFrame containing the results of the simulations.

        Returns:
            ProblemSpec: A new ProblemSpec object with the cleaned results and Sobol analysis.
        """
        results_np = results_df[sp.get("outputs")].to_numpy()
        clean_results_np = np.where(
            results_df["success_simulate"].to_numpy()[:, None], results_np, np.nan
        )
        # clean_results_df = pd.DataFrame(clean_results_np, columns=sp.get("outputs"))
        # clean_results_df.to_csv(self.save_dir / "clean_results.csv", index=False)

        num_clean_results = np.count_nonzero(~np.isnan(clean_results_np[:, 0]))
        total_results = len(results_np)
        logger.info(
            f"Number of clean results: {num_clean_results}, {num_clean_results / total_results * 100:.2f}% of total"
        )
        clean_sp = ProblemSpec(
            {
                "names": sp.get("names"),
                "groups": sp.get("groups"),
                "bounds": sp.get("bounds"),
                "outputs": sp.get("outputs"),
            }
        )
        clean_samples_np = sp.samples
        # Replace failed results in clean_samples_np and clean_results_np with the nearest successful result
        for i in range(len(clean_results_np)):
            if np.isnan(clean_results_np[i, 0]):
                # Find the nearest successful result
                distances = np.linalg.norm(
                    clean_samples_np[i] - clean_samples_np, axis=1
                )
                distances[i] = np.inf  # Ignore the distance to itself
                nearest_idx = np.argmin(distances)
                # If the nearest result is also a failure, continue to find the next one
                while np.isnan(clean_results_np[nearest_idx, 0]):
                    distances[nearest_idx] = np.inf  # Mark as checked
                    nearest_idx = np.argmin(distances)
                clean_results_np[i] = clean_results_np[nearest_idx]
                # Update the corresponding value in clean_samples_np
                clean_samples_np[i] = clean_samples_np[nearest_idx]

        clean_sp.set_samples(clean_samples_np)
        clean_sp.set_results(clean_results_np)
        clean_sp.analyze_sobol()
        return clean_sp

    @staticmethod
    def get_bound(value):
        """
        Calculate the bounds for a given value.

        This function returns a list containing the lower and upper bounds for the input value.
        If the value is zero, it returns [0, 1e-3]. Otherwise, it returns [value * 0.9, value * 1.1].

        Args:
            value (float): The input value for which bounds are to be calculated.

        Returns:
            list: A list containing the lower and upper bounds for the input value.
        """
        logger.debug(f"Calculating bounds for value: {value}")
        if value == 0:
            return [0, 1e-3]
        else:
            return [value * 0.9, value * 1.1]

    @staticmethod
    def get_model_options_type(category: str) -> str:
        """
        Returns a list of model options based on the given category.

        Parameters:
        category (str): The category of the model options.
                        Possible values are "geometry", "electrochemical", "thermal",
                        "degradation", "all", and "test".

        Returns:
        list: A list of model options corresponding to the given category.

        Raises:
        ValueError: If the category is unknown.
        """
        if category == "geometry":
            return []
        elif category == "electrochemical":
            return []
        elif category == "thermal":
            return ["thermal"]
        elif category == "degradation":
            return ["degradation"]
        elif category == "all":
            return ["thermal", "degradation"]
        else:
            raise ValueError(f"Unknown category: {category}")
