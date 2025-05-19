import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pybamm

from ..utils.logger import get_logger

logger = get_logger(__name__)


class Simulation:
    """
    Simulation class for configuring and running battery simulations using PyBaMM.

    This class provides methods to initialize, configure, and run battery simulations
    based on various experiment types and model options. It supports thermal and degradation
    model options and allows for customization of experiment configurations.

    Attributes:
    sim : pybamm.Simulation
        The configured PyBaMM Simulation object.
    outputs_keys : list[str]
        The keys for the simulation outputs.
    outputs_time_samples : np.ndarray, optional
        Array of time points at which to sample the simulation outputs.

    Methods:
    --------
    create() -> pybamm.Simulation
    get_pybamm_experiment() -> pybamm.Experiment
    solve() -> pybamm.Simulation
    outputs() -> dict[str, np.ndarray]
    get_outputs(outputs_time_samples: np.ndarray = None) -> dict[str, np.ndarray]
    save(file_path: Path) -> None
        Save the simulation data to a file.
    load(file_path: Path) -> "Simulation"
        Load the simulation data from a file.
    get_initial_soc(experiment_type: str) -> float
    get_time_samples_interval(experiment_type: str) -> float
    get_outputs_keys(model_options_type: list[str] = []) -> list[str]
    """

    def __init__(
        self,
        param: pybamm.ParameterValues,
        updated_param: dict = None,
        experiment_type: str = "cccv-cc",
        model_options_type: list[str] = [],
        model_options: dict = {},
        experiment_config: dict = {},
    ):
        """
        Initialize the Simulation class.

        Parameters:
        -----------
        param : pybamm.ParameterValues
            The parameter values for the simulation.
        updated_param : dict, optional
            A dictionary of updated parameters, by default None.
        experiment_type : str, optional
            The type of experiment to run, by default "cccv-cc".
        model_options_type : list[str], optional
            A list of model options types, by default an empty list.
        model_options : dict, optional
            A dictionary of model options, by default an empty dictionary.
        experiment_config : dict, optional
            A dictionary of experiment configurations, by default an empty dictionary.
        """
        self.param = param
        self.updated_param = updated_param
        self.experiment_type = experiment_type
        self.model_options_type = model_options_type
        self.model_options = model_options
        self.experiment_config = experiment_config
        self.sim = self.create()
        self.outputs_keys = self.get_outputs_keys(self.model_options_type)
        self.outputs_time_samples = None

    def create(self) -> pybamm.Simulation:
        """
        Create a PyBaMM Simulation object based on the specified model options.

        This method configures and returns a PyBaMM Simulation object using the
        provided model options, experiment setup, and parameter values. It supports
        thermal and degradation model options.

        Returns:
            pybamm.Simulation: A configured PyBaMM Simulation object.
        """
        if isinstance(self.model_options_type, str):
            self.model_options_type = [self.model_options_type]

        default_th_options = {"thermal": "lumped", "contact resistance": "true"}
        if "thermal" in self.model_options_type:
            self.model_options = {**default_th_options, **self.model_options}

        default_de_options = {
            "SEI": "interstitial-diffusion limited",
            # "SEI porosity change": "true",
            "lithium plating": "partially reversible",
            # "lithium plating porosity change": "true",  # alias for "SEI porosity change"
            "particle mechanics": ("swelling and cracking", "swelling only"),
            "SEI on cracks": "true",
            "loss of active material": "stress-driven",
            "calculate discharge energy": "true",  # for compatibility with older PyBaMM versions
        }
        if "degradation" in self.model_options_type:
            self.model_options = {**default_de_options, **self.model_options}

        model = pybamm.lithium_ion.DFN(options=self.model_options)
        pybamm_experiment = self.get_pybamm_experiment(
            self.experiment_type, self.experiment_config
        )
        sim = pybamm.Simulation(
            model, experiment=pybamm_experiment, parameter_values=self.param
        )
        if self.updated_param is not None:
            sim.parameter_values.update(self.updated_param)
        return sim

    @staticmethod
    def get_pybamm_experiment(
        experiment_type: str, experiment_config: dict
    ) -> pybamm.Experiment:
        """
        Generate a PyBaMM experiment based on the specified experiment type and configuration.

        Returns:
            pybamm.Experiment: The configured PyBaMM experiment.

        Raises:
            ValueError: If the experiment type is unknown.

        Experiment Types:
            - "cccv-cc": Constant current-constant voltage charge and discharge experiment.
                Configurable parameters:
                    - Crate (default: 1)
                    - period (default: "1 s")

            - "pulse-pulse": Pulse discharge and charge experiment.
                Configurable parameters:
                    - Crate (default: 5)
                    - period (default: "0.1 s")
                    - Rest time (default: "10 s")
                    - Pulse time (default: "10 s")
                    - Pulse number (default: 5)

            - "cccv-pulse": Constant current-constant voltage charge followed by pulse discharge experiment.
                Configurable parameters:
                    - Charge Crate (default: 1)
                    - Crate (default: 10)
                    - period (default: "0.1 s")
                    - Rest time (default: "1 min")
                    - Pulse time (default: "10 s")
                    - Pulse number (default: 18)
                    - Voltage cut-off (default: "2.0 V")

            - "xftg": Experiment based on external load profile.
                Configurable parameters:
                    - load_path (default: Path to "xftg_load.csv")
                    - Crate (default: 0.1)
                    - Nominal cell capacity [A.h] (default: 1.1)
                    - period (default: "1 min")
                    - load_sec_np (default: None)
        """
        if experiment_type == "cccv-cc":
            default_config = {
                "Crate": 1,
                "period": "1 s",
            }
            config = {**default_config, **experiment_config}
            return pybamm.Experiment(
                [
                    (
                        # rest 5min
                        # f"Charge at {config['Crate']}C until 3.6 V ({config['period']} period)",
                        # f"Hold at 3.6 V until 50 mA ({config['period']} period)",
                        # f"Rest for 5 min ({config['period']} period)",
                        # f"Discharge at {config['Crate']}C until 2.0 V ({config['period']} period)",
                        # f"Rest for 5 min ({config['period']} period)",
                        # rest 30 min, different period
                        f"Charge at {config['Crate']}C until 3.6 V (1 s period)",
                        "Hold at 3.6 V until 50 mA (1 s period)",
                        "Rest for 30 min (1 min period)",
                        f"Discharge at {config['Crate']}C until 2.0 V (1 s period)",
                        "Rest for 30 min (1 min period)",
                    )
                ]
            )
        elif experiment_type == "pulse-pulse":
            default_config = {
                "Crate": 5,
                "period": "0.1 s",
                "Rest time": "10 s",
                "Pulse time": "10 s",
                "Pulse number": 5,
            }
            config = {**default_config, **experiment_config}
            return pybamm.Experiment(
                [
                    (
                        f"Rest for {config['Rest time']} ({config['period']} period)",
                        f"Discharge at {config['Crate']}C for {config['Pulse time']} ({config['period']} period)",
                        f"Rest for {config['Rest time']} ({config['period']} period)",
                        f"Charge at {config['Crate']/2}C for {config['Pulse time']} ({config['period']} period)",
                    )
                ]
                * config["Pulse number"]
            )
        elif experiment_type == "cccv-pulse":
            default_config = {
                "Charge Crate": 1,
                "Crate": 10,  # pulse Crate
                "period": "0.1 s",
                "Rest time": "1 min",
                "Pulse time": "10 s",
                "Pulse number": 18,
                "Voltage cut-off": "2.0 V",
            }
            config = {**default_config, **experiment_config}
            return pybamm.Experiment(
                [
                    # # rest 5min
                    # (
                    #     f"Charge at {config['Charge Crate']}C until 3.6 V ({config['period']} period)",
                    #     f"Hold at 3.6 V until 50 mA ({config['period']} period)",
                    #     f"Rest for 5 min ({config['period']} period)",
                    # )
                    # + (
                    #     f"Discharge at {config['Crate']}C for {config['Pulse time']} or until {config['Voltage cut-off']} ({config['period']} period)",
                    #     f"Rest for {config['Rest time']} ({config['period']} period)",
                    # )
                    # * config["Pulse number"]
                    # rest 30min, different period
                    (
                        f"Charge at {config['Charge Crate']}C until 3.6 V (1 s period)",
                        "Hold at 3.6 V until 50 mA (1 s period)",
                        "Rest for 30 min (1 min period)",
                    )
                    + (
                        f"Discharge at {config['Crate']}C for {config['Pulse time']} or until {config['Voltage cut-off']} (0.1 s period)",
                        f"Rest for {config['Rest time']} (0.1 s period)",
                    )
                    * config["Pulse number"]
                ]
            )
        elif experiment_type == "xftg":
            default_config = {
                "load_path": Path(__file__).parent / "xftg_load.csv",
                "Crate": 0.1,
                "Nominal cell capacity [A.h]": 1.1,
                "period": "1 min",
                "load_sec_np": None,
            }
            config = {**default_config, **experiment_config}

            if config["load_sec_np"] is None:
                xftg_load = pd.read_csv(config["load_path"])
                # Calculate current and convert to second-level data
                xftg_load["Current [A]"] = (
                    xftg_load["Normalized Load"]
                    * config["Crate"]
                    * config["Nominal cell capacity [A.h]"]
                )
                # Convert the time column from minutes to seconds
                xftg_load["Time [sec]"] = xftg_load["Time [min]"] * 60
                # Reset the index to a column and convert to numpy array
                load_sec_np = xftg_load[["Time [sec]", "Current [A]"]].to_numpy()
            else:
                load_sec_np = config["load_sec_np"]

            return pybamm.Experiment(
                [
                    pybamm.step.current(
                        load_sec_np,
                        duration=f"{load_sec_np[-1, 0]} s",
                        period=config["period"],
                    )
                ]
            )
        else:
            raise ValueError(f"Unknown experiment: {experiment_type}")

    def solve(self) -> pybamm.Simulation:
        """
        Solves the battery simulation using the initial state of charge (SOC).

        This method retrieves the initial SOC based on the experiment type and
        solves the simulation using this initial SOC.

        Returns:
            pybamm.Simulation: The solved simulation object.
        """
        initial_soc = self.get_initial_soc(self.experiment_type)
        self.sol = self.sim.solve(initial_soc=initial_soc)

        if not hasattr(self, "_outputs"):
            self.get_outputs()
        return self.sol

    @property
    def outputs(self) -> dict[str, np.ndarray]:
        """
        Property to get the outputs of the simulation.

        Returns:
        dict: A dictionary containing the sampled outputs.
        """
        return self._outputs

    def get_outputs(
        self, outputs_time_samples: np.ndarray = None
    ) -> dict[str, np.ndarray]:
        """
        Get the outputs of the simulation at specified time samples.

        Parameters:
        outputs_time_samples (np.ndarray): Array of time points at which to sample the simulation outputs.

        Returns:
        dict: A dictionary containing the sampled outputs.
        """
        if outputs_time_samples is None:
            outputs_time_samples = np.arange(
                0,
                int(self.sim.solution["Time [s]"].entries[-1]),
                self.get_time_samples_interval(self.experiment_type),
            )
        self.outputs_time_samples = outputs_time_samples

        outputs = {}
        for key in self.outputs_keys:
            outputs[key] = self.sim.solution[key](outputs_time_samples)
        self._outputs = outputs

        return outputs

    def _get_save_data(self) -> dict:
        """
        Collects and returns the simulation data as a dictionary.

        Returns:
            dict: A dictionary containing the following keys and their corresponding values:
                - "param": The parameters used in the simulation.
                - "updated_param": The updated parameters after the simulation.
                - "experiment_type": The type of experiment conducted.
                - "model_options_type": The type of model options used.
                - "model_options": The specific model options used.
                - "experiment_config": The configuration of the experiment.
                - "outputs_keys": The keys for the output data.
                - "_outputs": The output data from the simulation.
                - "outputs_time_samples": The time samples for the output data.
        """
        return {
            "param": self.param,
            "updated_param": self.updated_param,
            "experiment_type": self.experiment_type,
            "model_options_type": self.model_options_type,
            "model_options": self.model_options,
            "experiment_config": self.experiment_config,
            "outputs_keys": self.outputs_keys,
            "_outputs": self.outputs,
            "outputs_time_samples": self.outputs_time_samples,
        }

    def save(self, file_path: Path) -> None:
        data = self._get_save_data()
        try:
            with open(file_path, "wb") as file:
                pickle.dump(data, file)
        except IOError as e:
            logger.error(f"Error saving simulation data to {file_path}: {e}")

    @classmethod
    def load(cls, file_path: Path) -> "Simulation":
        """
        Load a Simulation object from a file.

        Args:
            file_path (Path): The path to the file containing the serialized Simulation object.

        Returns:
            Simulation: The deserialized Simulation object if loading is successful, otherwise None.

        Raises:
            IOError: If there is an error opening or reading the file.

        Notes:
            The file should contain a dictionary with the following keys:
                - "param": Parameters for the simulation.
                - "updated_param": Updated parameters for the simulation.
                - "experiment_type": Type of the experiment.
                - "model_options_type": Type of model options.
                - "model_options": Model options.
                - "experiment_config": Configuration of the experiment.
                - "outputs_keys": Keys for the outputs.
                - "_outputs": Outputs of the simulation.
                - "outputs_time_samples": Time samples for the outputs.
        """
        try:
            with open(file_path, "rb") as file:
                data = pickle.load(file)
        except IOError as e:
            logger.error(f"Error loading simulation data from {file_path}: {e}")
            return None

        param = data["param"]
        updated_param = data["updated_param"]
        experiment_type = data["experiment_type"]
        model_options_type = data["model_options_type"]
        model_options = data["model_options"]
        experiment_config = data["experiment_config"]

        sim = cls(
            param,
            updated_param,
            experiment_type,
            model_options_type,
            model_options,
            experiment_config,
        )
        sim.outputs_keys = data["outputs_keys"]
        sim._outputs = data["_outputs"]
        sim.outputs_time_samples = data["outputs_time_samples"]

        return sim

    @staticmethod
    def get_initial_soc(experiment_type: str) -> float:
        """
        Get the initial state of charge (SOC) based on the experiment type.

        Parameters:
        experiment_type (str): The type of experiment. Supported values are "cccv-cc", "pulse-pulse", "cccv-pulse", and "xftg".

        Returns:
        float: The initial SOC value. Returns 0 for "cccv-cc", 1 for "pulse-pulse", 0 for "cccv-pulse", and 0 for "xftg".

        Raises:
        ValueError: If the experiment type is unknown.
        """
        if experiment_type == "cccv-cc":
            return 0
        elif experiment_type == "pulse-pulse":
            return 1
        elif experiment_type == "cccv-pulse":
            return 0
        elif experiment_type == "xftg":
            return 0
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")

    @staticmethod
    def get_time_samples_interval(experiment_type: str) -> float:
        """
        Get the time interval for sampling based on the experiment type.

        Parameters:
        experiment_type (str): The type of experiment. Supported values are "cccv-cc", "pulse-pulse", "cccv-pulse", and "xftg".

        Returns:
        float: The time interval for sampling. Returns 1 for "cccv-cc", 0.1 for "pulse-pulse" and "cccv-pulse", and 60 for "xftg".

        Raises:
        ValueError: If the experiment type is unknown.
        """
        if experiment_type == "cccv-cc":
            return 1
        elif experiment_type == "pulse-pulse":
            return 0.1
        elif experiment_type == "cccv-pulse":
            return 1
        elif experiment_type == "xftg":
            return 60
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")

    @staticmethod
    def get_outputs_keys(model_options_type: list[str] = []) -> list[str]:
        """
        Generate a list of output keys based on the provided model options.

        This function returns a list of keys that represent the outputs of a simulation.
        The base outputs include time, voltage, and current. If the "thermal" option is
        included in the model options, additional thermal-related outputs are added.

        Args:
            model_options_type (list[str]): A list of strings representing the model options.
                If "thermal" is included in the list, thermal-related outputs will be added.

        Returns:
            dict: A dictionary containing the keys for the simulation outputs. The keys include
            "Time [s]", "Voltage [V]", and "Current [A]". If "thermal" is in the model options,
            additional keys "Volume-averaged cell temperature [K]" and "Volume-averaged total heating [W.m-3]"
            are included.
        """
        based_outputs_keys = ["Time [s]", "Voltage [V]", "Current [A]"]
        if "thermal" in model_options_type:
            extra_outputs_keys = [
                "Volume-averaged cell temperature [K]",
                "Volume-averaged total heating [W.m-3]",
            ]
        else:
            extra_outputs_keys = []
        return based_outputs_keys + extra_outputs_keys


class AgingSimulation(Simulation):
    def __init__(
        self,
        param: pybamm.ParameterValues,
        updated_param: dict = None,
        experiment_type: str = "cccv-cc",
        model_options_type: list[str] = ["degradation"],
        model_options: dict = {},
        experiment_config: dict = {},
        aging_cycle_number: int = 20,
        aging_group_number: int = 10,
    ):
        """
        Initialize an aging simulation with specified parameters and experiment type.

        Parameters:
        param (pybamm.ParameterValues): The parameter values to use for the simulation.
        updated_param (dict, optional): A dictionary of parameters to update in the simulation. Defaults to None.
        experiment_type (str, optional): The type of experiment to run. Defaults to "cc-cccv".
        model_options_type (list[str], optional): A list of model option types to include. Defaults to an empty list.
            Available options:
            - "thermal": Includes thermal effects with default options.
        model_options (dict, optional): A dictionary of model options. Defaults to an empty dictionary.
            This can override the default options set by model_options_type.
        aging_cycle_number (int, optional): The number of aging cycles in each group. Defaults to 100.
        aging_group_number (int, optional): The number of aging groups. Defaults to 10.
        """
        self.aging_experiment_type = experiment_type
        self.aging_cycle_number = aging_cycle_number
        self.aging_group_number = aging_group_number
        self.save_at_cycles, self.standard_exp_list, self.group_last_exp_list = (
            self.get_save_at_cycles(self.aging_cycle_number, self.aging_group_number)
        )
        super().__init__(
            param,
            updated_param,
            experiment_type,
            model_options_type,
            model_options,
            experiment_config,
        )

    def create(self) -> pybamm.Simulation:
        """
        Create a PyBaMM Simulation object based on the specified model options.

        This method configures and returns a PyBaMM Simulation object using the
        provided model options, experiment setup, and parameter values. It supports
        thermal and degradation model options.

        Returns:
            pybamm.Simulation: A configured PyBaMM Simulation object.
        """
        sim = super().create()

        pybamm_experiment = self.get_aging_pybamm_experiment()
        sim.experiment = pybamm_experiment

        return sim

    def get_aging_pybamm_experiment(self) -> pybamm.Experiment:
        # 1. 1C cccv-cc 定容
        # 2. aging_experiment_type * aging_cycle_number
        # 3. 1C cccv-cc 定容
        # 4. [2, 3] * aging_group_number
        standard_experiment = self.get_pybamm_experiment(
            experiment_type="cccv-cc", experiment_config={"Crate": 1}
        )
        aging_cycle_experiecne = self.get_pybamm_experiment(
            experiment_type=self.aging_experiment_type,
            experiment_config=self.experiment_config,
        )
        return pybamm.Experiment(
            standard_experiment.cycles
            + (
                aging_cycle_experiecne.cycles * self.aging_cycle_number
                + standard_experiment.cycles
            )
            * self.aging_group_number
        )

    def solve(self) -> pybamm.Simulation:
        """
        Solves the aging simulation with the specified initial state of charge (SOC).

        Parameters:
        initial_soc (float): The initial state of charge for the simulation.

        Returns:
        pybamm.Simulation: The solved PyBaMM simulation object.
        """
        initial_soc = self.get_initial_soc(self.experiment_type)
        logger.debug(f"Save at cycles: {self.save_at_cycles}")
        self.sol = self.sim.solve(
            initial_soc=initial_soc, save_at_cycles=self.save_at_cycles
        )

        if not hasattr(self, "_outputs_list"):
            self.get_outputs_list()
        if not hasattr(self, "_outputs"):
            self.get_outputs()
        if not hasattr(self, "_aging_outputs"):
            self.get_aging_outputs()
        return self.sol

    # outputs_list 为 aging_group_number个 outputs
    # outputs 保存最后一个aging_cycle 的结果
    # aging_outputs 保存 summary_variables 以 Cycle number为横轴 以标定的结果为准

    @property
    def outputs_list(self) -> list[dict[str, np.ndarray]]:
        """
        Property to get the outputs of the simulation.

        Returns:
        dict: A dictionary containing the sampled outputs.
        """
        return self._outputs_list

    @property
    def outputs(self) -> dict[str, np.ndarray]:
        """
        Property to get the outputs of the simulation.

        Returns:
        dict: A dictionary containing the sampled outputs.
        """
        return self._outputs

    @property
    def aging_outputs(self) -> dict[str, np.ndarray]:
        """
        Property to get the outputs of the simulation.

        Returns:
        dict: A dictionary containing the sampled outputs.
        """
        return self._aging_outputs

    def get_outputs_list(
        self, outputs_time_samples: np.ndarray = None
    ) -> list[dict[str, np.ndarray]]:
        """
        Get the outputs of the simulation at specified time samples.

        Parameters:
        outputs_time_samples (np.ndarray): Array of time points at which to sample the simulation outputs.

        Returns:
        dict: A dictionary containing the sampled outputs.
        """
        if outputs_time_samples is None:
            outputs_time_samples_list = [None] * len(self.group_last_exp_list)
            for idx, cycle in enumerate(self.group_last_exp_list):
                outputs_time_samples = np.arange(
                    np.ceil(self.sim.solution.cycles[cycle]["Time [s]"].entries[0]),
                    int(self.sim.solution.cycles[cycle]["Time [s]"].entries[-1]),
                    self.get_time_samples_interval(self.aging_experiment_type),
                )
                outputs_time_samples_list[idx] = outputs_time_samples
        self.outputs_time_samples_list = outputs_time_samples_list

        outputs_list = [None] * len(self.group_last_exp_list)
        for idx, cycle in enumerate(self.group_last_exp_list):
            outputs = {}
            for key in self.outputs_keys:
                outputs[key] = self.sim.solution.cycles[cycle][key](
                    outputs_time_samples_list[idx]
                )
            outputs_list[idx] = outputs
        self._outputs_list = outputs_list

        return outputs_list

    def get_outputs(
        self, outputs_time_samples: np.ndarray = None
    ) -> dict[str, np.ndarray]:
        if outputs_time_samples is None:
            outputs_time_samples = np.arange(
                np.ceil(self.sim.solution.cycles[-1]["Time [s]"].entries[0]),
                int(self.sim.solution.cycles[-1]["Time [s]"].entries[-1]),
                self.get_time_samples_interval(self.experiment_type),
            )
        self.outputs_time_samples = outputs_time_samples

        outputs = {}
        for key in self.outputs_keys:
            outputs[key] = self.sim.solution.cycles[-1][key](outputs_time_samples)
        self._outputs = outputs

        return outputs

    def get_aging_outputs(self) -> dict[str, np.ndarray]:
        """
        use standard exp
        """
        aging_outputs = {}

        # 'Cycle number'
        aging_outputs["Cycle number"] = self.sim.solution.summary_variables[
            "Cycle number"
        ][self.standard_exp_list]

        # 'Capacity [A.h]'
        aging_outputs["Capacity [A.h]"] = self.sim.solution.summary_variables[
            "Capacity [A.h]"
        ][self.standard_exp_list]

        # 'Discharge capacity [A.h]'
        def get_discharge_capacity(cycle):
            discharge_capacity_data = self.sim.solution.cycles[cycle][
                "Discharge capacity [A.h]"
            ].entries
            return discharge_capacity_data[-1] - discharge_capacity_data.min()

        aging_outputs["Discharge capacity [A.h]"] = np.array(
            [get_discharge_capacity(cycle) for cycle in self.standard_exp_list]
        )

        self.aging_outputs_keys = list(aging_outputs.keys())
        self._aging_outputs = aging_outputs

        return aging_outputs

    def _get_save_data(self):
        super_data = super()._get_save_data()
        return {
            **super_data,
            "aging_experiment_type": self.aging_experiment_type,
            "aging_cycle_number": self.aging_cycle_number,
            "aging_group_number": self.aging_group_number,
            "save_at_cycles": self.save_at_cycles,
            "standard_exp_list": self.standard_exp_list,
            "group_last_exp_list": self.group_last_exp_list,
            "aging_outputs_keys": self.aging_outputs_keys,
            "_aging_outputs": self.aging_outputs,
            "outputs_time_samples_list": self.outputs_time_samples_list,
            "_outputs_list": self.outputs_list,
        }

    def save(self, file_path: Path) -> None:
        data = self._get_save_data()
        try:
            with open(file_path, "wb") as file:
                pickle.dump(data, file)
        except IOError as e:
            logger.error(f"Error saving simulation data to {file_path}: {e}")

    @classmethod
    def load(cls, file_path: Path) -> "AgingSimulation":
        try:
            with open(file_path, "rb") as file:
                data = pickle.load(file)
        except IOError as e:
            logger.error(f"Error loading simulation data from {file_path}: {e}")
            return None

        param = data["param"]
        updated_param = data["updated_param"]
        # experiment_type = data["experiment_type"]
        model_options_type = data["model_options_type"]
        model_options = data["model_options"]
        experiment_config = data["experiment_config"]
        aging_experiment_type = data["aging_experiment_type"]
        aging_cycle_number = data["aging_cycle_number"]
        aging_group_number = data["aging_group_number"]

        sim = cls(
            param,
            updated_param,
            aging_experiment_type,
            model_options_type,
            model_options,
            experiment_config,
            aging_cycle_number,
            aging_group_number,
        )
        sim.outputs_keys = data["outputs_keys"]
        sim._outputs = data["_outputs"]
        sim.outputs_time_samples = data["outputs_time_samples"]
        sim.aging_outputs_keys = data["aging_outputs_keys"]
        sim._aging_outputs = data["_aging_outputs"]
        sim.outputs_time_samples_list = data["outputs_time_samples_list"]
        sim._outputs_list = data["_outputs_list"]

        return sim

    @staticmethod
    def get_save_at_cycles(
        aging_cycle_number: int, aging_group_number: int
    ) -> list[int]:
        """
        Get the list of cycles at which to save the simulation results.

        Parameters:
        aging_cycle_number (int): The number of aging cycles in each group.
        aging_group_number (int): The number of aging groups.

        Returns:
        list[int]: The list of cycles at which to save the simulation results.
        """
        standard_exp_list = [0] + [
            (aging_cycle_number + 1) * (n_group + 1)
            for n_group in range(aging_group_number)
        ]
        group_last_exp_list = [
            (aging_cycle_number + 1) * n_group + aging_cycle_number
            for n_group in range(aging_group_number)
        ]
        save_at_cycles = list(standard_exp_list + group_last_exp_list)
        save_at_cycles.sort()
        save_at_cycles = [
            cycle_index + 1 for cycle_index in save_at_cycles
        ]  # 0-based to 1-based for PyBaMM
        return save_at_cycles, standard_exp_list, group_last_exp_list
