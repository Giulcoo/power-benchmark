import logging
from typing import List, Tuple, Optional, Union, Dict

from power_benchmark.constants.types import VALUE_LITERAL, VALUE_TYPE, VALUE_SCORE_TYPES

logger = logging.getLogger("Benchmarker")

class ScoreEntry:
    def __init__(self,
         name: str,
         values: Union[List[float], VALUE_TYPE, int],
         score_func: Union['MinMaxFunction', Dict[VALUE_LITERAL, 'MinMaxFunction']],
         n_digits_rounding: int = 2
    ):
        self.name = name
        self.values = values
        self.n_digits_rounding = n_digits_rounding

        self.scores: Union[List[float], VALUE_TYPE]

        if isinstance(values, dict):
            if isinstance(score_func, dict):
                self.scores = {key: score_func[key].calc(value) for key, value in values.items()}
            else:
                self.scores = {key: score_func.calc(value) for key, value in values.items()}
        elif isinstance(values, float) or isinstance(values, int):
            if isinstance(score_func, dict):
                raise ValueError("Score function must be a single function for scalar values.")
            self.scores = score_func.calc(values)
        elif isinstance(values, list):
            if isinstance(score_func, dict):
                raise ValueError("Score function must be a single function for a list of values.")
            self.scores = [score_func.calc(value) for value in values]
        else:
            raise ValueError(f"Invalid type for values. Expected list, dict, float or int. Got {type(values)} instead.")

        self.round_score()

    def round_score(self):
        if isinstance(self.scores, dict):
            self.scores = {key: round(score, self.n_digits_rounding) for key, score in self.scores.items()}
        elif isinstance(self.scores, float):
            self.scores = round(self.scores, self.n_digits_rounding)
        elif isinstance(self.scores, list):
            self.scores = [round(score, self.n_digits_rounding) for score in self.scores]
        else:
            raise ValueError(f"Invalid type for scores. Expected list, dict, float or int. Got {type(self.scores)} instead.")


    def zip_results(self) -> VALUE_SCORE_TYPES:
        if isinstance(self.values, dict) and isinstance(self.scores, dict):
            return {key: (value, self.scores[key]) for key, value in self.values.items()}
        elif isinstance(self.values, (int, float)) and isinstance(self.scores, (int, float)):
            return float(self.values), float(self.scores)
        elif isinstance(self.values, list) and isinstance(self.scores, list):
            return list(zip(self.values, self.scores))
        else:
            raise ValueError(f"Invalid type for values. Expected list, dict, float or int. Got {type(self.values)} instead.")

class ScoreFunction:
    def calc(self, value: float) -> float:
        return 0.0

    def print(
            self,
            x_range: Tuple[float, float] | float,
            num_points: int = 200
    ):
        """
        Print this ScoreFunction over a specified x-range.

        Parameters:
        -----------
        x_range : Tuple[float, float] | float
            The (min, max) range for the x-axis or a single float for (0, max)
        num_points : int
            Number of points to evaluate (default: 1000)
        print_steps : int
            Number of steps to print (default: 10)
        """
        import numpy as np

        # Generate x values
        if isinstance(x_range, (int, float)):
            x_range = (0.0, x_range)
        x_values = np.linspace(x_range[0], x_range[1], num_points)

        # Calculate y values using the score function
        y_values = np.array([self.calc(x) for x in x_values])

        # Create pandas DataFrame for better formatting
        import pandas as pd
        df = pd.DataFrame({
            "x": x_values,
            "score": y_values
        })
        print(df.to_string(index=False, float_format="{:.2f}".format))

    def plot(
        self,
        x_range: Tuple[float, float] | float,
        num_points: int = 200,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (10, 6)
    ):
        """
        Plot this ScoreFunction over a specified x-range.

        Parameters:
        -----------
        x_range : Tuple[float, float] | float
            The (min, max) range for the x-axis or a single float for (0, max)
        num_points : int
            Number of points to evaluate (default: 1000)
        title : Optional[str]
            Title of the plot (default: class name)
        figsize : Tuple[float, float]
            Size of the figure (default: (10, 6))

        Returns:
        --------
        plt.Axes : The axes object used for plotting
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # Generate x values
        if isinstance(x_range, (int, float)):
            x_range = (0.0, x_range)
        x_values = np.linspace(x_range[0], x_range[1], num_points)

        # Calculate y values using the score function
        y_values = np.array([self.calc(x) for x in x_values])

        # Create plot or use existing axes
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(x_values, y_values, color="blue", linewidth=2.0)

        # Set title (default to class name if not provided)
        if title is None:
            title = self.__class__.__name__
        ax.set_title(title, fontsize=14)

        # Labels and grid
        ax.set_xlabel("Value", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()

class MinMaxFunction(ScoreFunction):
    def __init__(self, min_value: float, max_value: float, multiplier: float = 100.0):
        self.min_value = min_value
        self.max_value = max_value
        self.reverse = False

        if max_value < min_value:
            self.min_value = max_value
            self.max_value = min_value
            self.reverse = True

        self.multiplier = multiplier

    def calc(self, value: float) -> float:
        if self.max_value == self.min_value:
            return 1.0 if value == self.max_value else 0.0

        if value <= self.min_value:
            return 0.0 if not self.reverse else self.multiplier
        elif value >= self.max_value:
            return self.multiplier if not self.reverse else 0.0
        else:
            result = (value - self.min_value) / (self.max_value - self.min_value)
            if self.reverse:
                result = 1.0 - result
            return result * self.multiplier