from enum import Flag, auto
from typing import List


class Task(Flag):
    NONE = 0

    TUNE = auto()         # 1    Tune model
    TRAIN = auto()        # 2    Train model from scratch
    LOAD = auto()         # 4    Load pretrained model
    RUN = auto()          # 8    Run benchmark scenarios
    EVALUATE = auto()     # 16   Evaluate results and compute scores
    STATISTICS = auto()   # 32   Create meta-data and statistics in case they did not load
    REPLAY = auto()       # 64   Replay actions and save plot of each step

    PLOT = auto()         # 128  Plot results
    WEBAPP = auto()       # 256  Start webapp for result exploration
    TEXTUAL = auto()      # 512  Generate textual report (alternative to webapp)
    INTERACTIVE = auto()  # 1024 Start interactive mode (CLI-based)

    COMPUTE_TASKS = TUNE | TRAIN | LOAD | RUN | EVALUATE | STATISTICS | REPLAY
    PRESENTER_TASKS = PLOT | WEBAPP | TEXTUAL | INTERACTIVE

    def validate(self,
        agent_type: str = None,
        output_dir: str = "",
    ) -> 'Task':
        """Validate flag combinations."""
        if Task.TRAIN in self and Task.LOAD in self:
            raise ValueError("TRAIN and LOAD cannot be combined.")

        if agent_type in ["DoNothing", "Greedy"]:
            if Task.TRAIN in self or Task.LOAD in self:
               raise ValueError(f"TRAIN and LOAD tasks are not applicable for Do Nothing or Greedy agents.")
        else:
            if Task.TRAIN not in self and Task.LOAD not in self and Task.RUN in self:
                raise ValueError(f"Either TRAIN or LOAD task must be specified for custom agents.")

        if Task.EVALUATE in self and output_dir == "":
            raise ValueError("Output directory must be specified for EVALUATE task.")
        return self

    @classmethod
    def from_list(cls,
        tasks: List[str],
        agent_type: str = None,
        output_dir: str = "",
    ) -> 'Task':
        task_flag = cls.NONE
        for task_str in tasks:
            try:
                task_flag |= cls[task_str.upper()]
            except KeyError:
                raise ValueError(f"Invalid task: {task_str}")
        return task_flag.validate(agent_type=agent_type, output_dir=output_dir)