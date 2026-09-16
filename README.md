# Power Benchmark

Power Benchmark is a configurable benchmarking framework for evaluating and comparing reinforcement learning agents for power grid operation.

It was developed as part of the Master's thesis:

**Power-Benchmark: Benchmarking RL-agents for Power Grids**  
Giulio Maximilian Pazzi  
University of Kassel, 2026

## Features

- Configurable power grid scenarios
- Multiple benchmark metrics and evaluation categories
- Hierarchical scoring from metric scores to an overall benchmark score
- Baseline and custom reinforcement learning agents
- Hyperparameter tuning support
- Textual and visual result analysis
- Replay functionality for inspecting agent actions and grid states
- Optional N-1 power flow evaluation

## Documentation

The full installation, configuration, usage, and development documentation is available in the GitHub Wiki:

[Power Benchmark Wiki](https://github.com/Giulcoo/power-benchmark/wiki)

## PPTopoGym

Power Benchmark uses **PPTopoGym** as the underlying reinforcement learning environment.

PPTopoGym is included as a Git submodule. Clone the repository recursively:

```bash
git clone --recurse-submodules https://github.com/Giulcoo/power-benchmark.git
```

If the repository was already cloned without its submodules:

```bash
git submodule update --init --recursive
```

The included PPTopoGym fork is maintained separately and remains subject to its own license and copyright terms.

## Master's Thesis

The associated thesis repository contains the thesis source, final PDF, experiment configurations, and experiment results:

[power-benchmark-thesis](https://github.com/Giulcoo/power-benchmark-thesis)

## Citation

```bibtex
@mastersthesis{pazzi2026powerbenchmark,
  author  = {Giulio Maximilian Pazzi},
  title   = {Power-Benchmark: Benchmarking RL-agents for Power Grids},
  school  = {University of Kassel},
  year    = {2026},
  type    = {Master's Thesis},
  address = {Kassel, Germany},
  month   = sep
}
```

## License

The Power Benchmark software is licensed under the **MIT License**.

See the [`LICENSE`](LICENSE) file for details.

The documentation in the GitHub Wiki is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

The included PPTopoGym submodule remains subject to its own license and copyright terms.