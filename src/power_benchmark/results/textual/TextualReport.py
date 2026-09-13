from power_benchmark.results.data.ResultData import ResultData, ScenarioResult
from typing import List, Dict, Any
from power_benchmark.results.textual.TextFormatter import TextFormatter
from pathlib import Path
import os


class TextualReport:
    def __init__(self, results: ResultData, output_dir: str | Path) -> None:
        self.results = results
        self.output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def report(self, save_report: bool = True, print_report: bool = True):
        if save_report:
            report = self._generate_report(TextFormatter(markdown=True))
            report_path = self.output_dir / Path("benchmark_report.md")

            with open(report_path, 'w') as f:
                f.write(report)
        if print_report:
            print(self._generate_report(TextFormatter(markdown=False)))

    def interactive_report(self):
        formatter = TextFormatter(markdown=False)

        print(self._generate_header(formatter))

        while True:
            print("╔═════════════════════════════════╗")
            print("║   Select Report Sections:       ║")
            print("╠═════════════════════════════════╣")
            print("║   1. Overview                   ║")
            print("║   2. Metrics Overview           ║")
            print("║   3. Scenario Overview          ║")
            print("║   4. Detailed Scenario Reports  ║")
            print("║   0. Exit                       ║")
            print("╚═════════════════════════════════╝")

            choice = input("Enter your choice:\n>>> ")
            if choice.strip() == "0":
                break
            elif choice.strip() == "1":
                print(formatter.horizontal_bar())
                print(formatter.horizontal_bar())
                print(self._generate_overview(formatter))
            elif choice.strip() == "2":
                print(formatter.horizontal_bar())
                print(formatter.horizontal_bar())
                print(self._generate_metrics_overview(formatter))
            elif choice.strip() == "3":
                print(formatter.horizontal_bar())
                print(formatter.horizontal_bar())
                print(self._generate_scenario_overview(formatter))
            elif choice.strip() == "4":
                print("\nSelect Scenario:")
                for idx, scenario in enumerate(self.results.scenarios):
                    print(f"{idx + 1}. {scenario.network_category} - {scenario.name}")
                scenario_choice = input("Enter scenario number (or 0 to go back):\n>>> ")
                if scenario_choice.strip() == "0":
                    continue
                try:
                    scenario_idx = int(scenario_choice.strip()) - 1
                    if 0 <= scenario_idx < len(self.results.scenarios):
                        scenario = self.results.scenarios[scenario_idx]
                        print(formatter.horizontal_bar())
                        print(formatter.horizontal_bar())
                        print(self._generate_single_scenario_report(scenario, formatter))
                    else:
                        print("Invalid scenario number.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
            else:
                print("Invalid choice. Please try again.")

            print("\n\n\n")
            input("[ Press Enter to continue ]")

    def _generate_report(self, formatter: TextFormatter) -> str:
        report_lines: List[str] = [
            self._generate_header(formatter),
            self._generate_overview(formatter),
            self._generate_metrics_overview(formatter),
            self._generate_scenario_overview(formatter),
            formatter.horizontal_bar(),
            *[
                self._generate_single_scenario_report(scenario, formatter)
                for scenario in self.results.scenarios
            ]
        ]

        return "\n".join(report_lines)

    @staticmethod
    def _gathered_count(gathered: Dict[str, Any], metric_name: str) -> Any:
        """metrics_gathered can either be a per-metric mapping or a collapsed
        summary dict (when every metric has the same count). Handle both."""
        if "total_amount_metric_gathered" in gathered:
            # Collapsed summary form: individual counts are all equal.
            amount = gathered.get("amount_of_metrics", 0)
            total = gathered.get("total_amount_metric_gathered", 0)
            return total // amount if amount else 0
        return gathered.get(metric_name, 0)

    def _generate_header(self, formatter: TextFormatter) -> str:
        lines: List[str] = [
            formatter.title(f"Benchmark Report: {self.results.name}", level=1)
        ]

        lines.append(formatter.key_value("Agent", self.results.agent))

        if self.results.description:
            lines.append(formatter.italic(self.results.description) + "\n")

        return "\n".join(lines)

    def _generate_overview(self, formatter: TextFormatter) -> str:
        total_results = sum(len(s.results) for s in self.results.scenarios)
        zero_categories = self._zero_weight_categories()

        content_lines = [
            formatter.key_value("Average Score", f"{self.results.average_score:.4f}"),
            formatter.key_value("Total Scenarios", len(self.results.scenarios)),
            formatter.key_value("Total Results", total_results),
        ]

        for category, score in self.results.average_category_scores.items():
            if category in zero_categories:                      # <-- skip
                continue
            content_lines.append(formatter.key_value(f"Category · {category}", f"{score:.4f}"))

        return formatter.section("Overview", "\n".join(content_lines), level=2)

    def _generate_metrics_overview(self, formatter: TextFormatter) -> str:
        gathered = self.results.metrics_gathered
        skipped = self._skipped_metric_names()  # <-- skip set

        data = [
            {
                "Metric": metric_name,
                "Avg Score": f"{avg['score']:.4f}",
                "Avg Value": f"{avg['value']:.4f}",
                "# Gathered": self._gathered_count(gathered, metric_name),
            }
            for metric_name, avg in self.results.metrics_average.items()
            if not self._is_metric_skipped(metric_name, skipped)  # <-- filter
        ]

        table = formatter.table(data)
        return formatter.section("Metrics Overview", table, level=2)

    def _generate_scenario_overview(self, formatter: TextFormatter) -> str:
        data = [
            {
                "Network Category": scenario.network_category,
                "Scenario": scenario.name,
                "Config": scenario.scenario_config,
                "Avg Score": f"{scenario.average_score:.4f}",
                "# Results": len(scenario.results),
            }
            for scenario in self.results.scenarios
        ]

        table = formatter.table(data)
        return formatter.section("Scenario Overview", table, level=2)

    def _generate_single_scenario_report(self, scenario: ScenarioResult, formatter: TextFormatter) -> str:
        zero_categories = self._zero_weight_categories()
        skipped = self._skipped_metric_names()

        overview_lines = [
            formatter.key_value("Configuration", scenario.scenario_config),
            formatter.key_value("Network Category", scenario.network_category),
            formatter.key_value("Average Score", f"{scenario.average_score:.4f}"),
        ]

        for category, score in scenario.average_category_scores.items():
            if category in zero_categories:  # <-- skip
                continue
            overview_lines.append(formatter.key_value(f"Category · {category}", f"{score:.4f}"))

        overview = formatter.section(
            f"Scenario: {scenario.network_category} - {scenario.name}",
            "\n".join(overview_lines),
            level=3
        )

        gathered = scenario.metrics_gathered
        metrics_data = [
            {
                "Metric": metric_name,
                "Avg Score": f"{avg['score']:.4f}",
                "Avg Value": f"{avg['value']:.4f}",
                "# Gathered": self._gathered_count(gathered, metric_name),
            }
            for metric_name, avg in scenario.metrics_average.items()
            if not self._is_metric_skipped(metric_name, skipped)  # <-- filter
        ]

        metrics_table = formatter.table(metrics_data)
        metrics_section = formatter.section("Metric Overview", metrics_table, level=4)

        return overview + metrics_section

    @staticmethod
    def _is_zero(weight: Any) -> bool:
        """A weight counts as zero if it is 0, or (for per-criterion dicts)
        all of its entries are 0."""
        if isinstance(weight, dict):
            return len(weight) > 0 and all(v == 0 for v in weight.values())
        return weight == 0

    def _zero_weight_categories(self) -> set:
        return {
            category
            for category, weight in self.results.weights.get("total", {}).items()
            if self._is_zero(weight)
        }

    def _skipped_metric_names(self) -> set:
        """Base metric names (without any `[criterion]` suffix) that should be
        hidden because their own weight — or their category's weight — is 0."""
        zero_categories = self._zero_weight_categories()
        skipped: set = set()

        for category, metrics in self.results.weights.items():
            if category == "total" or not isinstance(metrics, dict):
                continue
            category_is_zero = category in zero_categories
            for metric_name, weight in metrics.items():
                if category_is_zero or self._is_zero(weight):
                    skipped.add(metric_name)
        return skipped

    def _is_metric_skipped(self, metric_name: str, skipped: set) -> bool:
        # metrics_average may key entries as "name[criterion]" when several
        # eval criteria are used; match on the base name.
        base_name = metric_name.split("[", 1)[0]
        return base_name in skipped