import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

def plot_scores(json_file_path: str, category_name: str, name: str, version: int, use_months=False, aggregate_by_month=True):
    """
    Plot weighted scores from a JSON file for a specific category and name.

    Args:
        json_file_path: Path to the JSON file
        category_name: The category to extract (e.g., "CategoryNameXY")
        name: The name within the category (e.g., "nameXY")
        use_months: If True, convert day indices to months (default: False)
        aggregate_by_month: If True and use_months=True, aggregate scores by month (default: True)
    """
    # Read the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    entries = None

    if version <= 4:
        # Extract the data for the specified category and name
        if category_name not in data:
            raise ValueError(f"Category '{category_name}' not found in the data")

        if name not in data[category_name]:
            raise ValueError(f"Name '{name}' not found in category '{category_name}'")

        entries = data[category_name][name]
    else:
        entries = data

    # Extract indices and weighted scores
    indices = []
    weighted_scores = []
    amount_of_days = []

    for entry in entries:
        indices.append(entry['index'])
        amount_of_days.append(entry.get('amount_of_days', 1))

        if version == 0:
            weighted_scores.append(entry['total_score']['weighted_score'])
        elif version >= 1:
            weighted_scores.append(entry['total_score'])

    # Create the plot
    plt.figure(figsize=(14, 6))

    if use_months:
        if aggregate_by_month:
            # Aggregate data by month
            month_data = {i: [] for i in range(1, 13)}  # Months 1-12

            for idx, score in zip(indices, weighted_scores):
                # Convert day of year to month (assuming non-leap year)
                date = datetime(2023, 1, 1) + timedelta(days=int(idx))
                month = date.month
                month_data[month].append(score)

            # Calculate average score per month
            months = []
            avg_scores = []
            for month in sorted(month_data.keys()):
                if month_data[month]:  # Only include months with data
                    months.append(month)
                    avg_scores.append(np.mean(month_data[month]))

            # Create bar chart
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            x_labels = [month_names[m-1] for m in months]

            plt.bar(x_labels, avg_scores, width=0.6 * amount_of_days[0], alpha=0.7, edgecolor='black')
            plt.xlabel('Month')
            plt.ylabel('Average Weighted Score')
            plt.title(f'Average Weighted Scores by Month - {category_name}/{name}')

        else:
            # Show all data points with month labels on x-axis
            colors = plt.cm.tab20(np.linspace(0, 1, 12))
            month_colors = []

            for idx in indices:
                date = datetime(2023, 1, 1) + timedelta(days=int(idx))
                month_colors.append(colors[date.month - 1])

            plt.bar(indices, weighted_scores, width=1.0 * amount_of_days[0], alpha=0.7, color=month_colors, edgecolor='none')
            plt.xlabel('Day of Year (colored by month)')
            plt.ylabel('Weighted Score')
            plt.title(f'Weighted Scores by Day - {category_name}/{name}')

            # Add month markers on x-axis
            month_starts = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            plt.xticks(month_starts, month_names)
    else:
        # Standard histogram/bar chart with indices
        plt.bar(indices, weighted_scores, width=1.0, alpha=0.7, edgecolor='black')
        plt.xlabel('Index')
        plt.ylabel('Weighted Score')
        plt.title(f'Weighted Scores - {category_name}/{name}')

    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    # Show the plot
    plt.show()

def plot_worst_months(json_file_path: str, category_name: str, name: str, version: int, n=1, show_all_days=False,
                      use_blocks=False, block_size=7):
    """
    Plot the worst n months or n blocks based on average weighted scores.

    Args:
        json_file_path: Path to the JSON file
        category_name: The category to extract (e.g., "CategoryNameXY")
        name: The name within the category (e.g., "nameXY")
        n: Number of worst months/blocks to display (default: 1)
        show_all_days: If True, show all individual days in the worst months/blocks as bars.
                       If False, show only the average score per month/block (default: False)
        use_blocks: If True, use sliding blocks instead of months (default: False)
        block_size: Size of each block in days (default: 7, only used if use_blocks=True)
    """
    # Read the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    entries = None

    if version <= 4:
        # Extract the data for the specified category and name
        if category_name not in data:
            raise ValueError(f"Category '{category_name}' not found in the data")

        if name not in data[category_name]:
            raise ValueError(f"Name '{name}' not found in category '{category_name}'")

        entries = data[category_name][name]
    else:
        entries = data

    # Extract indices and weighted scores
    indices = []
    weighted_scores = []
    amount_of_days = []

    for entry in entries:
        indices.append(entry['index'])
        amount_of_days.append(entry.get('amount_of_days', 1))

        if version == 0:
            weighted_scores.append(entry['total_score']['weighted_score'])
        elif version >= 1:
            weighted_scores.append(entry['total_score'])

    # Sort by index to ensure proper ordering
    sorted_data = sorted(zip(indices, weighted_scores, amount_of_days), key=lambda x: x[0])
    indices, weighted_scores, amount_of_days = zip(*sorted_data) if sorted_data else ([], [], [])
    indices = list(indices)
    weighted_scores = list(weighted_scores)
    amount_of_days = list(amount_of_days)

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    if use_blocks:
        # Use sliding blocks approach
        if len(indices) < block_size:
            raise ValueError(f"Not enough data points ({len(indices)}) for block size {block_size}")

        # Create sliding windows
        blocks = []
        for i in range(len(indices) - block_size + 1):
            block_indices = indices[i:i+block_size]
            block_scores = weighted_scores[i:i+block_size]

            # Check if block is consecutive (no large gaps)
            if block_indices[-1] - block_indices[0] <= block_size + 5:  # Allow small gaps
                avg_score = np.mean(block_scores)
                blocks.append({
                    'start_idx': block_indices[0],
                    'end_idx': block_indices[-1],
                    'indices': block_indices,
                    'scores': block_scores,
                    'avg_score': avg_score,
                    'block_num': i
                })

        # Sort blocks by average score (ascending - worst first)
        blocks.sort(key=lambda x: x['avg_score'])

        # Get worst n blocks (avoid overlapping blocks)
        worst_blocks = []
        used_indices = set()

        for block in blocks:
            # Check if block overlaps with already selected blocks
            block_idx_set = set(block['indices'])
            if not block_idx_set.intersection(used_indices):
                worst_blocks.append(block)
                used_indices.update(block_idx_set)
                if len(worst_blocks) >= n:
                    break

        best_blocks = []
        used_indices = set()

        for block in reversed(blocks):
            # Check if block overlaps with already selected blocks
            block_idx_set = set(block['indices'])
            if not block_idx_set.intersection(used_indices):
                best_blocks.append(block)
                used_indices.update(block_idx_set)
                if len(best_blocks) >= n:
                    break

        # If we couldn't find enough non-overlapping blocks, allow overlap
        if len(worst_blocks) < n:
            worst_blocks = blocks[:n]

        # Create the plot
        plt.figure(figsize=(14, 6))

        if show_all_days:
            # Show all individual days from the worst blocks
            colors = plt.cm.tab10(np.linspace(0, 1, len(worst_blocks)))

            for i, block in enumerate(worst_blocks):
                plt.bar(block['indices'], block['scores'], width=1.0, alpha=0.7,
                       color=colors[i], edgecolor='black', linewidth=0.5,
                       label=f"Block {i+1}: Days {block['start_idx']}-{block['end_idx']}")

            plt.xlabel('Day of Year')
            plt.ylabel('Weighted Score')
            plt.title(f'Worst {n} Block(s) (Size={block_size} days) - All Days - {category_name}/{name}')
            plt.legend(loc='best')

        else:
            # Show average scores per block
            x_labels = [f"Days\n{b['start_idx']}-{b['end_idx']}" for b in worst_blocks]
            avg_scores = [b['avg_score'] for b in worst_blocks]

            colors = plt.cm.Reds(np.linspace(0.5, 0.9, len(worst_blocks)))
            plt.bar(range(len(worst_blocks)), avg_scores, width=0.6, alpha=0.7,
                   color=colors, edgecolor='black')
            plt.xticks(range(len(worst_blocks)), x_labels)
            plt.xlabel('Block (Day Range)')
            plt.ylabel('Average Weighted Score')
            plt.title(f'Worst {n} Block(s) (Size={block_size} days) - {category_name}/{name}')

            # Add value labels on bars
            for i, score in enumerate(avg_scores):
                plt.text(i, score, f'{score:.2f}', ha='center', va='bottom')

        # Print statistics
        print(f"\n{'='*70}")
        print(f"Worst {n} Block(s) Analysis (Block Size: {block_size} days)")
        print(f"Category: {category_name}/{name}")
        print(f"{'='*70}")
        for i, block in enumerate(worst_blocks, 1):
            min_score = min(block['scores'])
            max_score = max(block['scores'])
            start_date = datetime(2023, 1, 1) + timedelta(days=int(block['start_idx']))
            end_date = datetime(2023, 1, 1) + timedelta(days=int(block['end_idx']))
            print(f"{i}. Days {block['start_idx']:>3}-{block['end_idx']:>3} "
                  f"({start_date.strftime('%b %d')} - {end_date.strftime('%b %d')}): "
                  f"Avg={block['avg_score']:.4f}, Min={min_score:.4f}, Max={max_score:.4f}")
        print(f"{'='*70}\n")

        print(f"\n{'=' * 70}")
        print(f"Best {n} Block(s) Analysis (Block Size: {block_size} days)")
        print(f"Category: {category_name}/{name}")
        print(f"{'=' * 70}")
        for i, block in enumerate(best_blocks, 1):
            min_score = min(block['scores'])
            max_score = max(block['scores'])
            start_date = datetime(2023, 1, 1) + timedelta(days=int(block['start_idx']))
            end_date = datetime(2023, 1, 1) + timedelta(days=int(block['end_idx']))
            print(f"{i}. Days {block['start_idx']:>3}-{block['end_idx']:>3} "
                  f"({start_date.strftime('%b %d')} - {end_date.strftime('%b %d')}): "
                  f"Avg={block['avg_score']:.4f}, Min={min_score:.4f}, Max={max_score:.4f}")
        print(f"{'=' * 70}\n")

    else:
        # Use months approach (original behavior)
        # Aggregate data by month
        month_data = {i: {'scores': [], 'indices': []} for i in range(1, 13)}

        for idx, score in zip(indices, weighted_scores):
            date = datetime(2023, 1, 1) + timedelta(days=int(idx))
            month = date.month
            month_data[month]['scores'].append(score)
            month_data[month]['indices'].append(idx)

        # Calculate average score per month
        month_averages = {}
        for month, data_dict in month_data.items():
            if data_dict['scores']:
                month_averages[month] = np.mean(data_dict['scores'])

        # Sort months by average score (ascending - worst first)
        sorted_months = sorted(month_averages.items(), key=lambda x: x[1])
        worst_months = sorted_months[:min(n, len(sorted_months))]

        # Extract the worst month numbers
        worst_month_numbers = [month for month, _ in worst_months]

        # Create the plot
        plt.figure(figsize=(14, 6))

        if show_all_days:
            # Show all individual days from the worst months
            plot_indices = []
            plot_scores = []
            plot_colors = []

            colors = plt.cm.tab10(np.linspace(0, 1, n))
            color_map = {worst_month_numbers[i]: colors[i] for i in range(len(worst_month_numbers))}

            for idx, score in zip(indices, weighted_scores):
                date = datetime(2023, 1, 1) + timedelta(days=int(idx))
                month = date.month
                if month in worst_month_numbers:
                    plot_indices.append(idx)
                    plot_scores.append(score)
                    plot_colors.append(color_map[month])

            plt.bar(plot_indices, plot_scores, width=1.0 * amount_of_days[0], alpha=0.7,
                   color=plot_colors, edgecolor='black', linewidth=0.5)
            plt.xlabel('Day of Year')
            plt.ylabel('Weighted Score')
            plt.title(f'Worst {n} Month(s) - All Days - {category_name}/{name}')

            # Add legend
            legend_elements = [plt.Rectangle((0,0),1,1, fc=color_map[m], alpha=0.7,
                                            label=month_names[m-1])
                              for m in worst_month_numbers]
            plt.legend(handles=legend_elements, loc='best')

        else:
            # Show average scores per month
            x_labels = [month_names[month-1] for month, _ in worst_months]
            avg_scores = [avg_score for _, avg_score in worst_months]

            colors = plt.cm.Reds(np.linspace(0.5, 0.9, n))
            plt.bar(x_labels, avg_scores, width=0.6, alpha=0.7,
                   color=colors, edgecolor='black')
            plt.xlabel('Month')
            plt.ylabel('Average Weighted Score')
            plt.title(f'Worst {n} Month(s) by Average Score - {category_name}/{name}')

            # Add value labels on bars
            for i, (label, score) in enumerate(zip(x_labels, avg_scores)):
                plt.text(i, score, f'{score:.2f}', ha='center', va='bottom')

        # Print statistics
        print(f"\n{'='*60}")
        print(f"Worst {n} Month(s) Analysis for {category_name}/{name}")
        print(f"{'='*60}")
        for i, (month, avg_score) in enumerate(worst_months, 1):
            num_days = len(month_data[month]['scores'])
            min_score = min(month_data[month]['scores'])
            max_score = max(month_data[month]['scores'])
            print(f"{i}. {month_names[month-1]:>3}: Avg={avg_score:.4f}, "
                  f"Min={min_score:.4f}, Max={max_score:.4f}, Days={num_days}")
        print(f"{'='*60}\n")

    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()