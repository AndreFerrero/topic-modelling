import pandas as pd
import matplotlib as plt
import os

def tables(evaluation, type: str):
    # Specify the results folder
    results = R"results"
    results_folder = os.path.join(path, results)

    # Create individual DataFrames for each specific number
    dfs = {}
    for n, values in evaluation.items():
        dfs[n] = pd.DataFrame(values, columns=['Model', 'Coherence'])

    # Save each DataFrame as a PNG file with a title in the specified folder
    for n, df in dfs.items():
        fig, ax = plt.subplots(figsize=(4, 3))  # Adjust the figure size
        ax.axis('off')  # Turn off the axis

        # Set the width of the columns
        col_width = 1.0 / len(df.columns)
        cell_data = [df.columns] + df.values.tolist()  # Include column names as the first row
        table = ax.table(cellText=cell_data, loc='center', cellLoc='center', colLabels=None, edges='open')

        # Make column labels bold
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_text_props(fontweight='bold')

        # Adjust column width
        table.auto_set_column_width([0, 1])

        # Adjust the position of the table within the figure
        table.set_fontsize(13)  # Adjust font size
        table.scale(1, 2)  # Scale the table

        ax.set_title(f'Coherence with {n} {type}', fontsize=18, y=0.95)  # Add a title

        filename = os.path.join(results_folder, f'table_{n}_{type}.png')
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)  # Adjust padding
        plt.close()  # Close the figure to avoid overlapping when saving multiple files