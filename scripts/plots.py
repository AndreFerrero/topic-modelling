import os
import matplotlib as plt
import seaborn as sns
import pandas as pd

def plots(evaluation, type: str):
    figures = R'figures'
    figures_folder = os.path.join(path, figures)
        
    for n, metrics in evaluation.items():
        model_names, coherence_values = zip(*metrics)

        # Create a DataFrame for easy plotting with Seaborn
        data = {'Model': model_names, 'Coherence Value': coherence_values}
        df = pd.DataFrame(data)

        plt.figure(figsize=(8, 5))
        
        # Use Seaborn's barplot with the hue parameter
        sns.barplot(x='Model', y='Coherence Value', data=df, hue='Model', palette='viridis')
        
        plt.xlabel('Model')
        plt.ylabel('Coherence Value')
        plt.title(f'Coherence Evaluation for {n} {type}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        save_path = os.path.join(figures_folder, f'coherence_evaluation_{n}_{type}')
        
        plt.savefig(save_path)
        plt.show()