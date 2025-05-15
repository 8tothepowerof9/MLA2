from test.test_rebalance import get_balanced_dataframe
import pandas as pd
from utils.inspectors import plot_label_distribution
from collections import Counter
from test.test_rebalance import get_balanced_dataframe

def visualize_balanced_distribution(csv_path, label_col="label", strategy="oversample"):
    """
    Loads CSV, applies balancing strategy, and plots class distribution.

    Args:
        csv_path (str): Path to the metadata CSV.
        label_col (str): Name of the label column.
        strategy (str): Balancing strategy ("weights", "oversample", "undersample").
    """
    # === Load data
    df = pd.read_csv(csv_path)

    # === Apply balancing
    weights_tensor, balanced_df = get_balanced_dataframe(df, label_col=label_col, strategy=strategy)

    # === Handle 'weights' strategy
    if balanced_df is None:
        print("'weights' strategy does not modify the dataframe. Displaying original class distribution.")
        label_counter = Counter(df[label_col])
        class_names = sorted(label_counter.keys())
        plot_label_distribution(label_counter, class_names)
        print(f"Class weights: {weights_tensor.tolist()}")
        return

    # === Plot balanced distribution
    label_counter = Counter(balanced_df[label_col])
    class_names = sorted(label_counter.keys())
    plot_label_distribution(label_counter, class_names)
    print(f"Successfully plotted balanced distribution using strategy: {strategy}")


# visualize_balanced_distribution("data/meta_train.csv", strategy="oversample")
# visualize_balanced_distribution("data/meta_train.csv", strategy="weights")
# visualize_balanced_distribution("data/meta_train.csv", strategy="undersample")

