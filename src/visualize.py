"""Plotting functions."""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def create_plots(df, feat_col, tar_col):
    """Create plots.

    Args:
        df (pd.DataFrame): data
        feat_col (String): Feature column
        tar_col (String): Target column
    """
    plt.figure(figsize=(10, 5))

    plt.subplot(1,2,1)
    sns.violinplot(x=tar_col, y=feat_col, data=df)

    plt.subplot(1,2,2)
    sns.distplot(df[df[tar_col] == 1.0][feat_col], 
                 label="Duplicate",
                 color='red')
    sns.distplot(df[df[tar_col] == 0.0][feat_col],
                 label="Not duplicate",
                 color='blue')
    plt.legend()
    plt.savefig(f'../figures/{feat_col}_vs_{tar_col}.png')
    
def plot_confusion_matrix(y_true, y_pred, title):
    """Plot confusion matrix

    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        title (string): Title
    """
    fig = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred),
                       display_labels=['Not duplicate', 'Duplicate'])
    fig.plot()
    plt.savefig(f"../figures/{title}.png")
