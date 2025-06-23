import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional
from typing import List
import math

COLORS = {0: 'blue', 1: 'red'}

def filter_outliers(df: pd.DataFrame, 
                   outlier_feature: Optional[str] = None, 
                   percentile: float = 0.99) -> pd.DataFrame:
    """
    Filter outliers from a DataFrame based on a specified feature and percentile threshold.
    
    Args:
        df: Input DataFrame
        outlier_feature: Column name to filter outliers on (if None, returns original DataFrame)
        percentile: Percentile threshold for outlier removal (0.0 to 1.0)
    
    Returns:
        DataFrame with outliers removed
    
    Works with:
        - Numeric dtypes (int64, float64) for outlier_feature
    """
    if outlier_feature:
        threshold = df[outlier_feature].quantile(percentile)
        return df[df[outlier_feature] <= threshold]
    return df

def plot_categorical_dist(df: pd.DataFrame, 
                         feature: str, 
                         target: str, 
                         remove_outliers: bool = False, 
                         outlier_feature: Optional[str] = None, 
                         percentile: float = 0.99) -> None:
    """
    Plot count and proportion distributions of a categorical feature by target variable.
    
    Args:
        df: Input DataFrame
        feature: Categorical feature to analyze
        target: Binary target variable (0/1)
        remove_outliers: Whether to filter outliers
        outlier_feature: Feature to use for outlier removal (if applicable)
        percentile: Percentile threshold for outlier removal
    
    Works with:
        - Categorical features (will be converted to category dtype)
        - Binary target (int64, bool)
    """
    df = filter_outliers(df, outlier_feature if remove_outliers else None, percentile)
    df[feature] = df[feature].astype('category')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    count_data = pd.crosstab(df[feature], df[target])
    count_data.plot.bar(stacked=True, ax=ax1, color=[COLORS[0], COLORS[1]])
    ax1.set(title=f'{target} Count by {feature}', xlabel=feature, ylabel='Count')
    ax1.tick_params(axis='x', rotation=45)
    
    prop_data = pd.crosstab(df[feature], df[target], normalize='index')
    prop_data.plot.bar(stacked=True, ax=ax2, color=[COLORS[0], COLORS[1]])
    ax2.set(title=f'{target} Proportion by {feature}', xlabel=feature, ylabel='Proportion')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def plot_numeric_dist(df: pd.DataFrame, 
                     feature: str, 
                     target: str, 
                     remove_outliers: bool = False, 
                     outlier_feature: Optional[str] = None, 
                     percentile: float = 0.99) -> None:
    """
    Plot histogram and boxplot of a numerical feature by target variable.
    
    Args:
        df: Input DataFrame
        feature: Numerical feature to analyze
        target: Binary target variable (0/1)
        remove_outliers: Whether to filter outliers
        outlier_feature: Feature to use for outlier removal (if applicable)
        percentile: Percentile threshold for outlier removal
    
    Works with:
        - Numeric features (int64, float64)
        - Binary target (int64, bool)
    """
    df = filter_outliers(df, outlier_feature if remove_outliers else None, percentile)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [2, 1]})
    
    for value in [0, 1]:
        sns.histplot(df[df[target] == value][feature], kde=True, label=f'{target} = {value}', 
                    color=COLORS[value], alpha=0.5, ax=ax1)
    ax1.set_xlabel(feature)
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'{feature} Distribution by {target}')
    ax1.legend()
    
    sns.boxplot(x=target, y=feature, hue=target, data=df, ax=ax2, 
                palette=COLORS, legend=False)
    ax2.set_title(f'{feature} Boxplot by {target}')
    ax2.set_xlabel(target)
    ax2.set_ylabel(feature)
    
    plt.tight_layout()
    plt.show()

def plot_multiple_categorical_dists(df: pd.DataFrame,
                                    features: List[str],
                                    target: str) -> None:
    """
    Plot count and proportion distributions of multiple categorical features by target variable.

    Args:
        df: Input DataFrame
        features: List of categorical features to analyze
        target: Binary target variable (0/1)

    Notes:
        - Arranges 2 subplots per feature (count and proportion) with 4 subplots per row.
        - Assumes COLORS dictionary is defined globally with at least two color values.
    """
    num_features = len(features)
    plots_per_row = 4
    total_subplots = num_features * 2
    rows = math.ceil(total_subplots / plots_per_row)

    fig, axes = plt.subplots(rows, plots_per_row, figsize=(plots_per_row * 5, rows * 4))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        ax_count = axes[i * 2]
        count_data = pd.crosstab(df[feature].astype('category'), df[target])
        count_data.plot.bar(stacked=True, ax=ax_count, color=COLORS)
        ax_count.set_title(f'{target} Count by {feature}')
        ax_count.set_xlabel(feature)
        ax_count.set_ylabel('Count')
        ax_count.tick_params(axis='x', rotation=45)

        ax_prop = axes[i * 2 + 1]
        prop_data = pd.crosstab(df[feature].astype('category'), df[target], normalize='index')
        prop_data.plot.bar(stacked=True, ax=ax_prop, color=COLORS)
        ax_prop.set_title(f'{target} Proportion by {feature}')
        ax_prop.set_xlabel(feature)
        ax_prop.set_ylabel('Proportion')
        ax_prop.tick_params(axis='x', rotation=45)

    for j in range(total_subplots, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def plot_multiple_numeric_dists(df: pd.DataFrame,
                                features: List[str],
                                target: str,
                                remove_outliers: bool = False,
                                outlier_features: Optional[List[str]] = None,
                                percentile: float = 0.99) -> None:
    """
    Plot histograms and boxplots of multiple numerical features by target variable.
    
    Args:
        df: Input DataFrame
        features: List of numerical features to analyze
        target: Binary target variable (0/1)
        remove_outliers: Whether to filter outliers
        outlier_features: Features to use for outlier removal
        percentile: Percentile threshold for outlier removal
    """
    num_features = len(features)
    plots_per_row = 4
    rows = (num_features + plots_per_row - 1) // plots_per_row

    fig, axes = plt.subplots(rows * 2, plots_per_row, figsize=(plots_per_row * 5, rows * 8))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        outlier_feature = None
        if remove_outliers and outlier_features and i < len(outlier_features):
            outlier_feature = outlier_features[i]

        df_filtered = filter_outliers(df, outlier_feature, percentile)

        ax_hist = axes[i * 2]
        for value in [0, 1]:
            sns.histplot(df_filtered[df_filtered[target] == value][feature], kde=True,
                         label=f'{target} = {value}', color=COLORS[value], alpha=0.5, ax=ax_hist)
        ax_hist.set_title(f'{feature} Distribution')
        ax_hist.set_xlabel(feature)
        ax_hist.set_ylabel('Frequency')
        ax_hist.legend(loc='upper right')

        ax_box = axes[i * 2 + 1]
        sns.boxplot(x=target, y=feature, data=df_filtered, ax=ax_box,
            hue=target, palette=COLORS, legend=False)
        ax_box.set_title(f'{feature} Boxplot')
        ax_box.set_xlabel(target)
        ax_box.set_ylabel(feature)

    for j in range(num_features * 2, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df: pd.DataFrame, 
                           title: str, 
                           columns: list) -> None:
    """
    Plot a correlation heatmap for specified numeric columns in the DataFrame.
    
    Args:
        df: Input DataFrame
        title: Title for the heatmap
        columns: List of column names to include in correlation analysis
    
    Works with:
        - Numeric dtypes (int64, float64)
    """
    numeric_df = df[columns].select_dtypes(include=['float64', 'int64'])
    corr_matrix = numeric_df.corr()
    
    plt.figure(figsize=(15, 15))
    sns.heatmap(corr_matrix, vmin=0, vmax=1, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def calculate_woe_iv(df: pd.DataFrame, 
                    feature: str, 
                    target: str, 
                    bins: int = 10) -> float:
    """
    Calculate Weight of Evidence (WOE) and Information Value (IV) for a feature.
    
    Args:
        df: Input DataFrame
        feature: Feature to analyze
        target: Binary target variable (0/1)
        bins: Number of bins for numeric features
    
    Returns:
        Information Value (IV) as a float
    
    Works with:
        - Any feature type (numeric will be binned, categorical used as-is)
        - Binary target (int64, bool)
    """
    if df[feature].dtype in ['int64', 'float64']:
        df[feature] = pd.qcut(df[feature], bins, duplicates='drop')
    
    grouped = df.groupby(feature)[target].agg(['count', 'sum'])
    grouped.columns = ['Total', 'Bads']
    grouped['Goods'] = grouped['Total'] - grouped['Bads']
    
    grouped['% Goods'] = grouped['Goods'] / grouped['Goods'].sum()
    grouped['% Bads'] = grouped['Bads'] / grouped['Bads'].sum()
    
    grouped['WOE'] = np.log((grouped['% Goods'] + 1e-10) / (grouped['% Bads'] + 1e-10))
    grouped['IV'] = (grouped['% Goods'] - grouped['% Bads']) * grouped['WOE']
    
    return grouped['IV'].sum()

def compute_iv_for_dataset(df: pd.DataFrame, 
                         target_column: str) -> pd.DataFrame:
    """
    Compute Information Value (IV) for all features in the dataset.
    
    Args:
        df: Input DataFrame
        target_column: Name of the target variable
    
    Returns:
        DataFrame with features and their IV values, sorted descending
    
    Works with:
        - Any feature types (numeric will be binned)
        - Binary target (int64, bool)
    """
    iv_dict = {}
    for feature in df.columns:
        if feature != target_column:
            try:
                iv = calculate_woe_iv(df, feature, target_column)
                iv_dict[feature] = iv
            except Exception as e:
                print(f"Skipping {feature}: {e}")
    
    iv_df = pd.DataFrame(list(iv_dict.items()), columns=['Feature', 'IV'])
    return iv_df.sort_values(by='IV', ascending=False)
    
def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary of each column in the DataFrame.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with summary statistics for each column
    """
    summary_data = [
        {
            'col_name': col,
            'col_dtype': df[col].dtype,
            'num_of_nulls': df[col].isna().sum(),
            'num_of_non_nulls': df[col].notna().sum(),
            'num_of_distinct_values': df[col].nunique(),
            'distinct_values_counts': df[col].value_counts().to_dict() if df[col].nunique() <= 10
                                     else df[col].value_counts().nlargest(10).to_dict()
        }
        for col in df.columns
    ]
    return pd.DataFrame(summary_data)