import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from save_figures import save_fig

def generate_visuals(df):
    visualizations = []
    saved_files = []

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [col for col in df.select_dtypes('object')
                        if 1 < df[col].nunique() < 30]
    
    try:
        # 1. Correlation Heatmap
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = df[numeric_cols].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
            ax.set_title("Correlation Heatmap")
            path = save_fig(fig)
            visualizations.append(("Correlation Heatmap", path))
            saved_files.append(path)
    except Exception as e:
        print(f"Heatmap error: {e}")

    try:
        # 2. Pairplot (limit to first 5 numeric columns for performance)
        if 2 <= len(numeric_cols) <= 5:
            fig = sns.pairplot(df[numeric_cols].dropna(), diag_kind="kde", corner=True)
            fig.fig.suptitle("Pairplot of Numeric Variables", y=1.02)
            path = save_fig(fig.fig)
            visualizations.append(("Pairplot", path))
            saved_files.append(path)
        elif len(numeric_cols) > 5:
            fig = sns.pairplot(df[numeric_cols[:5]].dropna(), diag_kind="kde", corner=True)
            fig.fig.suptitle("Pairplot (First 5 Numeric Variables)", y=1.02)
            path = save_fig(fig.fig)
            visualizations.append(("Pairplot", path))
            saved_files.append(path)
    except Exception as e:
        print(f"Pairplot error: {e}")

    try:
        # 3. Histograms for Numeric Columns
        if numeric_cols:
            n_cols = min(len(numeric_cols), 12)
            cols_to_plot = numeric_cols[:n_cols]
            n_rows = (len(cols_to_plot) + 2) // 3
            fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 and len(cols_to_plot) == 1 else axes.flatten()
            
            for i, col in enumerate(cols_to_plot):
                sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color='steelblue')
                axes[i].set_title(f"Distribution of {col}")
                axes[i].set_xlabel(col)
            
            # Hide unused subplots
            for j in range(len(cols_to_plot), len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            path = save_fig(fig)
            visualizations.append(("Histograms - Numeric Distributions", path))
            saved_files.append(path)
    except Exception as e:
        print(f"Histogram error: {e}")

    try:
        # 4. Boxplots for Numeric Columns (Outlier Detection)
        if numeric_cols:
            n_cols = min(len(numeric_cols), 12)
            cols_to_plot = numeric_cols[:n_cols]
            fig, ax = plt.subplots(figsize=(14, 6))
            df[cols_to_plot].boxplot(ax=ax, rot=45)
            ax.set_title("Boxplots - Outlier Detection")
            plt.tight_layout()
            path = save_fig(fig)
            visualizations.append(("Boxplots - Outlier Detection", path))
            saved_files.append(path)
    except Exception as e:
        print(f"Boxplot error: {e}")

    try:
        # 5. Bar Charts for Categorical Columns
        if categorical_cols:
            n_cols = min(len(categorical_cols), 6)
            cols_to_plot = categorical_cols[:n_cols]
            n_rows = (len(cols_to_plot) + 1) // 2
            fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 and len(cols_to_plot) == 1 else axes.flatten()
            
            for i, col in enumerate(cols_to_plot):
                value_counts = df[col].value_counts().head(10)
                sns.barplot(x=value_counts.values, y=value_counts.index, ax=axes[i], palette="viridis")
                axes[i].set_title(f"Top Categories in {col}")
                axes[i].set_xlabel("Count")
            
            # Hide unused subplots
            for j in range(len(cols_to_plot), len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            path = save_fig(fig)
            visualizations.append(("Bar Charts - Categorical Distributions", path))
            saved_files.append(path)
    except Exception as e:
        print(f"Bar chart error: {e}")

    try:
        # 6. Violin Plots (Numeric by Categorical)
        if numeric_cols and categorical_cols:
            num_col = numeric_cols[0]
            cat_col = categorical_cols[0]
            if df[cat_col].nunique() <= 10:
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.violinplot(x=cat_col, y=num_col, data=df, ax=ax, palette="muted")
                ax.set_title(f"Violin Plot: {num_col} by {cat_col}")
                plt.xticks(rotation=45)
                plt.tight_layout()
                path = save_fig(fig)
                visualizations.append(("Violin Plot", path))
                saved_files.append(path)
    except Exception as e:
        print(f"Violin plot error: {e}")

    try:
        # 7. Count Plot for Categorical Variables
        if categorical_cols:
            cat_col = categorical_cols[0]
            if df[cat_col].nunique() <= 15:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.countplot(y=cat_col, data=df, ax=ax, palette="Set2", order=df[cat_col].value_counts().index[:15])
                ax.set_title(f"Count Plot: {cat_col}")
                plt.tight_layout()
                path = save_fig(fig)
                visualizations.append(("Count Plot", path))
                saved_files.append(path)
    except Exception as e:
        print(f"Count plot error: {e}")

    try:
        # 8. Scatter Plot (First Two Numeric Columns)
        if len(numeric_cols) >= 2:
            fig, ax = plt.subplots(figsize=(10, 6))
            x_col, y_col = numeric_cols[0], numeric_cols[1]
            
            if categorical_cols and df[categorical_cols[0]].nunique() <= 10:
                sns.scatterplot(x=x_col, y=y_col, hue=categorical_cols[0], data=df, ax=ax, palette="deep", alpha=0.7)
            else:
                sns.scatterplot(x=x_col, y=y_col, data=df, ax=ax, alpha=0.7, color='steelblue')
            
            ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
            plt.tight_layout()
            path = save_fig(fig)
            visualizations.append(("Scatter Plot", path))
            saved_files.append(path)
    except Exception as e:
        print(f"Scatter plot error: {e}")

    try:
        # 9. Missing Values Heatmap
        missing = df.isnull().sum()
        if missing.sum() > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap="viridis", ax=ax)
            ax.set_title("Missing Values Heatmap")
            plt.tight_layout()
            path = save_fig(fig)
            visualizations.append(("Missing Values Heatmap", path))
            saved_files.append(path)
    except Exception as e:
        print(f"Missing values heatmap error: {e}")

    try:
        # 10. KDE Plot (Density Estimation)
        if len(numeric_cols) >= 2:
            fig, ax = plt.subplots(figsize=(10, 6))
            for col in numeric_cols[:4]:
                sns.kdeplot(df[col].dropna(), ax=ax, label=col, fill=True, alpha=0.3)
            ax.set_title("KDE Plot - Density Comparison")
            ax.legend()
            plt.tight_layout()
            path = save_fig(fig)
            visualizations.append(("KDE Plot - Density Comparison", path))
            saved_files.append(path)
    except Exception as e:
        print(f"KDE plot error: {e}")

    try:
        # 11. Stacked Bar Chart (if multiple categorical columns)
        if len(categorical_cols) >= 2:
            cat1, cat2 = categorical_cols[0], categorical_cols[1]
            if df[cat1].nunique() <= 10 and df[cat2].nunique() <= 10:
                cross_tab = pd.crosstab(df[cat1], df[cat2])
                fig, ax = plt.subplots(figsize=(12, 6))
                cross_tab.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
                ax.set_title(f"Stacked Bar: {cat1} vs {cat2}")
                ax.legend(title=cat2, bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.xticks(rotation=45)
                plt.tight_layout()
                path = save_fig(fig)
                visualizations.append(("Stacked Bar Chart", path))
                saved_files.append(path)
    except Exception as e:
        print(f"Stacked bar chart error: {e}")

    try:
        # 12. Pie Chart for Top Categorical Column
        if categorical_cols:
            cat_col = categorical_cols[0]
            if 2 <= df[cat_col].nunique() <= 8:
                fig, ax = plt.subplots(figsize=(8, 8))
                value_counts = df[cat_col].value_counts()
                ax.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%', 
                       colors=sns.color_palette("pastel"), startangle=90)
                ax.set_title(f"Pie Chart: {cat_col}")
                plt.tight_layout()
                path = save_fig(fig)
                visualizations.append(("Pie Chart", path))
                saved_files.append(path)
    except Exception as e:
        print(f"Pie chart error: {e}")

    try:
        # 13. Swarm Plot (if small dataset)
        if len(df) <= 500 and numeric_cols and categorical_cols:
            num_col = numeric_cols[0]
            cat_col = categorical_cols[0]
            if df[cat_col].nunique() <= 8:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.swarmplot(x=cat_col, y=num_col, data=df, ax=ax, palette="Set1")
                ax.set_title(f"Swarm Plot: {num_col} by {cat_col}")
                plt.xticks(rotation=45)
                plt.tight_layout()
                path = save_fig(fig)
                visualizations.append(("Swarm Plot", path))
                saved_files.append(path)
    except Exception as e:
        print(f"Swarm plot error: {e}")

    try:
        # 14. Regression Plot
        if len(numeric_cols) >= 2:
            x_col, y_col = numeric_cols[0], numeric_cols[1]
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.regplot(x=x_col, y=y_col, data=df, ax=ax, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
            ax.set_title(f"Regression Plot: {x_col} vs {y_col}")
            plt.tight_layout()
            path = save_fig(fig)
            visualizations.append(("Regression Plot", path))
            saved_files.append(path)
    except Exception as e:
        print(f"Regression plot error: {e}")

    try:
        # 15. Heatmap of Value Counts (Categorical)
        if len(categorical_cols) >= 2:
            cat1, cat2 = categorical_cols[0], categorical_cols[1]
            if df[cat1].nunique() <= 10 and df[cat2].nunique() <= 10:
                cross_tab = pd.crosstab(df[cat1], df[cat2])
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(cross_tab, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
                ax.set_title(f"Cross-tabulation Heatmap: {cat1} vs {cat2}")
                plt.tight_layout()
                path = save_fig(fig)
                visualizations.append(("Cross-tabulation Heatmap", path))
                saved_files.append(path)
    except Exception as e:
        print(f"Cross-tabulation heatmap error: {e}")

    return visualizations, saved_files

import os
def cleanup_files(file_paths): 
    """Clean up temporary image files."""
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"Error deleting file {path}: {e}")
