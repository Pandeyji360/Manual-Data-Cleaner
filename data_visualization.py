import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

def plot_missing_data(df):
    """
    Create a visualization of missing data in the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure with missing data visualization
    """
    # Calculate missing values by column
    missing = df.isnull().sum().reset_index()
    missing.columns = ['column', 'count']
    missing['percentage'] = round(missing['count'] / len(df) * 100, 2)
    
    # Sort by count of missing values
    missing = missing.sort_values('count', ascending=False)
    
    # Only include columns with missing values
    missing = missing[missing['count'] > 0]
    
    if missing.empty:
        # Create a simple figure saying no missing data
        fig = go.Figure()
        fig.add_annotation(
            text="No missing data found in the dataset",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Create the plot using Plotly
    fig = go.Figure()
    
    # Add bar for count
    fig.add_trace(go.Bar(
        y=missing['column'],
        x=missing['count'],
        orientation='h',
        name='Missing Count',
        marker=dict(color='rgba(58, 71, 80, 0.6)'),
        text=missing['count'],
        textposition='auto'
    ))
    
    # Add bar for percentage
    fig.add_trace(go.Bar(
        y=missing['column'],
        x=missing['percentage'],
        orientation='h',
        name='Missing Percentage',
        marker=dict(color='rgba(246, 78, 139, 0.6)'),
        text=missing['percentage'].apply(lambda x: f"{x}%"),
        textposition='auto'
    ))
    
    # Update layout
    fig.update_layout(
        title='Missing Values by Column',
        xaxis_title='Count / Percentage',
        yaxis=dict(
            title='Column',
            categoryorder='total ascending'
        ),
        barmode='group',
        height=max(400, len(missing) * 30),
        margin=dict(l=100, r=20, t=50, b=50)
    )
    
    return fig

def plot_data_distribution(df, column):
    """
    Create a distribution plot for a numerical column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to visualize
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure with distribution visualization
    """
    # Check if column is numeric
    if not pd.api.types.is_numeric_dtype(df[column]):
        # Return a message figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Column '{column}' is not numeric. Cannot create distribution plot.",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=15)
        )
        return fig
    
    # Create a subplot with histogram and box plot
    fig = make_subplots(rows=2, cols=1, 
                        row_heights=[0.7, 0.3],
                        subplot_titles=["Histogram", "Box Plot"])
    
    # Add histogram
    fig.add_trace(
        go.Histogram(
            x=df[column],
            name="Distribution",
            nbinsx=30,
            marker_color='rgba(58, 71, 80, 0.6)'
        ),
        row=1, col=1
    )
    
    # Add box plot
    fig.add_trace(
        go.Box(
            x=df[column],
            name="Box Plot",
            marker_color='rgba(246, 78, 139, 0.6)'
        ),
        row=2, col=1
    )
    
    # Calculate basic statistics
    stats = df[column].describe()
    
    # Add annotations for statistics
    annotations = [
        dict(
            x=0.01, y=0.95,
            xref="paper", yref="paper",
            text=f"Mean: {stats['mean']:.2f}",
            showarrow=False,
            font=dict(size=12),
            align="left",
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="black",
            borderwidth=1
        ),
        dict(
            x=0.01, y=0.90,
            xref="paper", yref="paper",
            text=f"Std Dev: {stats['std']:.2f}",
            showarrow=False,
            font=dict(size=12),
            align="left",
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="black",
            borderwidth=1
        ),
        dict(
            x=0.01, y=0.85,
            xref="paper", yref="paper",
            text=f"Min: {stats['min']:.2f}",
            showarrow=False,
            font=dict(size=12),
            align="left",
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="black",
            borderwidth=1
        ),
        dict(
            x=0.01, y=0.80,
            xref="paper", yref="paper",
            text=f"Max: {stats['max']:.2f}",
            showarrow=False,
            font=dict(size=12),
            align="left",
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="black",
            borderwidth=1
        )
    ]
    
    # Update layout
    fig.update_layout(
        title=f"Distribution of {column}",
        height=500,
        annotations=annotations
    )
    
    return fig

def plot_categorical_data(df, column):
    """
    Create visualizations for categorical data.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to visualize
        
    Returns:
        tuple: (bar_chart, pie_chart) Plotly figures
    """
    # Get value counts
    value_counts = df[column].value_counts().reset_index()
    value_counts.columns = ['value', 'count']
    value_counts['percentage'] = round(value_counts['count'] / len(df) * 100, 2)
    
    # If too many categories, limit to top 15 plus "Others"
    if len(value_counts) > 15:
        top_15 = value_counts.iloc[:15]
        others_count = value_counts.iloc[15:]['count'].sum()
        others_pct = value_counts.iloc[15:]['percentage'].sum()
        
        others_row = pd.DataFrame({'value': ['Others'], 
                                 'count': [others_count],
                                 'percentage': [others_pct]})
        
        value_counts = pd.concat([top_15, others_row], ignore_index=True)
    
    # Create bar chart
    bar_fig = px.bar(
        value_counts, 
        x='value', 
        y='count',
        text='percentage',
        labels={'value': column, 'count': 'Count'},
        title=f"Bar Chart of {column}",
        color='count',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    # Update text format
    bar_fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    
    # Rotate x-axis labels if there are many categories
    if len(value_counts) > 5:
        bar_fig.update_layout(xaxis_tickangle=-45)
    
    # Create pie chart
    pie_fig = px.pie(
        value_counts, 
        names='value', 
        values='count',
        title=f"Pie Chart of {column}",
        hole=0.3,
        labels={'value': column, 'count': 'Count'},
        hover_data=['percentage'],
        custom_data=['percentage']
    )
    
    # Update hover template
    pie_fig.update_traces(
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{customdata[0]:.2f}%'
    )
    
    return bar_fig, pie_fig

def plot_correlation_matrix(df):
    """
    Create a correlation matrix visualization.
    
    Args:
        df (pd.DataFrame): Input DataFrame with numeric columns
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure with correlation matrix
    """
    # Calculate correlation matrix
    corr_matrix = df.corr(numeric_only=True)
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale=px.colors.diverging.RdBu_r,
        title="Correlation Matrix"
    )
    
    # Update layout
    fig.update_layout(
        height=max(400, 25 * len(corr_matrix)),
        width=max(400, 25 * len(corr_matrix))
    )
    
    return fig

def plot_outliers(df, column):
    """
    Create a visualization highlighting outliers in a column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to visualize
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure with outlier visualization
    """
    # Check if column is numeric
    if not pd.api.types.is_numeric_dtype(df[column]):
        # Return a message figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Column '{column}' is not numeric. Cannot create outlier plot.",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=15)
        )
        return fig
    
    # Calculate IQR
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
    
    # Create figure
    fig = go.Figure()
    
    # Add box plot
    fig.add_trace(go.Box(
        y=df[column],
        name=column,
        boxpoints='outliers',
        marker=dict(
            color='rgba(58, 71, 80, 0.6)',
            outliercolor='rgba(219, 64, 82, 0.6)',
            line=dict(
                outliercolor='rgba(219, 64, 82, 0.6)',
                outlierwidth=2
            )
        ),
        line_color='rgba(58, 71, 80, 0.6)'
    ))
    
    # Add scatter plot for outliers
    if not outliers.empty:
        fig.add_trace(go.Scatter(
            y=outliers,
            x=np.zeros(len(outliers)),
            mode='markers',
            name='Outliers',
            marker=dict(
                color='rgba(219, 64, 82, 0.6)',
                size=8,
                line=dict(width=1)
            ),
            hoverinfo='y'
        ))
    
    # Add horizontal lines for bounds
    fig.add_shape(
        type='line',
        y0=lower_bound,
        y1=lower_bound,
        x0=-0.5,
        x1=0.5,
        line=dict(
            color='rgba(255, 165, 0, 0.6)',
            width=2,
            dash='dash'
        )
    )
    
    fig.add_shape(
        type='line',
        y0=upper_bound,
        y1=upper_bound,
        x0=-0.5,
        x1=0.5,
        line=dict(
            color='rgba(255, 165, 0, 0.6)',
            width=2,
            dash='dash'
        )
    )
    
    # Add annotations
    fig.add_annotation(
        x=0.5, y=lower_bound,
        xref='paper',
        text=f"Lower bound: {lower_bound:.2f}",
        showarrow=False,
        xanchor='left',
        yanchor='bottom',
        bgcolor='rgba(255, 255, 255, 0.8)'
    )
    
    fig.add_annotation(
        x=0.5, y=upper_bound,
        xref='paper',
        text=f"Upper bound: {upper_bound:.2f}",
        showarrow=False,
        xanchor='left',
        yanchor='top',
        bgcolor='rgba(255, 255, 255, 0.8)'
    )
    
    # Update layout
    fig.update_layout(
        title=f"Outlier Analysis for {column}",
        yaxis_title=column,
        showlegend=False,
        height=500
    )
    
    # Remove x-axis ticks and title
    fig.update_xaxes(
        showticklabels=False,
        title=None
    )
    
    return fig

def plot_cleaned_vs_original(original_df, cleaned_df):
    """
    Create a comparison visualization between original and cleaned data.
    
    Args:
        original_df (pd.DataFrame): Original DataFrame
        cleaned_df (pd.DataFrame): Cleaned DataFrame
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure with comparison visualization
    """
    # Create a figure
    fig = go.Figure()
    
    # Add traces for comparison metrics
    metrics = [
        {
            'name': 'Row Count',
            'original': len(original_df),
            'cleaned': len(cleaned_df)
        },
        {
            'name': 'Missing Values',
            'original': original_df.isnull().sum().sum(),
            'cleaned': cleaned_df.isnull().sum().sum()
        },
        {
            'name': 'Duplicate Rows',
            'original': original_df.duplicated().sum(),
            'cleaned': cleaned_df.duplicated().sum()
        }
    ]
    
    # Get numeric columns in both dataframes
    numeric_cols = [col for col in original_df.columns if pd.api.types.is_numeric_dtype(original_df[col]) 
                   and col in cleaned_df.columns 
                   and pd.api.types.is_numeric_dtype(cleaned_df[col])]
    
    # Add metrics for numeric columns if available
    if numeric_cols:
        # Mean
        metrics.append({
            'name': 'Mean (numeric cols)',
            'original': original_df[numeric_cols].mean().mean(),
            'cleaned': cleaned_df[numeric_cols].mean().mean()
        })
        
        # Standard Deviation
        metrics.append({
            'name': 'Std Dev (numeric cols)',
            'original': original_df[numeric_cols].std().mean(),
            'cleaned': cleaned_df[numeric_cols].std().mean()
        })
    
    # Add bars for each metric
    categories = [metric['name'] for metric in metrics]
    original_values = [metric['original'] for metric in metrics]
    cleaned_values = [metric['cleaned'] for metric in metrics]
    
    fig.add_trace(go.Bar(
        x=categories,
        y=original_values,
        name='Original Data',
        marker_color='rgba(58, 71, 80, 0.6)'
    ))
    
    fig.add_trace(go.Bar(
        x=categories,
        y=cleaned_values,
        name='Cleaned Data',
        marker_color='rgba(246, 78, 139, 0.6)'
    ))
    
    # Calculate percentage changes
    percent_changes = []
    for original, cleaned in zip(original_values, cleaned_values):
        if original == 0:
            percent_changes.append("N/A")
        else:
            change = ((cleaned - original) / original) * 100
            percent_changes.append(f"{change:.2f}%")
    
    # Add percent change annotations
    for i, (category, pct_change) in enumerate(zip(categories, percent_changes)):
        # Skip if percent change is N/A
        if pct_change == "N/A":
            continue
        
        # Get the maximum of original and cleaned for this category
        max_val = max(original_values[i], cleaned_values[i])
        
        fig.add_annotation(
            x=category,
            y=max_val,
            text=pct_change,
            showarrow=False,
            yshift=10,
            font=dict(size=10)
        )
    
    # Update layout
    fig.update_layout(
        title='Comparison: Original vs Cleaned Data',
        barmode='group',
        height=500
    )
    
    return fig
