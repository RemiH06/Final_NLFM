"""
Functions for data visualization and exploratory data analysis using Plotly.
Includes time series plots, correlation analysis, and forecast visualization.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt


def PlotTimeSeries(
    dataFrame: pd.DataFrame,
    dateColumn: str,
    valueColumns: List[str],
    title: str = "Time Series Plot",
    ylabel: str = "Value",
    height: int = 600,
    savePath: Optional[str] = None
) -> None:
    """
    Plot one or more time series using Plotly.
    
    Parameters:
    -----------
    dataFrame : pd.DataFrame
        DataFrame with time series data
    dateColumn : str
        Name of date column
    valueColumns : List[str]
        List of columns to plot
    title : str
        Plot title
    ylabel : str
        Y-axis label
    height : int
        Figure height in pixels
    savePath : Optional[str]
        Path to save figure (HTML or PNG)
    """
    fig = go.Figure()
    
    for column in valueColumns:
        fig.add_trace(go.Scatter(
            x=dataFrame[dateColumn],
            y=dataFrame[column],
            name=column,
            mode='lines',
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#1f2937')),
        xaxis_title='Fecha',
        yaxis_title=ylabel,
        hovermode='x unified',
        height=height,
        template='plotly_white',
        font=dict(family='Arial', size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    if savePath:
        if savePath.endswith('.html'):
            fig.write_html(savePath)
        elif savePath.endswith('.png'):
            fig.write_image(savePath, width=1400, height=height)
        print(f"Figure saved to {savePath}")
    
    fig.show()


def PlotMultipleSubplots(
    dataFrame: pd.DataFrame,
    dateColumn: str,
    valueColumns: List[str],
    title: str = "Multiple Time Series",
    height: int = 800,
    savePath: Optional[str] = None
) -> None:
    """
    Plot multiple time series in separate subplots using Plotly.
    
    Parameters:
    -----------
    dataFrame : pd.DataFrame
        DataFrame with time series data
    dateColumn : str
        Name of date column
    valueColumns : List[str]
        List of columns to plot
    title : str
        Main title
    height : int
        Figure height in pixels
    savePath : Optional[str]
        Path to save figure
    """
    nPlots = len(valueColumns)
    
    fig = make_subplots(
        rows=nPlots,
        cols=1,
        subplot_titles=valueColumns,
        vertical_spacing=0.08
    )
    
    colors = px.colors.qualitative.Plotly
    
    for i, column in enumerate(valueColumns):
        fig.add_trace(
            go.Scatter(
                x=dataFrame[dateColumn],
                y=dataFrame[column],
                name=column,
                mode='lines',
                line=dict(color=colors[i % len(colors)], width=2),
                showlegend=False
            ),
            row=i+1,
            col=1
        )
    
    fig.update_xaxes(title_text="Fecha", row=nPlots, col=1)
    
    for i in range(nPlots):
        fig.update_yaxes(title_text="Valor", row=i+1, col=1)
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#1f2937')),
        height=height,
        template='plotly_white',
        font=dict(family='Arial', size=12),
        hovermode='x unified'
    )
    
    if savePath:
        if savePath.endswith('.html'):
            fig.write_html(savePath)
        elif savePath.endswith('.png'):
            fig.write_image(savePath, width=1400, height=height)
        print(f"Figure saved to {savePath}")
    
    fig.show()


def PlotCorrelationMatrix(
    dataFrame: pd.DataFrame,
    columns: Optional[List[str]] = None,
    title: str = "Correlation Matrix",
    height: int = 700,
    savePath: Optional[str] = None
) -> None:
    """
    Plot correlation matrix heatmap using Plotly.
    
    Parameters:
    -----------
    dataFrame : pd.DataFrame
        DataFrame with data
    columns : Optional[List[str]]
        Specific columns to include
    title : str
        Plot title
    height : int
        Figure height in pixels
    savePath : Optional[str]
        Path to save figure
    """
    if columns:
        data = dataFrame[columns]
    else:
        data = dataFrame.select_dtypes(include=[np.number])
    
    correlation = data.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation.values,
        x=correlation.columns,
        y=correlation.columns,
        colorscale='RdBu',
        zmid=0,
        text=correlation.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlación")
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#1f2937')),
        height=height,
        width=height,
        template='plotly_white',
        font=dict(family='Arial', size=12)
    )
    
    if savePath:
        if savePath.endswith('.html'):
            fig.write_html(savePath)
        elif savePath.endswith('.png'):
            fig.write_image(savePath, width=height, height=height)
        print(f"Figure saved to {savePath}")
    
    fig.show()


def PlotACFPACF(
    series: pd.Series,
    lags: int = 40,
    title: str = "ACF and PACF",
    figsize: Tuple[int, int] = (14, 6),
    savePath: Optional[str] = None
) -> None:
    """
    Plot Autocorrelation and Partial Autocorrelation functions.
    Note: Uses matplotlib as statsmodels doesn't have Plotly integration.
    
    Parameters:
    -----------
    series : pd.Series
        Time series data
    lags : int
        Number of lags to display
    title : str
        Main title
    figsize : Tuple[int, int]
        Figure size
    savePath : Optional[str]
        Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    plot_acf(series.dropna(), lags=lags, ax=axes[0])
    axes[0].set_title('Autocorrelation Function (ACF)', fontsize=12)
    
    plot_pacf(series.dropna(), lags=lags, ax=axes[1])
    axes[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=12)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if savePath:
        plt.savefig(savePath, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {savePath}")
    
    plt.show()


def PlotSeasonalDecomposition(
    series: pd.Series,
    period: int,
    model: str = 'additive',
    title: str = "Seasonal Decomposition",
    height: int = 1000,
    savePath: Optional[str] = None
) -> None:
    """
    Plot seasonal decomposition of time series using Plotly.
    
    Parameters:
    -----------
    series : pd.Series
        Time series data with datetime index
    period : int
        Period of seasonality
    model : str
        Type of decomposition: 'additive' or 'multiplicative'
    title : str
        Main title
    height : int
        Figure height in pixels
    savePath : Optional[str]
        Path to save figure
    """
    decomposition = seasonal_decompose(
        series.dropna(),
        model=model,
        period=period
    )
    
    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=('Original', 'Tendencia', 'Estacionalidad', 'Residuos'),
        vertical_spacing=0.06
    )
    
    # Original
    fig.add_trace(
        go.Scatter(x=decomposition.observed.index, y=decomposition.observed.values,
                  mode='lines', name='Original', line=dict(color='#3b82f6')),
        row=1, col=1
    )
    
    # Trend
    fig.add_trace(
        go.Scatter(x=decomposition.trend.index, y=decomposition.trend.values,
                  mode='lines', name='Tendencia', line=dict(color='#10b981')),
        row=2, col=1
    )
    
    # Seasonal
    fig.add_trace(
        go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal.values,
                  mode='lines', name='Estacionalidad', line=dict(color='#f59e0b')),
        row=3, col=1
    )
    
    # Residual
    fig.add_trace(
        go.Scatter(x=decomposition.resid.index, y=decomposition.resid.values,
                  mode='lines', name='Residuos', line=dict(color='#ef4444')),
        row=4, col=1
    )
    
    fig.update_xaxes(title_text="Fecha", row=4, col=1)
    fig.update_yaxes(title_text="Valor", row=1, col=1)
    fig.update_yaxes(title_text="Valor", row=2, col=1)
    fig.update_yaxes(title_text="Valor", row=3, col=1)
    fig.update_yaxes(title_text="Valor", row=4, col=1)
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#1f2937')),
        height=height,
        template='plotly_white',
        showlegend=False,
        font=dict(family='Arial', size=12)
    )
    
    if savePath:
        if savePath.endswith('.html'):
            fig.write_html(savePath)
        elif savePath.endswith('.png'):
            fig.write_image(savePath, width=1400, height=height)
        print(f"Figure saved to {savePath}")
    
    fig.show()


def PlotPredictionVsActual(
    yTrue: np.ndarray,
    yPred: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
    title: str = "Actual vs Predicted",
    ylabel: str = "Value",
    height: int = 600,
    savePath: Optional[str] = None
) -> None:
    """
    Plot actual vs predicted values using Plotly.
    
    Parameters:
    -----------
    yTrue : np.ndarray
        True values
    yPred : np.ndarray
        Predicted values
    dates : Optional[pd.DatetimeIndex]
        Date index for x-axis
    title : str
        Plot title
    ylabel : str
        Y-axis label
    height : int
        Figure height in pixels
    savePath : Optional[str]
        Path to save figure
    """
    fig = go.Figure()
    
    xAxis = dates if dates is not None else np.arange(len(yTrue))
    
    # Actual values
    fig.add_trace(go.Scatter(
        x=xAxis,
        y=yTrue,
        name='Real',
        mode='lines',
        line=dict(color='#3b82f6', width=2)
    ))
    
    # Predicted values
    fig.add_trace(go.Scatter(
        x=xAxis,
        y=yPred,
        name='Predicción',
        mode='lines',
        line=dict(color='#ef4444', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#1f2937')),
        xaxis_title='Fecha' if dates is not None else 'Tiempo',
        yaxis_title=ylabel,
        hovermode='x unified',
        height=height,
        template='plotly_white',
        font=dict(family='Arial', size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    if savePath:
        if savePath.endswith('.html'):
            fig.write_html(savePath)
        elif savePath.endswith('.png'):
            fig.write_image(savePath, width=1400, height=height)
        print(f"Figure saved to {savePath}")
    
    fig.show()


def PlotForecastWithConfidence(
    historical: pd.DataFrame,
    forecast: pd.DataFrame,
    dateColumn: str,
    valueColumn: str,
    confidenceLevel: float = 0.95,
    title: str = "Forecast with Confidence Intervals",
    ylabel: str = "Value",
    height: int = 600,
    savePath: Optional[str] = None
) -> None:
    """
    Plot historical data with future forecast and confidence intervals.
    
    Parameters:
    -----------
    historical : pd.DataFrame
        Historical data
    forecast : pd.DataFrame
        Forecasted data (should have 'lower' and 'upper' columns for CI)
    dateColumn : str
        Name of date column
    valueColumn : str
        Name of value column
    confidenceLevel : float
        Confidence level for intervals (e.g., 0.95 for 95%)
    title : str
        Plot title
    ylabel : str
        Y-axis label
    height : int
        Figure height in pixels
    savePath : Optional[str]
        Path to save figure
    """
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical[dateColumn],
        y=historical[valueColumn],
        name='Histórico',
        mode='lines',
        line=dict(color='#3b82f6', width=2)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast[dateColumn],
        y=forecast[valueColumn],
        name='Pronóstico',
        mode='lines+markers',
        line=dict(color='#ef4444', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    # Confidence intervals if available
    if 'lower' in forecast.columns and 'upper' in forecast.columns:
        # Upper bound
        fig.add_trace(go.Scatter(
            x=forecast[dateColumn],
            y=forecast['upper'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Lower bound with fill
        fig.add_trace(go.Scatter(
            x=forecast[dateColumn],
            y=forecast['lower'],
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(239, 68, 68, 0.2)',
            fill='tonexty',
            name=f'{int(confidenceLevel*100)}% Intervalo de Confianza',
            hoverinfo='skip'
        ))
    
    # Vertical line at forecast start
    try:
        forecastStartDate = pd.Timestamp(forecast[dateColumn].iloc[0])
        fig.add_vline(
            x=forecastStartDate,
            line_dash="dot",
            line_color="gray",
            annotation_text="Inicio Pronóstico",
            annotation_position="top"
        )
    except:
        # Si hay error, omitir la línea vertical
        pass
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#1f2937')),
        xaxis_title='Fecha',
        yaxis_title=ylabel,
        hovermode='x unified',
        height=height,
        template='plotly_white',
        font=dict(family='Arial', size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    if savePath:
        if savePath.endswith('.html'):
            fig.write_html(savePath)
        elif savePath.endswith('.png'):
            fig.write_image(savePath, width=1400, height=height)
        print(f"Figure saved to {savePath}")
    
    fig.show()


def PlotForecast(
    historical: pd.DataFrame,
    forecast: pd.DataFrame,
    dateColumn: str,
    valueColumn: str,
    title: str = "Forecast",
    ylabel: str = "Value",
    height: int = 600,
    savePath: Optional[str] = None
) -> None:
    """
    Plot historical data with future forecast using Plotly.
    
    Parameters:
    -----------
    historical : pd.DataFrame
        Historical data
    forecast : pd.DataFrame
        Forecasted data
    dateColumn : str
        Name of date column
    valueColumn : str
        Name of value column
    title : str
        Plot title
    ylabel : str
        Y-axis label
    height : int
        Figure height in pixels
    savePath : Optional[str]
        Path to save figure
    """
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical[dateColumn],
        y=historical[valueColumn],
        name='Histórico',
        mode='lines',
        line=dict(color='#3b82f6', width=2)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast[dateColumn],
        y=forecast[valueColumn],
        name='Pronóstico',
        mode='lines+markers',
        line=dict(color='#ef4444', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    # Vertical line at forecast start
    fig.add_vline(
        x=forecast[dateColumn].iloc[0],
        line_dash="dot",
        line_color="gray",
        annotation_text="Inicio Pronóstico",
        annotation_position="top"
    )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#1f2937')),
        xaxis_title='Fecha',
        yaxis_title=ylabel,
        hovermode='x unified',
        height=height,
        template='plotly_white',
        font=dict(family='Arial', size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    if savePath:
        if savePath.endswith('.html'):
            fig.write_html(savePath)
        elif savePath.endswith('.png'):
            fig.write_image(savePath, width=1400, height=height)
        print(f"Figure saved to {savePath}")
    
    fig.show()


def PlotTrainingHistory(
    history,
    metrics: List[str] = ['loss'],
    title: str = "Training History",
    height: int = 500,
    savePath: Optional[str] = None
) -> None:
    """
    Plot training history from Keras model using Plotly.
    
    Parameters:
    -----------
    history : keras.callbacks.History
        Training history object
    metrics : List[str]
        List of metrics to plot
    title : str
        Main title
    height : int
        Figure height in pixels
    savePath : Optional[str]
        Path to save figure
    """
    nMetrics = len(metrics)
    
    fig = make_subplots(
        rows=1,
        cols=nMetrics,
        subplot_titles=[m.upper() for m in metrics],
        horizontal_spacing=0.1
    )
    
    colors = {'train': '#3b82f6', 'val': '#ef4444'}
    
    for i, metric in enumerate(metrics):
        # Training metric
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(history.history[metric]) + 1)),
                y=history.history[metric],
                name=f'Train {metric}',
                mode='lines',
                line=dict(color=colors['train'], width=2),
                showlegend=(i == 0)
            ),
            row=1,
            col=i+1
        )
        
        # Validation metric
        valMetric = f'val_{metric}'
        if valMetric in history.history:
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(history.history[valMetric]) + 1)),
                    y=history.history[valMetric],
                    name=f'Val {metric}',
                    mode='lines',
                    line=dict(color=colors['val'], width=2, dash='dash'),
                    showlegend=(i == 0)
                ),
                row=1,
                col=i+1
            )
        
        fig.update_xaxes(title_text="Época", row=1, col=i+1)
        fig.update_yaxes(title_text=metric.upper(), row=1, col=i+1)
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#1f2937')),
        height=height,
        template='plotly_white',
        font=dict(family='Arial', size=12),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.15,
            xanchor="right",
            x=1
        )
    )
    
    if savePath:
        if savePath.endswith('.html'):
            fig.write_html(savePath)
        elif savePath.endswith('.png'):
            fig.write_image(savePath, width=1400, height=height)
        print(f"Figure saved to {savePath}")
    
    fig.show()


def PlotResiduals(
    yTrue: np.ndarray,
    yPred: np.ndarray,
    title: str = "Residual Analysis",
    height: int = 500,
    savePath: Optional[str] = None
) -> None:
    """
    Plot residual analysis using Plotly.
    
    Parameters:
    -----------
    yTrue : np.ndarray
        True values
    yPred : np.ndarray
        Predicted values
    title : str
        Main title
    height : int
        Figure height in pixels
    savePath : Optional[str]
        Path to save figure
    """
    residuals = yTrue - yPred
    
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=('Residuos en el Tiempo', 'Distribución de Residuos'),
        horizontal_spacing=0.12
    )
    
    # Residuals over time
    fig.add_trace(
        go.Scatter(
            x=list(range(len(residuals))),
            y=residuals,
            mode='lines',
            line=dict(color='#3b82f6', width=1.5),
            name='Residuos'
        ),
        row=1,
        col=1
    )
    
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="red",
        row=1,
        col=1
    )
    
    # Residuals distribution
    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=30,
            marker_color='#3b82f6',
            opacity=0.7,
            name='Distribución'
        ),
        row=1,
        col=2
    )
    
    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="red",
        row=1,
        col=2
    )
    
    fig.update_xaxes(title_text="Tiempo", row=1, col=1)
    fig.update_yaxes(title_text="Residuo", row=1, col=1)
    fig.update_xaxes(title_text="Residuo", row=1, col=2)
    fig.update_yaxes(title_text="Frecuencia", row=1, col=2)
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#1f2937')),
        height=height,
        template='plotly_white',
        showlegend=False,
        font=dict(family='Arial', size=12)
    )
    
    if savePath:
        if savePath.endswith('.html'):
            fig.write_html(savePath)
        elif savePath.endswith('.png'):
            fig.write_image(savePath, width=1400, height=height)
        print(f"Figure saved to {savePath}")
    
    fig.show()


def PlotMetricsComparison(
    metricsDict: dict,
    title: str = "Model Metrics Comparison",
    height: int = 500,
    savePath: Optional[str] = None
) -> None:
    """
    Plot comparison of multiple metrics using Plotly.
    
    Parameters:
    -----------
    metricsDict : dict
        Dictionary with metric names as keys and values
    title : str
        Plot title
    height : int
        Figure height in pixels
    savePath : Optional[str]
        Path to save figure
    """
    metrics = list(metricsDict.keys())
    values = list(metricsDict.values())
    
    fig = go.Figure(data=[
        go.Bar(
            x=metrics,
            y=values,
            text=[f'{v:.2f}' for v in values],
            textposition='outside',
            marker_color='#3b82f6',
            marker_line_color='#1e3a8a',
            marker_line_width=1.5
        )
    ])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#1f2937')),
        yaxis_title='Valor',
        height=height,
        template='plotly_white',
        font=dict(family='Arial', size=12),
        showlegend=False
    )
    
    if savePath:
        if savePath.endswith('.html'):
            fig.write_html(savePath)
        elif savePath.endswith('.png'):
            fig.write_image(savePath, width=1000, height=height)
        print(f"Figure saved to {savePath}")
    
    fig.show()


def CreateInteractiveDashboard(
    dataFrame: pd.DataFrame,
    dateColumn: str,
    valueColumns: List[str],
    predictions: Optional[pd.DataFrame] = None,
    title: str = "Interactive Dashboard",
    savePath: Optional[str] = None
) -> None:
    """
    Create an interactive dashboard with multiple plots.
    
    Parameters:
    -----------
    dataFrame : pd.DataFrame
        Historical data
    dateColumn : str
        Name of date column
    valueColumns : List[str]
        Columns to visualize
    predictions : Optional[pd.DataFrame]
        Predictions data with same structure
    title : str
        Dashboard title
    savePath : Optional[str]
        Path to save HTML file
    """
    nCols = len(valueColumns)
    nRows = 2
    
    fig = make_subplots(
        rows=nRows,
        cols=nCols,
        subplot_titles=valueColumns,
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    colors = px.colors.qualitative.Plotly
    
    # Row 1: Time series for each variable
    for i, column in enumerate(valueColumns):
        fig.add_trace(
            go.Scatter(
                x=dataFrame[dateColumn],
                y=dataFrame[column],
                name=column,
                mode='lines',
                line=dict(color=colors[i % len(colors)], width=2),
                showlegend=False
            ),
            row=1,
            col=i+1
        )
        
        if predictions is not None and column in predictions.columns:
            fig.add_trace(
                go.Scatter(
                    x=predictions[dateColumn],
                    y=predictions[column],
                    name=f'{column} (pred)',
                    mode='lines',
                    line=dict(color='red', width=2, dash='dash'),
                    showlegend=False
                ),
                row=1,
                col=i+1
            )
    
    # Row 2: Distribution for each variable
    for i, column in enumerate(valueColumns):
        fig.add_trace(
            go.Histogram(
                x=dataFrame[column],
                name=column,
                marker_color=colors[i % len(colors)],
                opacity=0.7,
                showlegend=False
            ),
            row=2,
            col=i+1
        )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=22, color='#1f2937')),
        height=800,
        template='plotly_white',
        font=dict(family='Arial', size=11)
    )
    
    if savePath:
        fig.write_html(savePath)
        print(f"Dashboard saved to {savePath}")

    fig.show()