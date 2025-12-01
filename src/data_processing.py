"""
Functions for data cleaning and preprocessing.
Handles missing values, outliers, and data quality checks.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def HandleMissingValues(
    dataFrame: pd.DataFrame,
    method: str = 'ffill',
    columnName: Optional[str] = None
) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Parameters:
    -----------
    dataFrame : pd.DataFrame
        Input DataFrame
    method : str
        Method to handle missing values: 'ffill', 'bfill', 'interpolate', 'drop'
    columnName : Optional[str]
        Specific column to process. If None, applies to all columns
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with handled missing values
    """
    df = dataFrame.copy()
    
    if columnName:
        if method == 'ffill':
            df[columnName] = df[columnName].fillna(method='ffill')
        elif method == 'bfill':
            df[columnName] = df[columnName].fillna(method='bfill')
        elif method == 'interpolate':
            df[columnName] = df[columnName].interpolate(method='linear')
        elif method == 'drop':
            df = df.dropna(subset=[columnName])
    else:
        if method == 'ffill':
            df = df.fillna(method='ffill')
        elif method == 'bfill':
            df = df.fillna(method='bfill')
        elif method == 'interpolate':
            df = df.interpolate(method='linear')
        elif method == 'drop':
            df = df.dropna()
    
    return df


def DetectOutliers(
    series: pd.Series,
    method: str = 'iqr',
    threshold: float = 1.5
) -> pd.Series:
    """
    Detect outliers in a series.
    
    Parameters:
    -----------
    series : pd.Series
        Input series
    method : str
        Method to detect outliers: 'iqr', 'zscore'
    threshold : float
        Threshold for outlier detection
        
    Returns:
    --------
    pd.Series
        Boolean series indicating outliers
    """
    if method == 'iqr':
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lowerBound = q1 - threshold * iqr
        upperBound = q3 + threshold * iqr
        outliers = (series < lowerBound) | (series > upperBound)
        
    elif method == 'zscore':
        mean = series.mean()
        std = series.std()
        zScores = np.abs((series - mean) / std)
        outliers = zScores > threshold
        
    return outliers


def HandleOutliers(
    dataFrame: pd.DataFrame,
    columnName: str,
    method: str = 'iqr',
    threshold: float = 1.5,
    action: str = 'cap'
) -> pd.DataFrame:
    """
    Handle outliers in a specific column.
    
    Parameters:
    -----------
    dataFrame : pd.DataFrame
        Input DataFrame
    columnName : str
        Column to process
    method : str
        Detection method: 'iqr', 'zscore'
    threshold : float
        Threshold for outlier detection
    action : str
        Action to take: 'cap', 'remove', 'median'
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with handled outliers
    """
    df = dataFrame.copy()
    series = df[columnName]
    
    outliers = DetectOutliers(series, method, threshold)
    
    if action == 'remove':
        df = df[~outliers]
    elif action == 'cap':
        if method == 'iqr':
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lowerBound = q1 - threshold * iqr
            upperBound = q3 + threshold * iqr
            df.loc[df[columnName] < lowerBound, columnName] = lowerBound
            df.loc[df[columnName] > upperBound, columnName] = upperBound
    elif action == 'median':
        median = series.median()
        df.loc[outliers, columnName] = median
        
    return df


def CreateLagFeatures(
    dataFrame: pd.DataFrame,
    targetColumn: str,
    lags: list
) -> pd.DataFrame:
    """
    Create lag features for time series.
    
    Parameters:
    -----------
    dataFrame : pd.DataFrame
        Input DataFrame
    targetColumn : str
        Column to create lags from
    lags : list
        List of lag periods
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with lag features
    """
    df = dataFrame.copy()
    
    for lag in lags:
        df[f'{targetColumn}_lag_{lag}'] = df[targetColumn].shift(lag)
        
    return df


def CreateRollingFeatures(
    dataFrame: pd.DataFrame,
    targetColumn: str,
    windows: list,
    functions: list = ['mean', 'std']
) -> pd.DataFrame:
    """
    Create rolling window features.
    
    Parameters:
    -----------
    dataFrame : pd.DataFrame
        Input DataFrame
    targetColumn : str
        Column to create rolling features from
    windows : list
        List of window sizes
    functions : list
        List of aggregation functions
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with rolling features
    """
    df = dataFrame.copy()
    
    for window in windows:
        for func in functions:
            columnName = f'{targetColumn}_rolling_{window}_{func}'
            df[columnName] = df[targetColumn].rolling(window=window).agg(func)
            
    return df


def CalculateReturns(
    dataFrame: pd.DataFrame,
    columnName: str,
    periods: int = 1
) -> pd.DataFrame:
    """
    Calculate percentage returns.
    
    Parameters:
    -----------
    dataFrame : pd.DataFrame
        Input DataFrame
    columnName : str
        Column to calculate returns from
    periods : int
        Number of periods for return calculation
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with returns column
    """
    df = dataFrame.copy()
    returnColumnName = f'{columnName}_return_{periods}d'
    df[returnColumnName] = df[columnName].pct_change(periods=periods) * 100
    
    return df