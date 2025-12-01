"""
Functions for feature engineering and data preparation for modeling.
Includes windowing, scaling, and train-test splits.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Tuple, List


def CreateTimeSeriesWindows(
    data: np.ndarray,
    windowSize: int,
    forecastHorizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows for time series forecasting.
    
    Parameters:
    -----------
    data : np.ndarray
        Input time series data
    windowSize : int
        Size of the input window (lookback period)
    forecastHorizon : int
        Number of steps ahead to forecast
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        X (features) and y (targets) arrays
    """
    X, y = [], []
    
    for i in range(len(data) - windowSize - forecastHorizon + 1):
        X.append(data[i:i + windowSize])
        y.append(data[i + windowSize:i + windowSize + forecastHorizon])
    
    return np.array(X), np.array(y)


def ScaleData(
    dataFrame: pd.DataFrame,
    columns: List[str],
    scalerType: str = 'minmax'
) -> Tuple[pd.DataFrame, object]:
    """
    Scale specified columns in DataFrame.
    
    Parameters:
    -----------
    dataFrame : pd.DataFrame
        Input DataFrame
    columns : List[str]
        Columns to scale
    scalerType : str
        Type of scaler: 'minmax' or 'standard'
        
    Returns:
    --------
    Tuple[pd.DataFrame, object]
        Scaled DataFrame and fitted scaler
    """
    df = dataFrame.copy()
    
    if scalerType == 'minmax':
        scaler = MinMaxScaler()
    elif scalerType == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("scalerType must be 'minmax' or 'standard'")
    
    df[columns] = scaler.fit_transform(df[columns])
    
    return df, scaler


def InverseScaleData(
    data: np.ndarray,
    scaler: object,
    columnIndex: int = 0
) -> np.ndarray:
    """
    Inverse transform scaled data back to original scale.
    
    Parameters:
    -----------
    data : np.ndarray
        Scaled data
    scaler : object
        Fitted scaler object
    columnIndex : int
        Index of the column to inverse transform (for multivariate)
        
    Returns:
    --------
    np.ndarray
        Data in original scale
    """
    # Create dummy array with same shape as original features
    nFeatures = scaler.n_features_in_
    dummy = np.zeros((len(data), nFeatures))
    dummy[:, columnIndex] = data.flatten()
    
    # Inverse transform
    inversed = scaler.inverse_transform(dummy)
    
    return inversed[:, columnIndex]


def SplitTrainTest(
    dataFrame: pd.DataFrame,
    trainRatio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into train and test sets.
    Maintains temporal order (no shuffle).
    
    Parameters:
    -----------
    dataFrame : pd.DataFrame
        Input DataFrame
    trainRatio : float
        Proportion of data for training
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        Train and test DataFrames
    """
    splitIndex = int(len(dataFrame) * trainRatio)
    
    trainDf = dataFrame.iloc[:splitIndex].copy()
    testDf = dataFrame.iloc[splitIndex:].copy()
    
    return trainDf, testDf


def PrepareMultivariateData(
    dataFrame: pd.DataFrame,
    targetColumn: str,
    featureColumns: List[str],
    windowSize: int,
    forecastHorizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare multivariate time series data for modeling.
    
    Parameters:
    -----------
    dataFrame : pd.DataFrame
        Input DataFrame
    targetColumn : str
        Name of target column
    featureColumns : List[str]
        List of feature column names
    windowSize : int
        Size of input window
    forecastHorizon : int
        Number of steps to forecast
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        X (features) and y (targets) arrays
    """
    # Combine target with features
    allColumns = [targetColumn] + featureColumns
    data = dataFrame[allColumns].values
    
    X, y = [], []
    
    for i in range(len(data) - windowSize - forecastHorizon + 1):
        X.append(data[i:i + windowSize])
        # Only predict target column
        y.append(data[i + windowSize:i + windowSize + forecastHorizon, 0])
    
    return np.array(X), np.array(y)