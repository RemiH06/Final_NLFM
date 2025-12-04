"""
Functions for acquiring data from Banxico API.
All API interactions and data fetching logic.
"""

import requests
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional


def GetBanxicoToken(tokenPath: str) -> str:
    """
    Read Banxico API token from file.
    Supports multiple formats:
    - Plain text: just the token
    - Key-value: token="value"
    - Key-value with quotes: token='value'
    
    Parameters:
    -----------
    tokenPath : str
        Path to file containing the Banxico token
        
    Returns:
    --------
    str
        API token
    """
    with open(tokenPath, 'r') as file:
        content = file.read().strip()
    
    # Check if it's in key-value format
    if '=' in content:
        # Extract value after '='
        token = content.split('=')[1].strip()
        # Remove quotes if present
        token = token.strip('"').strip("'")
    else:
        # Plain text format
        token = content
    
    return token


def FetchSeriesData(
    seriesId: str,
    token: str,
    startDate: str,
    endDate: str
) -> pd.DataFrame:
    """
    Fetch time series data from Banxico API.
    
    Parameters:
    -----------
    seriesId : str
        Banxico series identifier (e.g., 'SF43718')
    token : str
        Banxico API token
    startDate : str
        Start date in format 'YYYY-MM-DD'
    endDate : str
        End date in format 'YYYY-MM-DD'
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns ['fecha', 'valor']
    """
    baseUrl = "https://www.banxico.org.mx/SieAPIRest/service/v1/series"
    url = f"{baseUrl}/{seriesId}/datos/{startDate}/{endDate}"
    
    headers = {
        'Bmx-Token': token,
        'Accept': 'application/json'
    }
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    data = response.json()
    seriesData = data['bmx']['series'][0]['datos']
    
    df = pd.DataFrame(seriesData)
    df['fecha'] = pd.to_datetime(df['fecha'], format='%d/%m/%Y')
    df['valor'] = pd.to_numeric(df['dato'], errors='coerce')
    df = df[['fecha', 'valor']]
    
    return df


def FetchMultipleSeries(
    seriesIds: List[str],
    token: str,
    startDate: str,
    endDate: str
) -> Dict[str, pd.DataFrame]:
    """
    Fetch multiple time series from Banxico API.
    
    Parameters:
    -----------
    seriesIds : List[str]
        List of Banxico series identifiers
    token : str
        Banxico API token
    startDate : str
        Start date in format 'YYYY-MM-DD'
    endDate : str
        End date in format 'YYYY-MM-DD'
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary mapping series IDs to their DataFrames
    """
    seriesData = {}
    
    for seriesId in seriesIds:
        print(f"Fetching {seriesId}...")
        df = FetchSeriesData(seriesId, token, startDate, endDate)
        seriesData[seriesId] = df
        
    return seriesData


def MergeSeriesOnDate(
    seriesDict: Dict[str, pd.DataFrame],
    seriesNames: Dict[str, str],
    dateColumn: str = 'fecha'
) -> pd.DataFrame:
    """
    Merge multiple series into single DataFrame by date.
    
    Parameters:
    -----------
    seriesDict : Dict[str, pd.DataFrame]
        Dictionary of series DataFrames
    seriesNames : Dict[str, str]
        Mapping of series IDs to column names
    dateColumn : str
        Name of the date column
        
    Returns:
    --------
    pd.DataFrame
        Merged DataFrame with all series
    """
    mergedDf = None
    
    for seriesId, df in seriesDict.items():
        columnName = seriesNames.get(seriesId, seriesId)
        df = df.rename(columns={'valor': columnName})
        
        if mergedDf is None:
            mergedDf = df
        else:
            mergedDf = mergedDf.merge(
                df, 
                on=dateColumn, 
                how='outer'
            )
    
    mergedDf = mergedDf.sort_values(dateColumn).reset_index(drop=True)
    
    return mergedDf


def SaveDataToCSV(
    dataFrame: pd.DataFrame,
    filePath: str,
    includeIndex: bool = False
) -> None:
    """
    Save DataFrame to CSV file.
    
    Parameters:
    -----------
    dataFrame : pd.DataFrame
        DataFrame to save
    filePath : str
        Destination file path
    includeIndex : bool
        Whether to include index in CSV
    """
    dataFrame.to_csv(filePath, index=includeIndex)
    print(f"Data saved to {filePath}")