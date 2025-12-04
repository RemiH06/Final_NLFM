"""
Functions for building, training, and evaluating forecasting models.
Supports LSTM, GRU, and other architectures.
"""

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from typing import Tuple, Dict, Optional, List


def BuildFFNNModel(
    inputShape: Tuple[int, int],
    hiddenUnits: List[int] = [128, 64, 32],
    dropout: float = 0.2,
    learningRate: float = 0.001,
    outputUnits: int = 1
) -> keras.Model:
    """
    Build Feedforward Neural Network for time series forecasting.
    Input is flattened before passing through dense layers.
    
    Parameters:
    -----------
    inputShape : Tuple[int, int]
        Shape of input data (windowSize, nFeatures)
    hiddenUnits : List[int]
        Number of units in each hidden layer
    dropout : float
        Dropout rate for regularization
    learningRate : float
        Learning rate for optimizer
    outputUnits : int
        Number of output units (forecast horizon)
        
    Returns:
    --------
    keras.Model
        Compiled FFNN model
    """
    model = keras.Sequential()
    
    # Flatten input
    model.add(layers.Flatten(input_shape=inputShape))
    
    # Hidden layers
    for units in hiddenUnits:
        model.add(layers.Dense(units, activation='relu'))
        model.add(layers.Dropout(dropout))
    
    # Output layer
    model.add(layers.Dense(outputUnits))
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learningRate)
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model


def BuildCNNModel(
    inputShape: Tuple[int, int],
    convFilters: List[int] = [64, 32],
    kernelSize: int = 3,
    poolSize: int = 2,
    denseUnits: List[int] = [64, 32],
    dropout: float = 0.2,
    learningRate: float = 0.001,
    outputUnits: int = 1
) -> keras.Model:
    """
    Build 1D Convolutional Neural Network for time series forecasting.
    
    Parameters:
    -----------
    inputShape : Tuple[int, int]
        Shape of input data (windowSize, nFeatures)
    convFilters : List[int]
        Number of filters in each convolutional layer
    kernelSize : int
        Size of convolutional kernel
    poolSize : int
        Size of max pooling window
    denseUnits : List[int]
        Number of units in dense layers after convolution
    dropout : float
        Dropout rate for regularization
    learningRate : float
        Learning rate for optimizer
    outputUnits : int
        Number of output units (forecast horizon)
        
    Returns:
    --------
    keras.Model
        Compiled CNN model
    """
    model = keras.Sequential()
    
    # First convolutional layer
    model.add(layers.Conv1D(
        convFilters[0],
        kernel_size=kernelSize,
        activation='relu',
        input_shape=inputShape
    ))
    model.add(layers.MaxPooling1D(pool_size=poolSize))
    model.add(layers.Dropout(dropout))
    
    # Additional convolutional layers
    for filters in convFilters[1:]:
        model.add(layers.Conv1D(
            filters,
            kernel_size=kernelSize,
            activation='relu'
        ))
        model.add(layers.MaxPooling1D(pool_size=poolSize))
        model.add(layers.Dropout(dropout))
    
    # Flatten
    model.add(layers.Flatten())
    
    # Dense layers
    for units in denseUnits:
        model.add(layers.Dense(units, activation='relu'))
        model.add(layers.Dropout(dropout))
    
    # Output layer
    model.add(layers.Dense(outputUnits))
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learningRate)
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model


def BuildLSTMModel(
    inputShape: Tuple[int, int],
    lstmUnits: List[int] = [64, 32],
    dropout: float = 0.2,
    learningRate: float = 0.001,
    outputUnits: int = 1
) -> keras.Model:
    """
    Build LSTM model for time series forecasting.
    
    Parameters:
    -----------
    inputShape : Tuple[int, int]
        Shape of input data (windowSize, nFeatures)
    lstmUnits : List[int]
        Number of units in each LSTM layer
    dropout : float
        Dropout rate for regularization
    learningRate : float
        Learning rate for optimizer
    outputUnits : int
        Number of output units (forecast horizon)
        
    Returns:
    --------
    keras.Model
        Compiled LSTM model
    """
    model = keras.Sequential()
    
    # First LSTM layer
    model.add(layers.LSTM(
        lstmUnits[0],
        return_sequences=True if len(lstmUnits) > 1 else False,
        input_shape=inputShape
    ))
    model.add(layers.Dropout(dropout))
    
    # Additional LSTM layers
    for i in range(1, len(lstmUnits)):
        returnSeq = i < len(lstmUnits) - 1
        model.add(layers.LSTM(lstmUnits[i], return_sequences=returnSeq))
        model.add(layers.Dropout(dropout))
    
    # Output layer
    model.add(layers.Dense(outputUnits))
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learningRate)
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model


def BuildGRUModel(
    inputShape: Tuple[int, int],
    gruUnits: List[int] = [64, 32],
    dropout: float = 0.2,
    learningRate: float = 0.001,
    outputUnits: int = 1
) -> keras.Model:
    """
    Build GRU model for time series forecasting.
    
    Parameters:
    -----------
    inputShape : Tuple[int, int]
        Shape of input data (windowSize, nFeatures)
    gruUnits : List[int]
        Number of units in each GRU layer
    dropout : float
        Dropout rate for regularization
    learningRate : float
        Learning rate for optimizer
    outputUnits : int
        Number of output units (forecast horizon)
        
    Returns:
    --------
    keras.Model
        Compiled GRU model
    """
    model = keras.Sequential()
    
    # First GRU layer
    model.add(layers.GRU(
        gruUnits[0],
        return_sequences=True if len(gruUnits) > 1 else False,
        input_shape=inputShape
    ))
    model.add(layers.Dropout(dropout))
    
    # Additional GRU layers
    for i in range(1, len(gruUnits)):
        returnSeq = i < len(gruUnits) - 1
        model.add(layers.GRU(gruUnits[i], return_sequences=returnSeq))
        model.add(layers.Dropout(dropout))
    
    # Output layer
    model.add(layers.Dense(outputUnits))
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learningRate)
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model


def BuildBidirectionalLSTMModel(
    inputShape: Tuple[int, int],
    lstmUnits: List[int] = [64, 32],
    dropout: float = 0.2,
    learningRate: float = 0.001,
    outputUnits: int = 1
) -> keras.Model:
    """
    Build Bidirectional LSTM model for time series forecasting.
    
    Parameters:
    -----------
    inputShape : Tuple[int, int]
        Shape of input data (windowSize, nFeatures)
    lstmUnits : List[int]
        Number of units in each LSTM layer
    dropout : float
        Dropout rate for regularization
    learningRate : float
        Learning rate for optimizer
    outputUnits : int
        Number of output units (forecast horizon)
        
    Returns:
    --------
    keras.Model
        Compiled Bidirectional LSTM model
    """
    model = keras.Sequential()
    
    # First Bidirectional LSTM layer
    model.add(layers.Bidirectional(
        layers.LSTM(
            lstmUnits[0],
            return_sequences=True if len(lstmUnits) > 1 else False
        ),
        input_shape=inputShape
    ))
    model.add(layers.Dropout(dropout))
    
    # Additional Bidirectional LSTM layers
    for i in range(1, len(lstmUnits)):
        returnSeq = i < len(lstmUnits) - 1
        model.add(layers.Bidirectional(
            layers.LSTM(lstmUnits[i], return_sequences=returnSeq)
        ))
        model.add(layers.Dropout(dropout))
    
    # Output layer
    model.add(layers.Dense(outputUnits))
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learningRate)
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model


def TrainModel(
    model: keras.Model,
    xTrain: np.ndarray,
    yTrain: np.ndarray,
    xVal: np.ndarray,
    yVal: np.ndarray,
    epochs: int = 100,
    batchSize: int = 32,
    patience: int = 10,
    modelPath: Optional[str] = None
) -> keras.callbacks.History:
    """
    Train a Keras model with early stopping and model checkpointing.
    
    Parameters:
    -----------
    model : keras.Model
        Model to train
    xTrain : np.ndarray
        Training features
    yTrain : np.ndarray
        Training targets
    xVal : np.ndarray
        Validation features
    yVal : np.ndarray
        Validation targets
    epochs : int
        Maximum number of epochs
    batchSize : int
        Batch size for training
    patience : int
        Patience for early stopping
    modelPath : Optional[str]
        Path to save best model
        
    Returns:
    --------
    keras.callbacks.History
        Training history
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    if modelPath:
        callbacks.append(
            ModelCheckpoint(
                modelPath,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        )
    
    history = model.fit(
        xTrain, yTrain,
        validation_data=(xVal, yVal),
        epochs=epochs,
        batch_size=batchSize,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def PredictSequence(
    model: keras.Model,
    data: np.ndarray,
    windowSize: int,
    nSteps: int,
    externalFeatures: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Generate multi-step ahead predictions using recursive forecasting.
    Improved version that can handle external features.
    
    Parameters:
    -----------
    model : keras.Model
        Trained forecasting model
    data : np.ndarray
        Initial data window (windowSize, nFeatures)
    windowSize : int
        Size of input window
    nSteps : int
        Number of steps to forecast
    externalFeatures : Optional[np.ndarray]
        Future values of external features (nSteps, nExternalFeatures)
        If None, uses last known values
        
    Returns:
    --------
    np.ndarray
        Array of predictions
    """
    predictions = []
    currentWindow = data[-windowSize:].copy()
    
    nFeatures = currentWindow.shape[1]
    
    for step in range(nSteps):
        # Reshape for prediction
        inputData = currentWindow.reshape(1, windowSize, nFeatures)
        
        # Predict next value
        nextPred = model.predict(inputData, verbose=0)
        predictions.append(nextPred[0, 0])
        
        # Prepare next input window
        if externalFeatures is not None and step < len(externalFeatures):
            # Use provided external features
            nextRow = np.concatenate([[nextPred[0, 0]], externalFeatures[step]])
        else:
            # Use last known external features (less ideal but necessary)
            if nFeatures > 1:
                nextRow = np.concatenate([[nextPred[0, 0]], currentWindow[-1, 1:]])
            else:
                nextRow = nextPred[0]
        
        # Update window (slide forward)
        currentWindow = np.vstack([currentWindow[1:], nextRow.reshape(1, -1)])
    
    return np.array(predictions)


def PredictSequenceWithExogenous(
    model: keras.Model,
    initialWindow: np.ndarray,
    exogenousFeatures: np.ndarray,
    windowSize: int,
    nSteps: int,
    targetIndex: int = 0
) -> np.ndarray:
    """
    Generate multi-step predictions with known exogenous variables.
    Use this when you have future values of external features.
    
    Parameters:
    -----------
    model : keras.Model
        Trained forecasting model
    initialWindow : np.ndarray
        Initial data window
    exogenousFeatures : np.ndarray
        Future values of exogenous features (nSteps, nExogenousFeatures)
    windowSize : int
        Size of input window
    nSteps : int
        Number of steps to forecast
    targetIndex : int
        Index of the target variable
        
    Returns:
    --------
    np.ndarray
        Array of predictions
    """
    predictions = []
    currentWindow = initialWindow[-windowSize:].copy()
    nFeatures = currentWindow.shape[1]
    
    for i in range(nSteps):
        # Reshape for prediction
        inputData = currentWindow.reshape(1, windowSize, nFeatures)
        
        # Predict next value
        nextPred = model.predict(inputData, verbose=0)
        predictions.append(nextPred[0, 0])
        
        # Create new row with predicted target and known exogenous features
        newRow = exogenousFeatures[i].copy() if i < len(exogenousFeatures) else currentWindow[-1].copy()
        newRow[targetIndex] = nextPred[0, 0]
        
        # Update window
        currentWindow = np.vstack([currentWindow[1:], newRow.reshape(1, -1)])
    
    return np.array(predictions)


def CalculateRMSE(yTrue: np.ndarray, yPred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.
    
    Parameters:
    -----------
    yTrue : np.ndarray
        True values
    yPred : np.ndarray
        Predicted values
        
    Returns:
    --------
    float
        RMSE value
    """
    mse = np.mean((yTrue - yPred) ** 2)
    rmse = np.sqrt(mse)
    return rmse


def CalculateMAE(yTrue: np.ndarray, yPred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Parameters:
    -----------
    yTrue : np.ndarray
        True values
    yPred : np.ndarray
        Predicted values
        
    Returns:
    --------
    float
        MAE value
    """
    mae = np.mean(np.abs(yTrue - yPred))
    return mae


def CalculateMAPE(yTrue: np.ndarray, yPred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Parameters:
    -----------
    yTrue : np.ndarray
        True values
    yPred : np.ndarray
        Predicted values
        
    Returns:
    --------
    float
        MAPE value (as percentage)
    """
    # Avoid division by zero
    mask = yTrue != 0
    mape = np.mean(np.abs((yTrue[mask] - yPred[mask]) / yTrue[mask])) * 100
    return mape


def CalculateSMAPE(yTrue: np.ndarray, yPred: np.ndarray) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error.
    
    Parameters:
    -----------
    yTrue : np.ndarray
        True values
    yPred : np.ndarray
        Predicted values
        
    Returns:
    --------
    float
        sMAPE value (as percentage)
    """
    numerator = np.abs(yTrue - yPred)
    denominator = (np.abs(yTrue) + np.abs(yPred)) / 2
    smape = np.mean(numerator / denominator) * 100
    return smape


def EvaluateModel(
    yTrue: np.ndarray,
    yPred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate multiple evaluation metrics.
    
    Parameters:
    -----------
    yTrue : np.ndarray
        True values
    yPred : np.ndarray
        Predicted values
        
    Returns:
    --------
    Dict[str, float]
        Dictionary with all metrics
    """
    metrics = {
        'RMSE': CalculateRMSE(yTrue, yPred),
        'MAE': CalculateMAE(yTrue, yPred),
        'MAPE': CalculateMAPE(yTrue, yPred),
        'sMAPE': CalculateSMAPE(yTrue, yPred)
    }
    
    return metrics


def SaveModelArchitecture(
    model: keras.Model,
    filePath: str
) -> None:
    """
    Save model architecture to JSON file.
    
    Parameters:
    -----------
    model : keras.Model
        Model to save
    filePath : str
        Path to save JSON file
    """
    modelJson = model.to_json()
    with open(filePath, 'w') as jsonFile:
        jsonFile.write(modelJson)
    print(f"Model architecture saved to {filePath}")


def LoadModelArchitecture(filePath: str) -> keras.Model:
    """
    Load model architecture from JSON file.
    
    Parameters:
    -----------
    filePath : str
        Path to JSON file
        
    Returns:
    --------
    keras.Model
        Loaded model (not compiled)
    """
    with open(filePath, 'r') as jsonFile:
        modelJson = jsonFile.read()
    
    model = keras.models.model_from_json(modelJson)
    print(f"Model architecture loaded from {filePath}")
    
    return model

def BuildTransformerModel(
    inputShape: Tuple[int, int],
    numHeads: int = 4,
    ffDim: int = 128,
    numTransformerBlocks: int = 2,
    mlpUnits: List[int] = [128, 64],
    dropout: float = 0.2,
    learningRate: float = 0.001,
    outputUnits: int = 1
) -> keras.Model:
    """
    Build Transformer model for time series forecasting.
    Uses multi-head self-attention mechanism.
    
    Parameters:
    -----------
    inputShape : Tuple[int, int]
        Shape of input data (windowSize, nFeatures)
    numHeads : int
        Number of attention heads
    ffDim : int
        Hidden layer size in feed forward network inside transformer
    numTransformerBlocks : int
        Number of transformer blocks
    mlpUnits : List[int]
        Units in MLP head
    dropout : float
        Dropout rate
    learningRate : float
        Learning rate for optimizer
    outputUnits : int
        Number of output units
        
    Returns:
    --------
    keras.Model
        Compiled Transformer model
    """
    inputs = keras.Input(shape=inputShape)
    x = inputs
    
    # Transformer blocks
    for _ in range(numTransformerBlocks):
        # Multi-head attention
        attentionOutput = layers.MultiHeadAttention(
            num_heads=numHeads,
            key_dim=inputShape[1],
            dropout=dropout
        )(x, x)
        
        # Skip connection and layer normalization
        x = layers.Add()([x, attentionOutput])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Feed forward network
        ffn = keras.Sequential([
            layers.Dense(ffDim, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(inputShape[1])
        ])
        ffnOutput = ffn(x)
        
        # Skip connection and layer normalization
        x = layers.Add()([x, ffnOutput])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Global average pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # MLP head
    for units in mlpUnits:
        x = layers.Dense(units, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
    
    # Output layer
    outputs = layers.Dense(outputUnits)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile
    optimizer = keras.optimizers.Adam(learning_rate=learningRate)
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model