import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Union

# ===================================================================
#           Private Helper Functions (Internal Use)
# ===================================================================

def _calculate_pcc_protein_level(
    protein_names: List[str], 
    y_pred: np.ndarray, 
    y_true: np.ndarray
) -> pd.DataFrame:
    """Calculates Pearson Correlation Coefficient (PCC) for each protein (column)."""
    pccs = []
    for i, name in enumerate(protein_names):
        pred_col = y_pred[:, i]
        true_col = y_true[:, i]
        
        # Check for zero variance, which makes correlation undefined
        if np.std(pred_col) == 0 or np.std(true_col) == 0:
            pcc = np.nan
        else:
            pcc, _ = pearsonr(pred_col, true_col)
        pccs.append(pcc)

    return pd.DataFrame({"PCC": pccs}, index=protein_names)

def _calculate_rmse_protein_level(
    protein_names: List[str], 
    y_pred: np.ndarray, 
    y_true: np.ndarray
) -> pd.DataFrame:
    """Calculates Root Mean Squared Error (RMSE) for each protein (column)."""
    rmses = []
    for i, name in enumerate(protein_names):
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        rmses.append(rmse)
        
    return pd.DataFrame({"RMSE": rmses}, index=protein_names)

def _calculate_pcc_cell_level(
    cell_names: List[str], 
    y_pred: np.ndarray, 
    y_true: np.ndarray
) -> pd.DataFrame:
    """Calculates Pearson Correlation Coefficient (PCC) for each cell (row)."""
    pccs = []
    for i, name in enumerate(cell_names):
        pred_row = y_pred[i, :]
        true_row = y_true[i, :]
        
        if np.std(pred_row) == 0 or np.std(true_row) == 0:
            pcc = np.nan
        else:
            pcc, _ = pearsonr(pred_row, true_row)
        pccs.append(pcc)

    df = pd.DataFrame({"PCC": pccs}, index=cell_names)
    df.index.name = None
    return df

def _calculate_rmse_cell_level(
    cell_names: List[str], 
    y_pred: np.ndarray, 
    y_true: np.ndarray
) -> pd.DataFrame:
    """Calculates Root Mean Squared Error (RMSE) for each cell (row)."""
    rmses = []
    for i, name in enumerate(cell_names):
        rmse = np.sqrt(mean_squared_error(y_true[i, :], y_pred[i, :]))
        rmses.append(rmse)
        
    df = pd.DataFrame({"RMSE": rmses}, index=cell_names)
    df.index.name = None
    return df

# ===================================================================
#           Public Function (What you will import and use)
# ===================================================================

def evaluate_prediction(
    y_true: np.ndarray,
    y_pred: Union[np.ndarray, torch.Tensor],
    protein_names: List[str],
    cell_names: List[str],
    zscore_protein_rmse: bool = True,
    l2norm_cell_rmse: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Performs a comprehensive evaluation of protein expression predictions.

    This function calculates Pearson Correlation and RMSE from both the protein
    and cell perspectives, applying appropriate normalizations for RMSE calculations
    as seen in standard evaluation scripts.

    Parameters:
    ----------
    y_true : np.ndarray
        Ground truth protein expression. Shape: (n_cells, n_proteins).
    y_pred : Union[np.ndarray, torch.Tensor]
        Predicted protein expression from the model. Shape: (n_cells, n_proteins).
    protein_names : List[str]
        A list of protein names corresponding to the columns.
    cell_names : List[str]
        A list of cell names or barcodes corresponding to the rows.
    zscore_protein_rmse : bool, optional
        If True, applies Z-score normalization before calculating protein-level RMSE. 
        Defaults to True.
    l2norm_cell_rmse : bool, optional
        If True, applies L2 normalization before calculating cell-level RMSE. 
        Defaults to True.

    Returns:
    -------
    Dict[str, pd.DataFrame]
        A dictionary containing four DataFrames:
        - 'protein_pcc': PCC for each protein.
        - 'protein_rmse': RMSE for each protein.
        - 'cell_pcc': PCC for each cell.
        - 'cell_rmse': RMSE for each cell.
    """
    # --- Input Validation and Preparation ---
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
        
    if not isinstance(y_true, np.ndarray):
        raise TypeError(f"y_true must be a NumPy array, but got {type(y_true)}")
        
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true has shape {y_true.shape} and y_pred has shape {y_pred.shape}")
    
    print("✓ Inputs validated. Starting evaluation...")
    
    # --- Protein-Level Evaluation ---
    print("  - Calculating Protein-Level Metrics...")
    protein_pcc_df = _calculate_pcc_protein_level(protein_names, y_pred, y_true)
    
    y_true_protein_norm, y_pred_protein_norm = y_true, y_pred
    if zscore_protein_rmse:
        print("    - Applying Z-score normalization for protein RMSE.")
        scaler_true = StandardScaler()
        scaler_pred = StandardScaler()
        y_true_protein_norm = scaler_true.fit_transform(y_true)
        y_pred_protein_norm = scaler_pred.fit_transform(y_pred)
        
    protein_rmse_df = _calculate_rmse_protein_level(protein_names, y_pred_protein_norm, y_true_protein_norm)

    # --- Cell-Level Evaluation ---
    print("  - Calculating Cell-Level Metrics...")
    cell_pcc_df = _calculate_pcc_cell_level(cell_names, y_pred, y_true)
    
    y_true_cell_norm, y_pred_cell_norm = y_true, y_pred
    if l2norm_cell_rmse:
        print("    - Applying L2 normalization for cell RMSE.")
        true_norms = np.linalg.norm(y_true, axis=1, keepdims=True)
        pred_norms = np.linalg.norm(y_pred, axis=1, keepdims=True)
        y_true_cell_norm = y_true / (true_norms + 1e-8)
        y_pred_cell_norm = y_pred / (pred_norms + 1e-8)
        
    cell_rmse_df = _calculate_rmse_cell_level(cell_names, y_pred_cell_norm, y_true_cell_norm)
    
    print("✓ Evaluation complete.")
    
    return {
        'protein_pcc': protein_pcc_df,
        'protein_rmse': protein_rmse_df,
        'cell_pcc': cell_pcc_df,
        'cell_rmse': cell_rmse_df,
    }

