"""
PCA-based dimension reduction for economic indicators.

This module implements Principal Component Analysis (PCA) for reducing the dimensionality
of economic indicators, which is a key part of the GDP-anchored variable modeling system.
It supports finding optimal components, transforming data, and analyzing component contributions.
"""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Set up logging
logger = logging.getLogger(__name__)

class PCAHandler:
    """
    Handles PCA-based dimension reduction for economic indicators.
    
    This class provides methods for applying PCA to economic time series data,
    transforming data to and from the reduced space, and analyzing the components
    to understand their relationship with GDP and other variables.
    """
    
    def __init__(
        self,
        n_components: Optional[int] = None,
        variance_threshold: float = 0.95,
        scale_data: bool = True
    ):
        """
        Initialize the PCA handler.
        
        Args:
            n_components: Number of components to keep (if None, determined by variance_threshold)
            variance_threshold: Minimum cumulative explained variance to retain (default: 0.95)
            scale_data: Whether to standardize data before PCA (default: True)
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.scale_data = scale_data
        
        self.pca = None
        self.scaler = StandardScaler() if scale_data else None
        self.feature_names = None
        self.component_names = None
        self.component_descriptions = {}
    
    def fit(
        self, 
        data: pd.DataFrame,
        target_col: Optional[str] = None
    ) -> 'PCAHandler':
        """
        Fit PCA to the data.
        
        Args:
            data: DataFrame with economic indicators as columns
            target_col: Optional target column to exclude from PCA
            
        Returns:
            Self for method chaining
        """
        # Store feature names (excluding target if provided)
        if target_col and target_col in data.columns:
            self.feature_names = [col for col in data.columns if col != target_col]
            X = data[self.feature_names].copy()
        else:
            self.feature_names = data.columns.tolist()
            X = data.copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale data if requested
        if self.scale_data:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X.values
        
        # Determine number of components
        if self.n_components is None:
            # Use a temporary PCA to determine optimal components
            temp_pca = PCA()
            temp_pca.fit(X_scaled)
            
            # Find number of components needed to reach variance threshold
            explained_variance_ratio_cumsum = np.cumsum(temp_pca.explained_variance_ratio_)
            self.n_components = np.argmax(explained_variance_ratio_cumsum >= self.variance_threshold) + 1
            
            logger.info(f"Determined optimal number of components: {self.n_components} "
                       f"(explains {explained_variance_ratio_cumsum[self.n_components-1]:.2%} of variance)")
        
        # Initialize PCA with determined number of components
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X_scaled)
        
        # Generate component names
        self.component_names = [f"PC{i+1}" for i in range(self.n_components)]
        
        # Automatically generate component descriptions
        self._generate_component_descriptions()
        
        return self
    
    def transform(
        self, 
        data: pd.DataFrame,
        include_original: bool = False
    ) -> pd.DataFrame:
        """
        Transform data to principal component space.
        
        Args:
            data: DataFrame with same columns as used for fitting
            include_original: Whether to include original features in output
            
        Returns:
            DataFrame with principal components as columns
        """
        if self.pca is None:
            raise ValueError("PCA model must be fitted before transforming data")
        
        # Ensure same columns as during fit
        if not all(col in data.columns for col in self.feature_names):
            missing_cols = [col for col in self.feature_names if col not in data.columns]
            raise ValueError(f"Data missing columns used during fit: {missing_cols}")
        
        # Extract features in the same order as during fit
        X = data[self.feature_names].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale data if requested
        if self.scale_data:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Transform to component space
        components = self.pca.transform(X_scaled)
        
        # Create DataFrame with components
        pc_df = pd.DataFrame(
            components,
            index=data.index,
            columns=self.component_names
        )
        
        if include_original:
            # Include original features
            result = pd.concat([data, pc_df], axis=1)
            return result
        else:
            return pc_df
    
    def inverse_transform(
        self, 
        components: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Transform data from principal component space back to original space.
        
        Args:
            components: DataFrame with principal components
            
        Returns:
            DataFrame with original features reconstructed
        """
        if self.pca is None:
            raise ValueError("PCA model must be fitted before inverse transforming")
        
        # Extract component values
        component_values = components[self.component_names].values
        
        # Inverse transform to original scaled space
        X_scaled_reconstructed = self.pca.inverse_transform(component_values)
        
        # Inverse scale if data was scaled
        if self.scale_data:
            X_reconstructed = self.scaler.inverse_transform(X_scaled_reconstructed)
        else:
            X_reconstructed = X_scaled_reconstructed
        
        # Create DataFrame with original features
        reconstructed_df = pd.DataFrame(
            X_reconstructed,
            index=components.index,
            columns=self.feature_names
        )
        
        return reconstructed_df
    
    def _generate_component_descriptions(self):
        """
        Generate descriptions for principal components based on loadings.
        """
        if self.pca is None or self.feature_names is None:
            return
        
        # For each component, find the top contributing features
        for i, component_name in enumerate(self.component_names):
            # Get loadings for this component
            loadings = self.pca.components_[i]
            
            # Sort features by absolute loading values
            sorted_indices = np.argsort(np.abs(loadings))[::-1]
            top_features = [(self.feature_names[idx], loadings[idx]) for idx in sorted_indices[:5]]
            
            # Create description
            description = f"Component {i+1} - Top contributors: "
            for feature, loading in top_features:
                direction = "+" if loading > 0 else "-"
                description += f"{feature} ({direction}{abs(loading):.3f}), "
            
            self.component_descriptions[component_name] = description[:-2]  # Remove trailing comma and space
    
    def get_explained_variance_ratio(self) -> np.ndarray:
        """
        Get the explained variance ratio for each component.
        
        Returns:
            Array with explained variance ratio for each component
        """
        if self.pca is None:
            raise ValueError("PCA model must be fitted first")
        
        return self.pca.explained_variance_ratio_
    
    def get_cumulative_explained_variance(self) -> np.ndarray:
        """
        Get the cumulative explained variance ratio.
        
        Returns:
            Array with cumulative explained variance
        """
        if self.pca is None:
            raise ValueError("PCA model must be fitted first")
        
        return np.cumsum(self.pca.explained_variance_ratio_)
    
    def get_component_loadings(self) -> pd.DataFrame:
        """
        Get component loadings for each feature.
        
        Returns:
            DataFrame with features as index and components as columns
        """
        if self.pca is None or self.feature_names is None:
            raise ValueError("PCA model must be fitted first")
        
        loadings = pd.DataFrame(
            self.pca.components_.T,
            index=self.feature_names,
            columns=self.component_names
        )
        
        return loadings
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Calculate the overall importance of each feature across all components.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.pca is None or self.feature_names is None:
            raise ValueError("PCA model must be fitted first")
        
        # Get loadings and explained variance
        loadings = self.pca.components_
        explained_variance = self.pca.explained_variance_ratio_
        
        # Calculate importance for each feature
        feature_importance = {}
        
        for i, feature in enumerate(self.feature_names):
            # Weighted sum of squared loadings
            importance = sum(
                (loadings[j, i] ** 2) * explained_variance[j]
                for j in range(self.n_components)
            )
            feature_importance[feature] = importance
        
        # Normalize to sum to 1
        total_importance = sum(feature_importance.values())
        for feature in feature_importance:
            feature_importance[feature] /= total_importance
        
        return feature_importance
    
    def plot_explained_variance(
        self,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot explained variance by components.
        
        Args:
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib figure
        """
        if self.pca is None:
            raise ValueError("PCA model must be fitted first")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get explained variance ratio
        explained_variance = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        # Plot individual and cumulative explained variance
        bar_positions = np.arange(len(explained_variance))
        ax.bar(bar_positions, explained_variance, alpha=0.7, label='Individual')
        ax.plot(bar_positions, cumulative_variance, 'ro-', label='Cumulative')
        
        # Add threshold line
        ax.axhline(y=self.variance_threshold, color='g', linestyle='--', 
                  label=f'{self.variance_threshold:.0%} Threshold')
        
        # Add labels and legend
        ax.set_xlabel('Principal Components')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_title('Explained Variance by Principal Components')
        ax.set_xticks(bar_positions)
        ax.set_xticklabels([f'PC{i+1}' for i in bar_positions])
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_component_loadings(
        self,
        component_idx: int = 0,
        n_features: int = 10,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot feature loadings for a specific component.
        
        Args:
            component_idx: Index of component to plot (0-based)
            n_features: Number of top features to display
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib figure
        """
        if self.pca is None or self.feature_names is None:
            raise ValueError("PCA model must be fitted first")
        
        if component_idx >= self.n_components:
            raise ValueError(f"Component index {component_idx} out of range (max: {self.n_components-1})")
        
        # Get loadings for this component
        loadings = self.pca.components_[component_idx]
        
        # Sort features by absolute loading values
        sorted_indices = np.argsort(np.abs(loadings))[::-1][:n_features]
        sorted_features = [self.feature_names[i] for i in sorted_indices]
        sorted_loadings = loadings[sorted_indices]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot horizontal bar chart
        colors = ['b' if loading > 0 else 'r' for loading in sorted_loadings]
        y_pos = np.arange(len(sorted_features))
        ax.barh(y_pos, sorted_loadings, color=colors)
        
        # Add labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_features)
        ax.set_xlabel('Loading Value')
        ax.set_title(f'Feature Loadings for {self.component_names[component_idx]}')
        
        plt.tight_layout()
        return fig
    
    def get_component_correlations(
        self, 
        data: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate correlations between components and original features.
        
        Args:
            data: DataFrame with original data
            features: List of features to include (None for all)
            
        Returns:
            DataFrame with correlations
        """
        if self.pca is None:
            raise ValueError("PCA model must be fitted first")
        
        # Transform data to component space
        pc_df = self.transform(data)
        
        # Select features to include in correlation
        if features is None:
            features = self.feature_names
        else:
            features = [f for f in features if f in data.columns]
        
        # Calculate correlations
        combined = pd.concat([pc_df, data[features]], axis=1)
        correlations = combined.corr().loc[self.component_names, features]
        
        return correlations
    
    def find_gdp_related_components(
        self,
        data: pd.DataFrame,
        gdp_col: str,
        threshold: float = 0.3
    ) -> List[Tuple[str, float]]:
        """
        Find components that are most correlated with GDP.
        
        Args:
            data: DataFrame containing both the GDP column and features
            gdp_col: Name of GDP column
            threshold: Minimum absolute correlation to consider
            
        Returns:
            List of (component_name, correlation) tuples sorted by absolute correlation
        """
        if self.pca is None:
            raise ValueError("PCA model must be fitted first")
        
        if gdp_col not in data.columns:
            raise ValueError(f"GDP column '{gdp_col}' not found in data")
        
        # Transform data to component space
        pc_df = self.transform(data)
        
        # Add GDP column
        pc_df[gdp_col] = data[gdp_col]
        
        # Calculate correlations with GDP
        gdp_correlations = []
        for component in self.component_names:
            corr = pc_df[[component, gdp_col]].corr().loc[component, gdp_col]
            gdp_correlations.append((component, corr))
        
        # Filter by threshold and sort by absolute correlation
        filtered_correlations = [
            (comp, corr) for comp, corr in gdp_correlations
            if abs(corr) >= threshold
        ]
        sorted_correlations = sorted(
            filtered_correlations,
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        return sorted_correlations
    
    def save(self, filepath: str):
        """
        Save the PCA model to a file.
        
        Args:
            filepath: Path to save the model
        """
        import pickle
        
        # Create dictionary with all necessary attributes
        save_dict = {
            'n_components': self.n_components,
            'variance_threshold': self.variance_threshold,
            'scale_data': self.scale_data,
            'pca': self.pca,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'component_names': self.component_names,
            'component_descriptions': self.component_descriptions
        }
        
        # Save to file
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        
        logger.info(f"PCA model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'PCAHandler':
        """
        Load a PCA model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded PCAHandler instance
        """
        import pickle
        
        # Load from file
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        # Create a new instance
        instance = cls(
            n_components=save_dict['n_components'],
            variance_threshold=save_dict['variance_threshold'],
            scale_data=save_dict['scale_data']
        )
        
        # Restore attributes
        instance.pca = save_dict['pca']
        instance.scaler = save_dict['scaler']
        instance.feature_names = save_dict['feature_names']
        instance.component_names = save_dict['component_names']
        instance.component_descriptions = save_dict['component_descriptions']
        
        logger.info(f"PCA model loaded from {filepath}")
        return instance


def find_optimal_components(
    data: pd.DataFrame,
    variance_threshold: float = 0.95,
    max_components: Optional[int] = None,
    scale_data: bool = True
) -> Tuple[int, float]:
    """
    Find the optimal number of principal components based on explained variance.
    
    Args:
        data: DataFrame with features
        variance_threshold: Minimum cumulative explained variance to retain
        max_components: Maximum number of components to consider
        scale_data: Whether to standardize data before PCA
        
    Returns:
        Tuple of (optimal_components, explained_variance)
    """
    # Handle missing values
    data_clean = data.fillna(data.mean())
    
    # Scale data if requested
    if scale_data:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data_clean)
    else:
        X_scaled = data_clean.values
    
    # Set maximum number of components
    if max_components is None:
        max_components = min(len(data_clean.columns), len(data_clean) - 1)
    else:
        max_components = min(max_components, len(data_clean.columns), len(data_clean) - 1)
    
    # Initialize PCA
    pca = PCA(n_components=max_components)
    pca.fit(X_scaled)
    
    # Calculate cumulative explained variance
    explained_variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)
    
    # Find number of components needed to reach variance threshold
    for i, cumulative_variance in enumerate(explained_variance_ratio_cumsum):
        if cumulative_variance >= variance_threshold:
            return i + 1, cumulative_variance
    
    # If threshold not reached, return max components
    return max_components, explained_variance_ratio_cumsum[-1]