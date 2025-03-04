"""
Feature clustering module for economic indicator grouping.

This module implements methods for clustering similar economic variables,
which complements the PCA-based dimension reduction approach. It supports
hierarchical and k-means clustering methods and provides tools for determining
the optimal number of clusters and extracting representative variables.
"""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import pearsonr
import matplotlib.cm as cm

# Set up logging
logger = logging.getLogger(__name__)

class FeatureClustering:
    """
    Handles clustering of features for economic indicator grouping.
    
    This class provides methods for applying different clustering techniques
    to group similar economic indicators, identifying representative variables
    for each cluster, and analyzing the cluster structure. It complements PCA
    by providing an alternative, more interpretable dimension reduction approach.
    """
    
    def __init__(
        self,
        method: str = 'hierarchical',
        n_clusters: Optional[int] = None,
        scale_data: bool = True
    ):
        """
        Initialize the feature clustering handler.
        
        Args:
            method: Clustering method ('hierarchical' or 'kmeans')
            n_clusters: Number of clusters (if None, determined automatically)
            scale_data: Whether to standardize data before clustering (default: True)
        """
        self.method = method
        self.n_clusters = n_clusters
        self.scale_data = scale_data
        
        self.clusterer = None
        self.scaler = StandardScaler() if scale_data else None
        self.feature_names = None
        self.cluster_labels = None
        self.cluster_representatives = {}
        self.feature_clusters = {}
        self.linkage_matrix = None  # For hierarchical clustering
    
    def fit(
        self, 
        data: pd.DataFrame,
        correlation_threshold: float = 0.7,
        max_clusters: int = 10
    ) -> 'FeatureClustering':
        """
        Fit clustering to the data.
        
        Args:
            data: DataFrame with economic indicators as columns
            correlation_threshold: Threshold for determining related features (hierarchical only)
            max_clusters: Maximum number of clusters to consider if n_clusters is None
            
        Returns:
            Self for method chaining
        """
        # Store feature names
        self.feature_names = data.columns.tolist()
        
        # Handle missing values
        X = data.copy().fillna(data.mean())
        
        # Scale data if requested
        if self.scale_data:
            X_scaled = self.scaler.fit_transform(X.T)  # Transpose to cluster features
        else:
            X_scaled = X.T.values
        
        # Calculate correlation matrix for use in clustering
        correlation_matrix = data.corr().abs()
        
        # Determine number of clusters if not specified
        if self.n_clusters is None:
            self.n_clusters = self._determine_optimal_clusters(
                X_scaled, correlation_matrix, max_clusters
            )
            logger.info(f"Determined optimal number of clusters: {self.n_clusters}")
        
        # Apply clustering
        if self.method == 'hierarchical':
            # Compute linkage matrix for hierarchical clustering
            self.linkage_matrix = linkage(X_scaled, method='ward')
            
            # Create clusterer
            self.clusterer = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                linkage='ward'
            )
            
            # Fit clusterer
            self.cluster_labels = self.clusterer.fit_predict(X_scaled)
            
        elif self.method == 'kmeans':
            # Create and fit KMeans
            self.clusterer = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10
            )
            
            self.cluster_labels = self.clusterer.fit_predict(X_scaled)
        
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
        
        # Map features to clusters
        self._map_features_to_clusters()
        
        # Identify representative variable for each cluster
        self._identify_cluster_representatives(data, correlation_threshold)
        
        return self
    
    def _determine_optimal_clusters(
        self,
        X_scaled: np.ndarray,
        correlation_matrix: pd.DataFrame,
        max_clusters: int
    ) -> int:
        """
        Determine the optimal number of clusters.
        
        Args:
            X_scaled: Scaled feature matrix (transposed - features as rows)
            correlation_matrix: Correlation matrix of features
            max_clusters: Maximum number of clusters to consider
            
        Returns:
            Optimal number of clusters
        """
        # Limit max clusters based on data size
        max_clusters = min(max_clusters, X_scaled.shape[0] - 1)
        
        if self.method == 'hierarchical':
            # For hierarchical clustering, use the elbow method with inertia
            # First, calculate inertia for different cluster numbers using KMeans
            inertias = []
            for n in range(1, max_clusters + 1):
                kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                inertias.append(kmeans.inertia_)
            
            # Find elbow point (largest second derivative)
            inertias = np.array(inertias)
            second_derivative = np.diff(np.diff(inertias))
            elbow_point = np.argmax(second_derivative) + 2
            
            # Also consider feature correlation structure
            # More highly correlated features suggest fewer clusters
            avg_correlation = correlation_matrix.mean().mean()
            correlation_adjustment = int(avg_correlation * 3)  # Adjust clusters based on correlation
            
            # Balance elbow point and correlation structure
            optimal_clusters = max(2, min(elbow_point - correlation_adjustment, max_clusters))
            return optimal_clusters
            
        elif self.method == 'kmeans':
            # For KMeans, use silhouette score
            silhouette_scores = []
            for n in range(2, max_clusters + 1):  # Silhouette requires at least 2 clusters
                kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_scaled)
                score = silhouette_score(X_scaled, cluster_labels)
                silhouette_scores.append((n, score))
            
            # Find the number of clusters with highest silhouette score
            optimal_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
            return optimal_clusters
        
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
    
    def _map_features_to_clusters(self):
        """
        Map features to their assigned clusters.
        """
        self.feature_clusters = {}
        
        # Create a dictionary with cluster IDs as keys and lists of features as values
        for i, cluster_id in enumerate(self.cluster_labels):
            if cluster_id not in self.feature_clusters:
                self.feature_clusters[cluster_id] = []
            
            self.feature_clusters[cluster_id].append(self.feature_names[i])
    
    def _identify_cluster_representatives(
        self, 
        data: pd.DataFrame,
        correlation_threshold: float
    ):
        """
        Identify a representative variable for each cluster.
        
        Args:
            data: Original data with features
            correlation_threshold: Threshold for determining related features
        """
        for cluster_id, features in self.feature_clusters.items():
            if not features:
                continue
                
            if len(features) == 1:
                # If only one feature in cluster, it's the representative
                self.cluster_representatives[cluster_id] = features[0]
                continue
            
            # Calculate the sum of correlations for each feature with others in the cluster
            correlation_sums = {}
            cluster_data = data[features]
            correlation_matrix = cluster_data.corr().abs()
            
            for feature in features:
                # Sum of correlations with other features in the cluster
                correlation_sums[feature] = correlation_matrix[feature].sum() - 1  # Subtract self-correlation
            
            # Select feature with highest sum of correlations as representative
            representative = max(correlation_sums.items(), key=lambda x: x[1])[0]
            self.cluster_representatives[cluster_id] = representative
    
    def get_cluster_members(self) -> Dict[int, List[str]]:
        """
        Get members of each cluster.
        
        Returns:
            Dictionary mapping cluster IDs to lists of features
        """
        return self.feature_clusters
    
    def get_cluster_representatives(self) -> Dict[int, str]:
        """
        Get the representative variable for each cluster.
        
        Returns:
            Dictionary mapping cluster IDs to representative features
        """
        return self.cluster_representatives
    
    def get_feature_cluster(self, feature: str) -> Optional[int]:
        """
        Get the cluster ID for a specific feature.
        
        Args:
            feature: Name of feature
            
        Returns:
            Cluster ID or None if feature not found
        """
        for cluster_id, features in self.feature_clusters.items():
            if feature in features:
                return cluster_id
        return None
    
    def get_cluster_summary(self) -> pd.DataFrame:
        """
        Get a summary of clusters with sizes and representatives.
        
        Returns:
            DataFrame with cluster summary
        """
        summary_data = []
        
        for cluster_id, features in self.feature_clusters.items():
            representative = self.cluster_representatives.get(cluster_id, "None")
            summary_data.append({
                'cluster_id': cluster_id,
                'size': len(features),
                'representative': representative,
                'members': ", ".join(features)
            })
        
        return pd.DataFrame(summary_data)
    
    def predict_cluster(
        self, 
        features: List[str]
    ) -> Dict[str, int]:
        """
        Predict the cluster assignment for a list of features.
        
        Args:
            features: List of feature names
            
        Returns:
            Dictionary mapping features to predicted cluster IDs
        """
        results = {}
        
        for feature in features:
            # First check if feature is already in a known cluster
            cluster_id = self.get_feature_cluster(feature)
            if cluster_id is not None:
                results[feature] = cluster_id
                continue
            
            # Feature not in training data, use closest cluster representative
            closest_cluster = None
            max_correlation = -1
            
            for cluster_id, representative in self.cluster_representatives.items():
                if feature in self.feature_names and representative in self.feature_names:
                    # Calculate correlation between feature and representative
                    correlation = abs(pearsonr(self.feature_data[feature], self.feature_data[representative])[0])
                    
                    if correlation > max_correlation:
                        max_correlation = correlation
                        closest_cluster = cluster_id
            
            results[feature] = closest_cluster
        
        return results
    
    def predict_from_representatives(
        self, 
        data: pd.DataFrame,
        target_features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Predict values for all features using only the cluster representatives.
        
        Args:
            data: DataFrame with at least the representative features
            target_features: List of features to predict (None for all)
            
        Returns:
            DataFrame with predicted values for target features
        """
        if target_features is None:
            target_features = self.feature_names
        
        # Check if all representatives are in the data
        missing_reps = [rep for rep in set(self.cluster_representatives.values()) 
                        if rep not in data.columns]
        if missing_reps:
            raise ValueError(f"Missing representative features in data: {missing_reps}")
        
        # Create output DataFrame with same index as input
        result = pd.DataFrame(index=data.index)
        
        # For each feature, predict value based on its cluster representative
        for feature in target_features:
            # If feature is a representative or already in data, use it directly
            if feature in data.columns:
                result[feature] = data[feature]
                continue
            
            # Get the cluster for this feature
            cluster_id = self.get_feature_cluster(feature)
            if cluster_id is None:
                logger.warning(f"Feature {feature} not found in any cluster, skipping")
                continue
            
            # Get the representative for this cluster
            representative = self.cluster_representatives.get(cluster_id)
            if representative is None:
                logger.warning(f"No representative found for cluster {cluster_id}, skipping feature {feature}")
                continue
            
            # If we have regression coefficients, use them
            if hasattr(self, 'regression_models') and feature in self.regression_models:
                # Apply the regression model
                model = self.regression_models[feature]
                intercept, slope = model['intercept'], model['slope']
                result[feature] = intercept + slope * data[representative]
            else:
                # If no regression model, just copy the representative
                result[feature] = data[representative]
        
        return result
    
    def fit_regression_models(self, data: pd.DataFrame):
        """
        Fit simple regression models from representatives to other features.
        
        Args:
            data: DataFrame with all features
        """
        self.regression_models = {}
        self.feature_data = data  # Store for later use
        
        # For each cluster, fit regression models from representative to each member
        for cluster_id, features in self.feature_clusters.items():
            representative = self.cluster_representatives.get(cluster_id)
            if representative is None or representative not in data:
                continue
            
            # Get the representative data
            rep_data = data[representative].values.reshape(-1, 1)
            
            # Fit a simple linear regression for each feature in the cluster
            for feature in features:
                if feature == representative or feature not in data:
                    continue
                
                # Get feature data
                feature_data = data[feature].values
                
                # Handle missing values
                valid_indices = ~(np.isnan(rep_data) | np.isnan(feature_data))
                
                if np.sum(valid_indices) < 2:
                    logger.warning(f"Not enough valid data to fit regression for {feature}")
                    continue
                
                # Extract valid data points
                x_valid = rep_data[valid_indices]
                y_valid = feature_data[valid_indices]
                
                # Fit simple linear regression
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(x_valid, y_valid)
                
                # Store coefficients
                self.regression_models[feature] = {
                    'intercept': model.intercept_,
                    'slope': model.coef_[0],
                    'representative': representative
                }
    
    def plot_dendrogram(
        self,
        figsize: Tuple[int, int] = (12, 8),
        color_threshold: Optional[float] = None,
        truncate_mode: Optional[str] = None,
        p: Optional[int] = None
    ) -> plt.Figure:
        """
        Plot dendrogram for hierarchical clustering.
        
        Args:
            figsize: Figure size (width, height)
            color_threshold: Color threshold for dendrogram
            truncate_mode: How to truncate the dendrogram ('level' or 'lastp')
            p: Truncation parameter
            
        Returns:
            Matplotlib figure
        """
        if self.method != 'hierarchical' or self.linkage_matrix is None:
            raise ValueError("Dendrogram can only be plotted for hierarchical clustering")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot dendrogram
        dendrogram(
            self.linkage_matrix,
            labels=self.feature_names,
            color_threshold=color_threshold,
            truncate_mode=truncate_mode,
            p=p,
            ax=ax
        )
        
        # Rotate labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=90, ha='right')
        
        # Add labels and title
        ax.set_title('Hierarchical Clustering Dendrogram')
        ax.set_xlabel('Features')
        ax.set_ylabel('Distance')
        
        plt.tight_layout()
        return fig
    
    def plot_cluster_map(
        self,
        data: pd.DataFrame,
        figsize: Tuple[int, int] = (12, 10)
    ) -> plt.Figure:
        """
        Plot a heatmap of data with features grouped by cluster.
        
        Args:
            data: DataFrame with features
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get features ordered by cluster
        ordered_features = []
        for cluster_id in sorted(self.feature_clusters.keys()):
            # Add representative first
            rep = self.cluster_representatives.get(cluster_id)
            if rep in self.feature_clusters[cluster_id]:
                ordered_features.append(rep)
                
                # Add other features in cluster
                ordered_features.extend([f for f in self.feature_clusters[cluster_id] if f != rep])
            else:
                # No representative, add all features
                ordered_features.extend(self.feature_clusters[cluster_id])
        
        # Create heatmap of data with features organized by cluster
        if hasattr(data, 'values'):
            # For DataFrame input
            im = ax.imshow(data[ordered_features].values.T, aspect='auto', cmap='viridis')
        else:
            # For correlation matrix or other array input
            ordered_indices = [self.feature_names.index(f) for f in ordered_features]
            im = ax.imshow(data[ordered_indices, :][:, ordered_indices], aspect='auto', cmap='viridis')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        
        # Add feature names as y-tick labels
        ax.set_yticks(np.arange(len(ordered_features)))
        ax.set_yticklabels(ordered_features)
        
        # Add cluster boundaries
        y_pos = 0
        for cluster_id in sorted(self.feature_clusters.keys()):
            y_pos += len(self.feature_clusters[cluster_id])
            ax.axhline(y_pos - 0.5, color='red', linewidth=1)
        
        # Add labels and title
        ax.set_title('Feature Clusters Heatmap')
        ax.set_xlabel('Samples')
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_by_cluster(
        self,
        data: pd.DataFrame,
        figsize: Tuple[int, int] = (12, 10)
    ) -> plt.Figure:
        """
        Plot correlation matrix with features grouped by cluster.
        
        Args:
            data: DataFrame with features
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Calculate correlation matrix
        corr_matrix = data.corr().abs()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get features ordered by cluster
        ordered_features = []
        for cluster_id in sorted(self.feature_clusters.keys()):
            # Add representative first
            rep = self.cluster_representatives.get(cluster_id)
            if rep in self.feature_clusters[cluster_id]:
                ordered_features.append(rep)
                
                # Add other features in cluster
                ordered_features.extend([f for f in self.feature_clusters[cluster_id] if f != rep])
            else:
                # No representative, add all features
                ordered_features.extend(self.feature_clusters[cluster_id])
        
        # Create ordered correlation matrix
        ordered_corr = corr_matrix.loc[ordered_features, ordered_features]
        
        # Plot heatmap
        im = ax.imshow(ordered_corr, cmap='viridis', vmin=0, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Absolute Correlation')
        
        # Add tick labels
        ax.set_xticks(np.arange(len(ordered_features)))
        ax.set_yticks(np.arange(len(ordered_features)))
        ax.set_xticklabels(ordered_features, rotation=90)
        ax.set_yticklabels(ordered_features)
        
        # Add cluster boundaries
        y_pos = 0
        for cluster_id in sorted(self.feature_clusters.keys()):
            y_pos += len(self.feature_clusters[cluster_id])
            ax.axhline(y_pos - 0.5, color='red', linewidth=1)
            ax.axvline(y_pos - 0.5, color='red', linewidth=1)
        
        # Add title
        ax.set_title('Feature Correlation Matrix by Cluster')
        
        plt.tight_layout()
        return fig
    
    def save(self, filepath: str):
        """
        Save the feature clustering model to a file.
        
        Args:
            filepath: Path to save the model
        """
        import pickle
        
        # Create dictionary with all necessary attributes
        save_dict = {
            'method': self.method,
            'n_clusters': self.n_clusters,
            'scale_data': self.scale_data,
            'clusterer': self.clusterer,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'cluster_labels': self.cluster_labels,
            'cluster_representatives': self.cluster_representatives,
            'feature_clusters': self.feature_clusters,
            'linkage_matrix': self.linkage_matrix
        }
        
        # Add regression models if available
        if hasattr(self, 'regression_models'):
            save_dict['regression_models'] = self.regression_models
        
        # Save to file
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        
        logger.info(f"Feature clustering model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'FeatureClustering':
        """
        Load a feature clustering model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded FeatureClustering instance
        """
        import pickle
        
        # Load from file
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        # Create a new instance
        instance = cls(
            method=save_dict['method'],
            n_clusters=save_dict['n_clusters'],
            scale_data=save_dict['scale_data']
        )
        
        # Restore attributes
        instance.clusterer = save_dict['clusterer']
        instance.scaler = save_dict['scaler']
        instance.feature_names = save_dict['feature_names']
        instance.cluster_labels = save_dict['cluster_labels']
        instance.cluster_representatives = save_dict['cluster_representatives']
        instance.feature_clusters = save_dict['feature_clusters']
        instance.linkage_matrix = save_dict['linkage_matrix']
        
        # Restore regression models if available
        if 'regression_models' in save_dict:
            instance.regression_models = save_dict['regression_models']
        
        logger.info(f"Feature clustering model loaded from {filepath}")
        return instance


def cluster_gdp_related_features(
    data: pd.DataFrame,
    gdp_col: str,
    correlation_threshold: float = 0.4,
    min_clusters: int = 2,
    max_clusters: int = 10,
    method: str = 'hierarchical'
) -> Tuple[FeatureClustering, Dict[int, str], float]:
    """
    Cluster features based on their relationship with GDP.
    
    Args:
        data: DataFrame with economic indicators including GDP
        gdp_col: Name of GDP column
        correlation_threshold: Minimum absolute correlation with GDP to include a feature
        min_clusters: Minimum number of clusters
        max_clusters: Maximum number of clusters
        method: Clustering method ('hierarchical' or 'kmeans')
        
    Returns:
        Tuple of (clustering_model, gdp_correlated_representatives, avg_gdp_correlation)
    """
    if gdp_col not in data.columns:
        raise ValueError(f"GDP column '{gdp_col}' not found in data")
    
    # Calculate correlations with GDP
    gdp_correlations = {}
    for col in data.columns:
        if col != gdp_col:
            corr = data[[col, gdp_col]].corr().iloc[0, 1]
            gdp_correlations[col] = corr
    
    # Filter features by GDP correlation
    gdp_related_features = [col for col, corr in gdp_correlations.items() 
                           if abs(corr) >= correlation_threshold]
    
    if not gdp_related_features:
        raise ValueError(f"No features with GDP correlation above {correlation_threshold}")
    
    # Create subset with GDP-related features
    subset = data[gdp_related_features]
    
    # Determine optimal number of clusters
    n_clusters = max(min_clusters, min(len(gdp_related_features) // 3, max_clusters))
    
    # Create and fit clustering model
    clustering = FeatureClustering(method=method, n_clusters=n_clusters)
    clustering.fit(subset)
    
    # Get cluster representatives
    representatives = clustering.get_cluster_representatives()
    
    # Calculate average absolute GDP correlation of representatives
    rep_features = list(representatives.values())
    avg_gdp_correlation = np.mean([abs(gdp_correlations[feat]) for feat in rep_features])
    
    # Map cluster IDs to their representative's GDP correlation
    gdp_correlated_representatives = {
        cluster_id: f"{rep} (GDP r={gdp_correlations[rep]:.2f})"
        for cluster_id, rep in representatives.items()
    }
    
    return clustering, gdp_correlated_representatives, avg_gdp_correlation