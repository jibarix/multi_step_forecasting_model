"""
Hierarchical forecasting module for economic forecasting.

This module implements hierarchical forecasting capabilities, allowing
for consistent forecasts across different levels of aggregation such as
total → regions → stores. It includes various reconciliation methods
to ensure consistency across the hierarchy.
"""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import numpy as np
import pandas as pd
from collections import defaultdict

# Set up logging
logger = logging.getLogger(__name__)

class HierarchicalForecaster:
    """
    Class for hierarchical forecasting across different levels of aggregation.
    
    This enables forecasting at multiple levels of a hierarchy (e.g., total sales,
    regional sales, store sales) while ensuring consistency across levels through
    various reconciliation methods.
    """
    
    def __init__(self, reconciliation_method: str = 'bottom_up'):
        """
        Initialize the hierarchical forecaster.
        
        Args:
            reconciliation_method: Method for reconciling forecasts
                ('bottom_up', 'top_down', 'middle_out', 'optimal')
        """
        self.reconciliation_method = reconciliation_method
        self.hierarchy = {}  # Dictionary to store hierarchical structure
        self.models = {}     # Dictionary to store models for each node
        self.base_forecasts = {}  # Dictionary to store forecasts before reconciliation
        self.reconciled_forecasts = {}  # Dictionary to store reconciled forecasts
        self.node_data = {}  # Dictionary to store data for each node
        
    def set_hierarchy(self, hierarchy_dict: Dict[str, List[str]]) -> None:
        """
        Set the hierarchical structure.
        
        Args:
            hierarchy_dict: Dictionary defining the hierarchical structure
                Example: {'total': ['region1', 'region2'], 'region1': ['store1', 'store2']}
        """
        self.hierarchy = hierarchy_dict
        self._validate_hierarchy()
        logger.info(f"Hierarchy set with {len(hierarchy_dict)} parent nodes")
        
    def _validate_hierarchy(self) -> None:
        """
        Validate the hierarchical structure to ensure it's a valid tree.
        
        Raises:
            ValueError: If the hierarchy is invalid (e.g., contains cycles)
        """
        # Check for cycles
        visited = set()
        to_visit = list(self.hierarchy.keys())
        
        while to_visit:
            current = to_visit.pop(0)
            if current in visited:
                raise ValueError(f"Cyclic dependency detected in hierarchy at node {current}")
            visited.add(current)
            
            # Add children to visit list
            children = self.hierarchy.get(current, [])
            to_visit.extend([child for child in children if child not in visited])
            
        # Check if all children have a parent or are root nodes
        all_nodes = set(self.hierarchy.keys())
        all_children = set()
        
        for children in self.hierarchy.values():
            all_children.update(children)
            
        orphaned_nodes = all_children - all_nodes
        if orphaned_nodes:
            logger.warning(f"Orphaned nodes found in hierarchy: {orphaned_nodes}")
        
        # Find and store root nodes
        self.root_nodes = all_nodes - all_children
        if not self.root_nodes:
            raise ValueError("No root nodes found in hierarchy")
            
        logger.info(f"Hierarchy validation passed. Root nodes: {self.root_nodes}")
    
    def add_data(self, node_id: str, data: pd.DataFrame, date_col: str = 'date', value_col: str = None) -> None:
        """
        Add data for a specific node in the hierarchy.
        
        Args:
            node_id: Identifier for the node
            data: DataFrame with time series data
            date_col: Name of date column
            value_col: Name of value column (if None, use node_id)
        """
        # Make a copy of the data to avoid modifying the original
        node_data = data.copy()
        
        # If value_col is not specified, use node_id
        if value_col is None:
            value_col = node_id
            
        # Ensure date column is datetime
        if date_col in node_data.columns:
            node_data[date_col] = pd.to_datetime(node_data[date_col])
            
        # Store the data
        self.node_data[node_id] = {
            'data': node_data,
            'date_col': date_col,
            'value_col': value_col
        }
        
        logger.info(f"Added data for node {node_id} with {len(node_data)} records")
    
    def add_model(self, node_id: str, model: Any) -> None:
        """
        Add a forecasting model for a specific node.
        
        Args:
            node_id: Identifier for the node
            model: Model instance for the node
        """
        self.models[node_id] = model
        logger.info(f"Added model for node {node_id}")
    
    def fit_models(
        self, 
        model_factory_func: Callable,
        default_model_type: str = 'auto',
        **model_params
    ) -> None:
        """
        Fit models for all nodes in the hierarchy using a model factory function.
        
        Args:
            model_factory_func: Function to create and fit models
            default_model_type: Default model type to use
            **model_params: Additional parameters for model fitting
        """
        for node_id, node_info in self.node_data.items():
            logger.info(f"Fitting model for node {node_id}")
            
            try:
                # Get data for this node
                data = node_info['data']
                value_col = node_info['value_col']
                
                # Create and fit model
                model = model_factory_func(
                    data=data, 
                    target_col=value_col, 
                    model_type=default_model_type,
                    **model_params
                )
                
                # Store model
                self.add_model(node_id, model)
                
            except Exception as e:
                logger.error(f"Error fitting model for node {node_id}: {e}")
    
    def generate_base_forecasts(self, horizon: int, future_covariates: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, pd.DataFrame]:
        """
        Generate base forecasts for each node before reconciliation.
        
        Args:
            horizon: Forecast horizon (number of periods)
            future_covariates: Dictionary of future covariates for each node
            
        Returns:
            Dictionary of base forecasts for each node
        """
        self.base_forecasts = {}
        
        for node_id, model in self.models.items():
            logger.info(f"Generating base forecast for node {node_id}")
            
            try:
                # Get node data info
                node_info = self.node_data.get(node_id, {})
                date_col = node_info.get('date_col', 'date')
                
                # Get future covariates for this node if available
                node_covariates = None
                if future_covariates and node_id in future_covariates:
                    node_covariates = future_covariates[node_id]
                
                # Generate forecast
                forecast = self._forecast_node(node_id, model, horizon, node_covariates)
                
                # Store forecast
                self.base_forecasts[node_id] = forecast
                
            except Exception as e:
                logger.error(f"Error generating forecast for node {node_id}: {e}")
        
        return self.base_forecasts
    
    def _forecast_node(
        self, 
        node_id: str, 
        model: Any, 
        horizon: int, 
        future_covariates: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate forecast for a specific node using its model.
        
        Args:
            node_id: Identifier for the node
            model: Model instance for the node
            horizon: Forecast horizon
            future_covariates: Future covariates for this node
            
        Returns:
            DataFrame with forecast
        """
        # Get node data info
        node_info = self.node_data.get(node_id, {})
        date_col = node_info.get('date_col', 'date')
        value_col = node_info.get('value_col', node_id)
        
        # Check model type and generate forecast accordingly
        if hasattr(model, 'predict') and callable(getattr(model, 'predict')):
            # For scikit-learn style models
            if future_covariates is not None:
                forecast = model.predict(future_covariates)
            else:
                # Create dummy X data if necessary
                last_date = node_info['data'][date_col].max()
                freq = pd.infer_freq(node_info['data'][date_col])
                if freq is None:
                    # Try to estimate frequency
                    date_diffs = np.diff(node_info['data'][date_col].astype(np.int64))
                    avg_diff = np.mean(date_diffs)
                    if avg_diff < 100 * 86400 * 1e9:  # Less than ~3 months in nanoseconds
                        freq = 'MS'  # Month start
                    else:
                        freq = 'QS'  # Quarter start
                
                future_dates = pd.date_range(start=last_date, periods=horizon+1, freq=freq)[1:]
                
                # Simple forecast (this would need more sophistication in practice)
                forecast = model.predict(np.arange(horizon).reshape(-1, 1))
                
                # Create forecast DataFrame
                forecast_df = pd.DataFrame({
                    date_col: future_dates,
                    'forecast': forecast
                })
                
                return forecast_df
                
        elif hasattr(model, 'forecast') and callable(getattr(model, 'forecast')):
            # For time series models
            forecast = model.forecast(steps=horizon)
            
            # Convert to DataFrame if needed
            if isinstance(forecast, pd.DataFrame):
                return forecast
            else:
                # Create forecast DataFrame
                last_date = node_info['data'][date_col].max()
                freq = pd.infer_freq(node_info['data'][date_col])
                if freq is None:
                    freq = 'MS'  # Default to month start
                    
                future_dates = pd.date_range(start=last_date, periods=horizon+1, freq=freq)[1:]
                
                forecast_df = pd.DataFrame({
                    date_col: future_dates,
                    'forecast': forecast
                })
                
                return forecast_df
        
        # If we got here, we don't know how to generate forecasts for this model
        raise ValueError(f"Don't know how to generate forecasts for model type {type(model)}")
    
    def reconcile_forecasts(self) -> Dict[str, pd.DataFrame]:
        """
        Reconcile forecasts to ensure consistency across the hierarchy.
        
        Returns:
            Dictionary of reconciled forecasts for each node
        """
        if not self.base_forecasts:
            raise ValueError("No base forecasts available for reconciliation")
            
        if self.reconciliation_method == 'bottom_up':
            self.reconciled_forecasts = self._reconcile_bottom_up()
        elif self.reconciliation_method == 'top_down':
            self.reconciled_forecasts = self._reconcile_top_down()
        elif self.reconciliation_method == 'middle_out':
            self.reconciled_forecasts = self._reconcile_middle_out()
        elif self.reconciliation_method == 'optimal':
            self.reconciled_forecasts = self._reconcile_optimal()
        else:
            raise ValueError(f"Unknown reconciliation method: {self.reconciliation_method}")
            
        logger.info(f"Forecasts reconciled using {self.reconciliation_method} method")
        return self.reconciled_forecasts
    
    def _reconcile_bottom_up(self) -> Dict[str, pd.DataFrame]:
        """
        Implement bottom-up reconciliation.
        
        In bottom-up reconciliation, forecasts at higher levels are created 
        by aggregating forecasts from the lowest level.
        
        Returns:
            Dictionary of reconciled forecasts for each node
        """
        # Make a copy of base forecasts
        reconciled = {k: v.copy() for k, v in self.base_forecasts.items()}
        
        # Identify bottom level nodes (leaves)
        leaves = self._get_leaf_nodes()
        logger.debug(f"Identified leaf nodes: {leaves}")
        
        # Process nodes in reverse topological order (bottom to top)
        for node_id in self._get_reverse_topological_order():
            # Skip leaf nodes
            if node_id in leaves:
                continue
                
            # Get children of this node
            children = self.hierarchy.get(node_id, [])
            
            # Skip if no children or no children with forecasts
            valid_children = [c for c in children if c in reconciled]
            if not valid_children:
                continue
            
            # Aggregate children's forecasts
            child_dfs = [reconciled[c] for c in valid_children]
            
            # Ensure all children have same date column
            date_cols = set(df.columns[0] for df in child_dfs)  # Assume first column is date
            if len(date_cols) > 1:
                logger.warning(f"Multiple date columns found in children of {node_id}: {date_cols}")
                
            date_col = list(date_cols)[0]
            
            # Create aggregate by summing forecasts from children
            # First, create an empty DataFrame with dates
            all_dates = pd.concat([df[date_col] for df in child_dfs]).drop_duplicates().sort_values()
            
            node_forecast = pd.DataFrame({date_col: all_dates})
            node_forecast['forecast'] = 0
            
            # Sum forecasts from children
            for child_df in child_dfs:
                # Merge with the node forecast
                merged = pd.merge(node_forecast, child_df, on=date_col, how='left')
                # Add child forecast to the total
                node_forecast['forecast'] += merged['forecast'].fillna(0)
            
            # Store the aggregated forecast
            reconciled[node_id] = node_forecast
            
        return reconciled
    
    def _reconcile_top_down(self) -> Dict[str, pd.DataFrame]:
        """
        Implement top-down reconciliation.
        
        In top-down reconciliation, forecasts at lower levels are created 
        by distributing forecasts from the highest level based on historical proportions.
        
        Returns:
            Dictionary of reconciled forecasts for each node
        """
        # Make a copy of base forecasts
        reconciled = {k: v.copy() for k, v in self.base_forecasts.items()}
        
        # Identify root nodes
        root_nodes = self._get_root_nodes()
        if not root_nodes:
            raise ValueError("No root nodes found in hierarchy")
        
        # If multiple roots, we can't do simple top-down
        if len(root_nodes) > 1:
            logger.warning(
                f"Multiple root nodes found: {root_nodes}. Using first one: {list(root_nodes)[0]}"
            )
            
        root_node = list(root_nodes)[0]
        
        # Calculate historical proportions
        proportions = self._calculate_historical_proportions()
        
        # Process nodes in topological order (top to bottom)
        for node_id in self._get_topological_order():
            # Skip the root node
            if node_id == root_node:
                continue
                
            # Find parent of this node
            parent = self._find_parent(node_id)
            if not parent or parent not in reconciled:
                logger.warning(f"Parent not found for node {node_id} or no forecast available. Skipping.")
                continue
                
            # Get parent forecast
            parent_forecast = reconciled[parent]
            
            # Get proportion for this node
            proportion = proportions.get((parent, node_id), 1.0)
            
            # Create forecast for this node by applying proportion to parent
            node_forecast = parent_forecast.copy()
            node_forecast['forecast'] = node_forecast['forecast'] * proportion
            
            # Store the result
            reconciled[node_id] = node_forecast
            
        return reconciled
    
    def _reconcile_middle_out(self) -> Dict[str, pd.DataFrame]:
        """
        Implement middle-out reconciliation.
        
        In middle-out reconciliation, forecasts are generated at a middle level,
        then propagated up using bottom-up and down using top-down approaches.
        
        Returns:
            Dictionary of reconciled forecasts for each node
        """
        # This is a simplified implementation that combines bottom-up and top-down
        # In a full implementation, you would identify middle-level nodes
        
        # Make a copy of base forecasts
        reconciled = {k: v.copy() for k, v in self.base_forecasts.items()}
        
        # Get middle level nodes (simplified - choose nodes with both parents and children)
        all_nodes = set(reconciled.keys())
        middle_nodes = set()
        
        for node_id in all_nodes:
            # Check if node has both a parent and children
            has_parent = self._find_parent(node_id) is not None
            has_children = bool(self.hierarchy.get(node_id, []))
            
            if has_parent and has_children:
                middle_nodes.add(node_id)
                
        if not middle_nodes:
            logger.warning("No suitable middle nodes found. Falling back to bottom-up.")
            return self._reconcile_bottom_up()
            
        logger.info(f"Using middle-out reconciliation with middle nodes: {middle_nodes}")
        
        # Process bottom-up from middle nodes to the top
        # First, make all nodes below middle nodes consistent with middle nodes
        
        # Identify nodes below middle nodes
        below_middle = set()
        for middle_node in middle_nodes:
            below_middle.update(self._get_descendants(middle_node))
            
        # Apply top-down reconciliation to middle nodes and below
        proportions = self._calculate_historical_proportions()
        for node_id in self._get_topological_order():
            if node_id in middle_nodes or node_id not in below_middle:
                continue
                
            # Find parent of this node
            parent = self._find_parent(node_id)
            if not parent or parent not in reconciled:
                continue
                
            # Get parent forecast
            parent_forecast = reconciled[parent]
            
            # Get proportion for this node
            proportion = proportions.get((parent, node_id), 1.0)
            
            # Create forecast for this node by applying proportion to parent
            node_forecast = parent_forecast.copy()
            node_forecast['forecast'] = node_forecast['forecast'] * proportion
            
            # Store the result
            reconciled[node_id] = node_forecast
        
        # Process bottom-up from bottom to middle nodes
        for middle_node in middle_nodes:
            # Skip if node has no children or no forecast
            if middle_node not in reconciled or middle_node not in self.hierarchy:
                continue
                
            # Get direct children
            children = self.hierarchy[middle_node]
            
            # Skip if no children with forecasts
            valid_children = [c for c in children if c in reconciled]
            if not valid_children:
                continue
            
            # For simplicity, we don't modify the middle node forecast
            # In a full implementation, you might reconcile based on bottom-up aggregation
            
        # Process bottom-up from middle nodes to the top
        for node_id in self._get_reverse_topological_order():
            # Skip if node is a middle node or below
            if node_id in middle_nodes or node_id in below_middle:
                continue
                
            # Skip if node has no children or no forecast
            if node_id not in self.hierarchy:
                continue
                
            # Get children of this node
            children = self.hierarchy[node_id]
            
            # Skip if no children with forecasts
            valid_children = [c for c in children if c in reconciled]
            if not valid_children:
                continue
            
            # Aggregate children's forecasts
            child_dfs = [reconciled[c] for c in valid_children]
            
            # Ensure all children have same date column
            date_cols = set(df.columns[0] for df in child_dfs)  # Assume first column is date
            date_col = list(date_cols)[0]
            
            # Create aggregate by summing forecasts from children
            all_dates = pd.concat([df[date_col] for df in child_dfs]).drop_duplicates().sort_values()
            
            node_forecast = pd.DataFrame({date_col: all_dates})
            node_forecast['forecast'] = 0
            
            # Sum forecasts from children
            for child_df in child_dfs:
                merged = pd.merge(node_forecast, child_df, on=date_col, how='left')
                node_forecast['forecast'] += merged['forecast'].fillna(0)
            
            # Store the aggregated forecast
            reconciled[node_id] = node_forecast
            
        return reconciled
    
    def _reconcile_optimal(self) -> Dict[str, pd.DataFrame]:
        """
        Implement optimal reconciliation using the MinT approach.
        
        This is a simplified implementation of optimal reconciliation.
        A full implementation would use the MinT approach or a similar method.
        
        Returns:
            Dictionary of reconciled forecasts for each node
        """
        # For now, we'll use a simple weighted average of bottom-up and top-down
        bottom_up = self._reconcile_bottom_up()
        top_down = self._reconcile_top_down()
        
        # Combine with equal weights
        reconciled = {}
        for key in self.base_forecasts:
            if key in bottom_up and key in top_down:
                # Create a copy of the structure
                reconciled[key] = bottom_up[key].copy()
                # Average the forecasts
                reconciled[key]['forecast'] = 0.5 * bottom_up[key]['forecast'] + 0.5 * top_down[key]['forecast']
            elif key in bottom_up:
                reconciled[key] = bottom_up[key]
            elif key in top_down:
                reconciled[key] = top_down[key]
        
        return reconciled
    
    def _get_leaf_nodes(self) -> set:
        """
        Get leaf nodes (nodes without children) in the hierarchy.
        
        Returns:
            Set of leaf node IDs
        """
        all_nodes = set(self.hierarchy.keys())
        all_children = set()
        
        for children in self.hierarchy.values():
            all_children.update(children)
            
        # Nodes that are in all_children but not in all_nodes are childless
        return all_children - all_nodes
    
    def _get_root_nodes(self) -> set:
        """
        Get root nodes (nodes without parents) in the hierarchy.
        
        Returns:
            Set of root node IDs
        """
        all_nodes = set(self.hierarchy.keys())
        all_children = set()
        
        for children in self.hierarchy.values():
            all_children.update(children)
            
        # Nodes that are in all_nodes but not in all_children are parentless
        return all_nodes - all_children
    
    def _find_parent(self, node_id: str) -> Optional[str]:
        """
        Find the parent of a node.
        
        Args:
            node_id: ID of the node
            
        Returns:
            Parent node ID or None if not found
        """
        for parent, children in self.hierarchy.items():
            if node_id in children:
                return parent
        return None
    
    def _get_descendants(self, node_id: str) -> set:
        """
        Get all descendants of a node (children, grandchildren, etc.).
        
        Args:
            node_id: ID of the node
            
        Returns:
            Set of descendant node IDs
        """
        if node_id not in self.hierarchy:
            return set()
            
        descendants = set(self.hierarchy[node_id])
        
        for child in self.hierarchy[node_id]:
            descendants.update(self._get_descendants(child))
            
        return descendants
    
    def _get_topological_order(self) -> List[str]:
        """
        Get nodes in topological order (parents before children).
        
        Returns:
            List of node IDs in topological order
        """
        # Implementation using Kahn's algorithm for topological sort
        # 1. Find nodes with no incoming edges (roots)
        # 2. Add them to the result list and remove them from the graph
        # 3. Repeat until all nodes are processed
        
        # Create a copy of the hierarchy
        hierarchy_copy = {k: list(v) for k, v in self.hierarchy.items()}
        
        # Create a dictionary of in-degrees (number of parents)
        in_degree = defaultdict(int)
        
        # Count in-degrees
        for parent, children in self.hierarchy.items():
            for child in children:
                in_degree[child] += 1
                
        # Find nodes with no incoming edges (roots)
        roots = [node for node in self.hierarchy if in_degree[node] == 0]
        
        # Initialize result list
        result = []
        
        # Process roots
        while roots:
            # Get a root and add to result
            current = roots.pop(0)
            result.append(current)
            
            # Process children
            if current in hierarchy_copy:
                for child in hierarchy_copy[current]:
                    # Decrement in-degree of child
                    in_degree[child] -= 1
                    
                    # If child has no more parents, add to roots
                    if in_degree[child] == 0:
                        roots.append(child)
                        
                # Remove current node from hierarchy
                del hierarchy_copy[current]
        
        # Check if there are unprocessed nodes (indicates a cycle)
        if any(hierarchy_copy.values()):
            logger.warning("Cycle detected in hierarchy during topological sort")
            
        return result
    
    def _get_reverse_topological_order(self) -> List[str]:
        """
        Get nodes in reverse topological order (children before parents).
        
        Returns:
            List of node IDs in reverse topological order
        """
        # Simply reverse the topological order
        return list(reversed(self._get_topological_order()))
    
    def _calculate_historical_proportions(self) -> Dict[Tuple[str, str], float]:
        """
        Calculate historical proportions between parent and child nodes.
        
        Returns:
            Dictionary mapping (parent, child) tuples to proportions
        """
        proportions = {}
        
        # Process all parent-child relationships
        for parent, children in self.hierarchy.items():
            # Skip if parent has no data
            if parent not in self.node_data:
                continue
                
            parent_info = self.node_data[parent]
            parent_data = parent_info['data']
            parent_value_col = parent_info['value_col']
            parent_date_col = parent_info['date_col']
            
            # Calculate total value for parent
            parent_total = parent_data[parent_value_col].sum()
            
            if parent_total == 0:
                # If parent total is zero, assign equal proportions
                for child in children:
                    proportions[(parent, child)] = 1.0 / len(children)
                continue
                
            # Calculate proportion for each child
            for child in children:
                # Skip if child has no data
                if child not in self.node_data:
                    continue
                    
                child_info = self.node_data[child]
                child_data = child_info['data']
                child_value_col = child_info['value_col']
                
                # Calculate total value for child
                child_total = child_data[child_value_col].sum()
                
                # Calculate proportion
                proportion = child_total / parent_total
                
                # Store proportion
                proportions[(parent, child)] = proportion
        
        return proportions
    
    def evaluate_reconciliation(self, actuals: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate the quality of the reconciliation.
        
        Args:
            actuals: Dictionary mapping node IDs to DataFrames with actual values
            
        Returns:
            Dictionary with evaluation metrics for each node
        """
        if not self.reconciled_forecasts:
            raise ValueError("No reconciled forecasts available for evaluation")
            
        results = {}
        
        for node_id, forecast_df in self.reconciled_forecasts.items():
            # Skip if no actuals for this node
            if node_id not in actuals:
                continue
                
            # Get actuals for this node
            actual_df = actuals[node_id]
            
            # Get date and value columns
            date_col = forecast_df.columns[0]  # Assume first column is date
            value_col = actual_df.columns[1] if len(actual_df.columns) > 1 else actual_df.columns[0]
            
            # Merge forecast and actuals
            merged = pd.merge(
                forecast_df,
                actual_df[[date_col, value_col]],
                on=date_col,
                how='inner'
            )
            
            # Skip if no overlapping dates
            if len(merged) == 0:
                continue
                
            # Calculate metrics
            mse = ((merged['forecast'] - merged[value_col])**2).mean()
            rmse = np.sqrt(mse)
            mae = (merged['forecast'] - merged[value_col]).abs().mean()
            mape = ((merged['forecast'] - merged[value_col]).abs() / merged[value_col].abs()).mean() * 100
            
            # Store results
            results[node_id] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'mape': mape
            }
        
        return results
    
    def plot_hierarchical_forecast(
        self, 
        nodes: Optional[List[str]] = None,
        actuals: Optional[Dict[str, pd.DataFrame]] = None,
        figsize: Tuple[int, int] = (12, 8)
    ):
        """
        Plot hierarchical forecasts for specified nodes.
        
        Args:
            nodes: List of node IDs to plot (None for all nodes)
            actuals: Optional dictionary with actual values
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        import matplotlib.pyplot as plt
        
        if not self.reconciled_forecasts:
            raise ValueError("No reconciled forecasts available for plotting")
            
        if nodes is None:
            nodes = list(self.reconciled_forecasts.keys())
            
        # Calculate number of rows and columns for subplots
        n_plots = len(nodes)
        cols = min(3, n_plots)
        rows = (n_plots + cols - 1) // cols
        
        # Create figure and subplots
        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
        
        for i, node_id in enumerate(nodes):
            # Get row and column for subplot
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            
            # Skip if no forecast for this node
            if node_id not in self.reconciled_forecasts:
                ax.text(0.5, 0.5, f"No forecast for {node_id}", ha='center', va='center')
                continue
                
            # Get forecast for this node
            forecast_df = self.reconciled_forecasts[node_id]
            
            # Get date column (assume first column is date)
            date_col = forecast_df.columns[0]
            
            # Plot forecast
            ax.plot(forecast_df[date_col], forecast_df['forecast'], 'b-', label='Forecast')
            
            # Plot actuals if available
            if actuals and node_id in actuals:
                actual_df = actuals[node_id]
                
                # Get value column (assume second column if available, otherwise first)
                value_col = actual_df.columns[1] if len(actual_df.columns) > 1 else actual_df.columns[0]
                
                # Plot actuals
                ax.plot(actual_df[date_col], actual_df[value_col], 'r-', label='Actual')
                
            # Add title and legend
            ax.set_title(f"Node: {node_id}")
            ax.legend()
            
            # Rotate date labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Hide unused subplots
        for i in range(len(nodes), rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].set_visible(False)
            
        plt.tight_layout()
        return fig
    
    def check_coherence(self, tolerance: float = 1e-8) -> Dict[str, float]:
        """
        Check coherence of reconciled forecasts for each parent-child relationship.
        
        This verifies that the sum of forecasts from children equals the 
        forecast for their parent within the specified tolerance.
        
        Args:
            tolerance: Tolerance for floating-point comparison
            
        Returns:
            Dictionary with coherence error for each parent node
        """
        if not self.reconciled_forecasts:
            raise ValueError("No reconciled forecasts available for coherence check")
            
        coherence_errors = {}
        
        for parent, children in self.hierarchy.items():
            # Skip if no forecast for parent
            if parent not in self.reconciled_forecasts:
                continue
                
            # Get parent forecast
            parent_forecast = self.reconciled_forecasts[parent]
            date_col = parent_forecast.columns[0]  # Assume first column is date
            
            # Get forecasts for valid children
            valid_children = [c for c in children if c in self.reconciled_forecasts]
            
            if not valid_children:
                continue
                
            # Initialize aggregated forecast
            all_dates = parent_forecast[date_col].copy()
            children_sum = pd.DataFrame({date_col: all_dates})
            children_sum['forecast'] = 0.0
            
            # Sum forecasts from children
            for child in valid_children:
                child_forecast = self.reconciled_forecasts[child]
                
                # Merge with children sum
                merged = pd.merge(children_sum, child_forecast, on=date_col, how='left')
                
                # Add child forecast to the sum
                children_sum['forecast'] += merged['forecast'].fillna(0.0)
            
            # Merge parent and children sum
            merged = pd.merge(parent_forecast, children_sum, on=date_col, suffixes=('_parent', '_children'), how='left')
            
            # Calculate mean absolute difference
            diff = (merged['forecast_parent'] - merged['forecast_children']).abs()
            mean_diff = diff.mean()
            
            # Store result
            coherence_errors[parent] = mean_diff
            
            # Log warning if error exceeds tolerance
            if mean_diff > tolerance:
                logger.warning(f"Coherence error for node {parent}: {mean_diff:.6f} exceeds tolerance {tolerance:.6f}")
        
        return coherence_errors