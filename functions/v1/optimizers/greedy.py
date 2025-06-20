import numpy as np
import torch
from typing import Dict, List, Tuple
from .base import BaseOptimizer
from ..system import Terrain, Well
from scipy.optimize import minimize
import wandb

class GreedyOptimizer(BaseOptimizer):
    """
    A greedy optimization algorithm that places wells at locations with the highest error.
    This algorithm prioritizes immediate error reduction over long-term optimization.
    It iteratively places wells at locations with the highest discrepancy between current and goal terrain.
    """
    
    def __init__(self, 
                 terrainSize: int,
                 maxIterations: int,
                 depthBounds: Tuple[float, float],
                 volumeBounds: Tuple[float, float],
                 monetaryLimit: float,
                 timeLimit: float,
                 fidelity: float,
                 seed: int = None,
                 algorithm=None,
                 progress_callback=None):
        super().__init__(
            terrainSize=terrainSize,
            maxIterations=maxIterations,
            depthBounds=depthBounds,
            volumeBounds=volumeBounds,
            monetaryLimit=monetaryLimit,
            timeLimit=timeLimit,
            fidelity=fidelity,
            seed=seed,
            algorithm=algorithm,
            progress_callback=progress_callback
        )
        self.algorithm = algorithm
        self.progress_callback = progress_callback
    
    def optimize(self) -> Dict:
        """
        Run the greedy optimization algorithm.
        
        Returns:
            Dict containing optimization results
        """
        try:
            # Get initial and goal terrains
            initial_terrain = self.terrain.initial_terrain
            goal_terrain = self.terrain.goal_terrain
            current_terrain = initial_terrain.clone()
            
            wells = []
            iteration = 0
            monetaryCost = 0.0
            timeCost = 0.0
            
            while iteration < self.maxIterations:
                # Calculate error between current and target terrain
                discrepancy = goal_terrain - current_terrain
                error = torch.abs(discrepancy)
                
                # Find location with maximum error
                max_error_idx = torch.argmax(error)
                max_error_pos = (max_error_idx // self.terrainSize, max_error_idx % self.terrainSize)
                
                # Create well at the position with maximum error
                well = Well(
                    x0=max_error_pos[0],
                    y0=max_error_pos[1],
                    depth=0,  # Will be optimized
                    volume=0  # Will be optimized
                )
                
                # Optimize well parameters
                self._optimize_well_parameters([well], current_terrain)
            
                # Add well to list
                wells.append(well)
                
                # Update costs
                monetaryCost += well.monetaryCost()
                timeCost += well.time_cost()
                
                # Update terrain
                current_terrain = self.terrain.apply_wells(wells)
                
                # Calculate metrics
                mse = float(torch.mean((goal_terrain - current_terrain) ** 2))
                fidelity = 1.0 - mse
                
                # Update metrics
                self.update_metrics(
                    iteration=iteration,
                    wellsPlaced=len(wells),
                    mse=mse,
                    monetaryCost=monetaryCost,
                    timeCost=timeCost,
                    fidelity=fidelity,
                    wells=wells
                )
                
                if self.progress_callback:
                    self.progress_callback(self.get_metrics())
                
                # Check if we've reached target fidelity
                if fidelity >= self.fidelity:
                    break
                    
                iteration += 1
            
            # Convert wells to dictionary format
            wells_dict = [
                {
                    'x': int(well.x0),
                    'y': int(well.y0),
                    'depth': float(well.depth),
                    'volume': float(well.volume)
                }
                for well in wells
            ]
            
           
            return {
                'wells': wells_dict,
                'metrics': self.get_metrics(),
                'terrain_summary': self.get_summary()
            }
        except Exception as e:
            print(f"Error in greedy optimization: {str(e)}")
            raise
    
    def _optimize_well_parameters(self, wells: List[Well], current_terrain: torch.Tensor):
        """Optimize well parameters using scipy's minimize."""
        initial_params = np.array([param for well in wells for param in (well.depth, well.volume)])
        bounds = [(self.depthBounds[0], self.depthBounds[1]), 
                 (self.volumeBounds[0], self.volumeBounds[1])] * len(wells)
        
        def objective(params):
            for i, well in enumerate(wells):
                well.depth, well.volume = params[i*2], params[i*2+1]
            modified_terrain = self.terrain.apply_wells(wells)
            discrepancy = modified_terrain - self.terrain.goal_terrain
            
            # Define weights for the loss function
            overshoot_weight = 20  # Higher weight for overshooting
            undershoot_weight = 1  # Lower weight for undershooting
            
            # Compute the asymmetric loss
            loss = torch.where(discrepancy > 0,
                             overshoot_weight * discrepancy ** 2,
                             undershoot_weight * discrepancy ** 2)
            return torch.mean(loss).item()
        
        result = minimize(
            objective, 
            initial_params, 
            method='Nelder-Mead', 
            bounds=bounds, 
            options={'maxiter': 10000, 'disp': False}
        )
        
        # Update well parameters with optimized values
        for i, well in enumerate(wells):
            well.depth, well.volume = result.x[i*2], result.x[i*2+1] 