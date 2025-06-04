from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch
import wandb
from ..system import Terrain, Well

class BaseOptimizer(ABC):
    """Base class for all terrain optimization algorithms."""
    
    def __init__(
        self,
        terrain_size: int,
        noise: float,
        smoothness: float,
        max_iterations: int,
        depth_bounds: Tuple[float, float],
        volume_bounds: Tuple[float, float],
        monetary_limit: float,
        time_limit: float,
        target_fidelity: float,
        seed: int = 42,
        initial_terrain_params: Dict = None,
        target_terrain_params: Dict = None
    ):
        """
        Initialize the optimizer with simulation parameters.
        
        Args:
            terrain_size: Size of the terrain grid
            noise: Level of noise in terrain generation
            smoothness: Smoothness factor for terrain
            max_iterations: Maximum number of iterations allowed
            depth_bounds: Tuple of (min_depth, max_depth) for wells
            volume_bounds: Tuple of (min_volume, max_volume) for wells
            monetary_limit: Maximum monetary cost allowed
            time_limit: Maximum time cost allowed
            target_fidelity: Target fidelity level to achieve
            seed: Random seed for reproducibility
            initial_terrain_params: Initial terrain parameters
            target_terrain_params: Target terrain parameters
        """
        # Initialize wandb
        wandb.init(
            project="terrain-simulation",
            name=f"{self.__class__.__name__}-{terrain_size}",
            config={
                "terrain_size": terrain_size,
                "noise": noise,
                "smoothness": smoothness,
                "max_iterations": max_iterations,
                "depth_bounds": depth_bounds,
                "volume_bounds": volume_bounds,
                "monetary_limit": monetary_limit,
                "time_limit": time_limit,
                "target_fidelity": target_fidelity,
                "seed": seed
            }
        )
        
        # Rest of initialization
        self.terrain_size = terrain_size
        self.noise = noise
        self.smoothness = smoothness
        self.max_iterations = max_iterations
        self.depth_bounds = depth_bounds
        self.volume_bounds = volume_bounds
        self.monetary_limit = monetary_limit
        self.time_limit = time_limit
        self.target_fidelity = target_fidelity
        self.seed = seed
        self.initial_terrain_params = initial_terrain_params or {}
        self.target_terrain_params = target_terrain_params or {}
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Create terrain instance
        self.terrain = Terrain(
            terrain_size=terrain_size,
            noise=noise,
            smoothness=smoothness,
            seed=seed
        )
        
        # Initialize metrics tracking
        self.metrics = {
            'iterations': [],
            'wells_placed': [],
            'mean_squared_error': [],
            'monetary_cost': [],
            'time_cost': [],
            'fidelity': []
        }
    
    @abstractmethod
    def optimize(self) -> Dict:
        """
        Run the optimization algorithm.
        
        Returns:
            Dict containing:
            - wells: List of placed wells
            - metrics: Dictionary of performance metrics
            - terrain_summary: Summary of final terrain state
        """
        pass
    
    def update_metrics(self, 
                      iteration: int,
                      wells_placed: int,
                      mse: float,
                      monetary_cost: float,
                      time_cost: float,
                      fidelity: float,
                      wells: List[Well] = None):
        """Update the metrics tracking and log to wandb."""
        # Validate input types
        if not isinstance(iteration, int):
            raise TypeError(f"iteration must be int, got {type(iteration)}")
        if not isinstance(wells_placed, int):
            raise TypeError(f"wells_placed must be int, got {type(wells_placed)}")
        if not isinstance(mse, (int, float)):
            raise TypeError(f"mse must be numeric, got {type(mse)}")
        if not isinstance(monetary_cost, (int, float)):
            raise TypeError(f"monetary_cost must be numeric, got {type(monetary_cost)}")
        if not isinstance(time_cost, (int, float)):
            raise TypeError(f"time_cost must be numeric, got {type(time_cost)}")
        if not isinstance(fidelity, (int, float)):
            raise TypeError(f"fidelity must be numeric, got {type(fidelity)}")
        
        # Update local metrics
        self.metrics['iterations'].append(iteration)
        self.metrics['wells_placed'].append(wells_placed)
        self.metrics['mean_squared_error'].append(float(mse))
        self.metrics['monetary_cost'].append(float(monetary_cost))
        self.metrics['time_cost'].append(float(time_cost))
        self.metrics['fidelity'].append(float(fidelity))
        
        # Calculate total costs
        total_monetary_cost = sum(self.metrics['monetary_cost'])
        total_time_cost = sum(self.metrics['time_cost'])
        
        # Prepare well placement data
        well_data = []
        if wells:
            if not isinstance(wells, list):
                raise TypeError(f"wells must be list, got {type(wells)}")
            well_data = [
                {
                    "x": float(well.x0),
                    "y": float(well.y0),
                    "depth": float(well.depth),
                    "volume": float(well.volume)
                }
                for well in wells
            ]
        
        # Log to wandb with consistent metric names
        wandb.log({
            "iteration": iteration,
            "wells_placed": wells_placed,
            "mean_squared_error": float(mse),
            "monetary_cost": float(monetary_cost),
            "time_cost": float(time_cost),
            "fidelity": float(fidelity),
            "total_monetary_cost": float(total_monetary_cost),
            "total_time_cost": float(total_time_cost),
            "wells": well_data
        })
    
    def get_metrics(self) -> Dict:
        """Get the current metrics."""
        return self.metrics
    
    def get_summary(self) -> Dict:
        """Get a summary of the optimization results and log to wandb."""
        # Ensure all values are of correct type
        summary = {
            'total_iterations': int(len(self.metrics['iterations'])),
            'total_wells': int(self.metrics['wells_placed'][-1] if self.metrics['wells_placed'] else 0),
            'final_mse': float(self.metrics['mean_squared_error'][-1] if self.metrics['mean_squared_error'] else float('inf')),
            'total_monetary_cost': float(self.metrics['monetary_cost'][-1] if self.metrics['monetary_cost'] else 0),
            'total_time_cost': float(self.metrics['time_cost'][-1] if self.metrics['time_cost'] else 0),
            'final_fidelity': float(self.metrics['fidelity'][-1] if self.metrics['fidelity'] else 0)
        }
        
        # Log final summary to wandb
        wandb.log({"final_terrain_summary": summary})
        
        return summary 