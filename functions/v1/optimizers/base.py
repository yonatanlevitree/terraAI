from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch
from ..system import Terrain, Well

class BaseOptimizer(ABC):
    """Base class for all terrain optimization algorithms."""
    
    def __init__(
        self,
        terrainSize: int,
        maxIterations: int,
        depthBounds: Tuple[float, float],
        volumeBounds: Tuple[float, float],
        monetaryLimit: float,
        timeLimit: float,
        fidelity: float,
        seed: int = 42,
        initialTerrainParams: Dict = None,
        targetTerrainParams: Dict = None,
        algorithm: str = 'greedy',
        progress_callback=None
    ):
        """
        Initialize the optimizer with simulation parameters.
        
        Args:
            terrainSize: Size of the terrain grid
            maxIterations: Maximum number of iterations allowed
            depthBounds: Tuple of (min_depth, max_depth) for wells
            volumeBounds: Tuple of (min_volume, max_volume) for wells
            monetaryLimit: Maximum monetary cost allowed
            timeLimit: Maximum time cost allowed
            fidelity: Target fidelity level to achieve
            seed: Random seed for reproducibility
            initial_terrain_params: Initial terrain parameters
            target_terrain_params: Target terrain parameters
            algorithm: Optimization algorithm to use
            progress_callback: Callback function for progress updates
        """
      
        # Rest of initialization
        self.terrainSize = terrainSize
        self.maxIterations = maxIterations
        self.depthBounds = depthBounds
        self.volumeBounds = volumeBounds
        self.monetaryLimit = monetaryLimit
        self.timeLimit = timeLimit
        self.fidelity = fidelity
        self.seed = seed if seed is not None else 42
        self.progress_callback = progress_callback
        self.algorithm = algorithm
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Create terrain instance
        self.terrain = Terrain(
            terrainSize=terrainSize,
            noise=0.5,
            smoothness=0.5,
            epsilon=1e-3,
            device='cpu',
            regenerate=False
        )
        
        # Initialize metrics tracking
        self.metrics = {
            'iterations': [],
            'wellsPlaced': [],
            'meanSquaredError': [],
            'monetaryCost': [],
            'timeCost': [],
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
                      wellsPlaced: int,
                      mse: float,
                      monetaryCost: float,
                      timeCost: float,
                      fidelity: float,
                      wells: List[Well] = None):
        # Validate input types
        if not isinstance(iteration, int):
            raise TypeError(f"iteration must be int, got {type(iteration)}")
        if not isinstance(wellsPlaced, int):
            raise TypeError(f"wellsPlaced must be int, got {type(wellsPlaced)}")
        if not isinstance(mse, (int, float)):
            raise TypeError(f"mse must be numeric, got {type(mse)}")
        if not isinstance(monetaryCost, (int, float)):
            raise TypeError(f"monetaryCost must be numeric, got {type(monetaryCost)}")
        if not isinstance(timeCost, (int, float)):
            raise TypeError(f"timeCost must be numeric, got {type(timeCost)}")
        if not isinstance(fidelity, (int, float)):
            raise TypeError(f"fidelity must be numeric, got {type(fidelity)}")
        
        # Update local metrics
        self.metrics['iterations'].append(iteration)
        self.metrics['wellsPlaced'].append(wellsPlaced)
        self.metrics['meanSquaredError'].append(float(mse))
        self.metrics['monetaryCost'].append(float(monetaryCost))
        self.metrics['timeCost'].append(float(timeCost))
        self.metrics['fidelity'].append(float(fidelity))
        
        # Calculate total costs
        totalMonetaryCost = sum(self.metrics['monetaryCost'])
        totalTimeCost = sum(self.metrics['timeCost'])
        
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
        
        # Call progress callback
        if self.progress_callback:
            self.progress_callback(self.get_metrics())
    
    def get_metrics(self) -> Dict:
        """Get the current metrics."""
        return self.metrics
    
    def get_summary(self) -> Dict:
        summary = {
            'totalIterations': int(len(self.metrics.get('iterations', []))),
            'totalWells': int(self.metrics.get('wellsPlaced', [0])[-1] if self.metrics.get('wellsPlaced') else 0),
            'finalMSE': float(self.metrics.get('meanSquaredError', [float('inf')])[-1] if self.metrics.get('meanSquaredError') else float('inf')),
            'monetaryCost': float(self.metrics.get('monetaryCost', [0])[-1] if self.metrics.get('monetaryCost') else 0),
            'timeCost': float(self.metrics.get('timeCost', [0])[-1] if self.metrics.get('timeCost') else 0),
            'finalFidelity': float(self.metrics.get('fidelity', [0])[-1] if self.metrics.get('fidelity') else 0)
        }
        return summary 