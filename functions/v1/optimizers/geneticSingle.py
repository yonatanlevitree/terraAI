import torch
import numpy as np
from typing import List, Tuple, Dict, Any
from .base import BaseOptimizer
from ..system import Terrain, Well
from scipy.optimize import minimize

class GeneticOptimizer(BaseOptimizer):
    """
    A genetic algorithm optimizer for well placement that uses evolutionary strategies.
    This optimizer maintains a population of wells and evolves them through
    selection, crossover, and mutation to find optimal well placements.
    Each well is placed sequentially, with previous wells affecting the terrain
    for subsequent well placements.
    """
    
    def __init__(self, 
                 terrainSize: int,
                 noise: float,
                 smoothness: float,
                 maxIterations: int,
                 depthBounds: Tuple[float, float],
                 volumeBounds: Tuple[float, float],
                 monetaryLimit: float,
                 timeLimit: float,
                 fidelity: float,
                 seed: int = None,
                 populationSize: int = 50,
                 mutationRate: float = 0.1,
                 tournamentSize: int = 6,
                 eliteSize: int = 3,
                 algorithm=None,
                 progress_callback=None):
        """
        Initialize the genetic algorithm optimizer
        
        Args:
            terrainSize: Size of the terrain grid
            noise: Level of noise in terrain generation
            smoothness: Smoothness factor for terrain
            maxIterations: Maximum number of iterations allowed
            depthBounds: Tuple of (min_depth, max_depth) for wells
            volumeBounds: Tuple of (min_volume, max_volume) for wells
            monetaryLimit: Maximum monetary cost allowed
            timeLimit: Maximum time cost allowed
            fidelity: Target fidelity level to achieve
            seed: Random seed for reproducibility
            populationSize: Number of solutions in each generation
            mutationRate: Probability of mutation for each well
            tournamentSize: Number of solutions to compare in tournament selection
            eliteSize: Number of best solutions to preserve in each generation
            algorithm: Algorithm name for consistency
            progress_callback: Callback function for progress updates
        """
        super().__init__(
            terrainSize=terrainSize,
            noise=noise,
            smoothness=smoothness,
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
        self.populationSize = populationSize
        self.mutationRate = mutationRate
        self.tournamentSize = tournamentSize
        self.eliteSize = eliteSize
        self.algorithm = algorithm
        self.progress_callback = progress_callback
        
    def _initialize_population(self) -> List[Well]:
        """Initialize a population of random wells"""
        population = []
        for _ in range(self.populationSize):
            x = int(np.random.randint(0, self.terrainSize))
            y = int(np.random.randint(0, self.terrainSize))
            depth = float(np.random.uniform(self.depthBounds[0], self.depthBounds[1]))
            volume = float(np.random.uniform(self.volumeBounds[0], self.volumeBounds[1]))
            population.append(Well(x, y, depth, volume))
        return population
    
    def _calculate_fitness(self, well: Well, current_terrain: torch.Tensor) -> float:
        """Calculate asymmetric fitness loss (penalize overshoot more than undershoot)"""
        modified_terrain = self.terrain.apply_wells([well])

        discrepancy = modified_terrain - self.terrain.goal_terrain
        overshoot_weight = 20
        undershoot_weight = 1

        loss = torch.where(discrepancy > 0,
                        overshoot_weight * discrepancy ** 2,
                        undershoot_weight * discrepancy ** 2)

        mse = float(torch.mean(loss))

        if well.monetaryCost() > self.monetaryLimit or well.time_cost() > self.timeLimit:
            mse = float('inf')

        return mse
    
    def _tournament_selection(self, population: List[Well], 
                            fitnesses: List[float]) -> Well:
        """Select a well using tournament selection"""
        tournament_indices = np.random.choice(len(population), self.tournamentSize, replace=False)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitnesses)]  # Lower MSE is better
        return population[winner_idx]
    
    def _crossover(self, parent1: Well, parent2: Well) -> Well:
        """Perform crossover between two parent wells"""
        # Create a new well with parameters from either parent
        if np.random.random() < 0.5:
            x, y = parent1.x0, parent1.y0
        else:
            x, y = parent2.x0, parent2.y0
            
        if np.random.random() < 0.5:
            depth = parent1.depth
        else:
            depth = parent2.depth
            
        if np.random.random() < 0.5:
            volume = parent1.volume
        else:
            volume = parent2.volume
            
        return Well(x, y, depth, volume)
    
    def _mutate(self, well: Well) -> Well:
        """Mutate a well by randomly modifying its parameters.
        For x,y coordinates, changes are restricted to a neighborhood around the original position."""
        # Define neighborhood size (can be adjusted)
        neighborhood_size = 5
        
        if np.random.random() < self.mutationRate:
            # Mutate x within neighborhood
            x_min = max(0, well.x0 - neighborhood_size)
            x_max = min(self.terrainSize - 1, well.x0 + neighborhood_size)
            x = int(np.random.randint(x_min, x_max + 1))
        else:
            x = well.x0
            
        if np.random.random() < self.mutationRate:
            # Mutate y within neighborhood
            y_min = max(0, well.y0 - neighborhood_size)
            y_max = min(self.terrainSize - 1, well.y0 + neighborhood_size)
            y = int(np.random.randint(y_min, y_max + 1))
        else:
            y = well.y0
            
        if np.random.random() < self.mutationRate:
            depth = float(np.random.uniform(self.depthBounds[0], self.depthBounds[1]))
        else:
            depth = well.depth
            
        if np.random.random() < self.mutationRate:
            volume = float(np.random.uniform(self.volumeBounds[0], self.volumeBounds[1]))
        else:
            volume = well.volume
            
        return Well(x, y, depth, volume)
    
    def optimize(self) -> Dict:
        """Run the genetic optimization algorithm."""
        # Get initial and goal terrains
        initial_terrain = self.terrain.initial_terrain
        goal_terrain = self.terrain.goal_terrain
        current_terrain = initial_terrain.clone()
        
        wells = []
        iteration = 0
        monetaryCost = 0.0
        timeCost = 0.0
        
        while iteration < self.maxIterations:
            # Initialize population for this iteration
            population = self._initialize_population()
            best_well = None
            best_fitness = float('inf')
            
            # Evolve population for this well placement
            for generation in range(150):  # Evolve for 50 generations per well
                # Calculate fitness for all wells
                fitnesses = [self._calculate_fitness(well, current_terrain) for well in population]
                
                # Update best well
                min_fitness_idx = np.argmin(fitnesses)
                if fitnesses[min_fitness_idx] < best_fitness:
                    best_fitness = fitnesses[min_fitness_idx]
                    best_well = population[min_fitness_idx]
                
                # Create new population
                new_population = []
                
                # Elitism: keep best solutions
                elite_indices = np.argsort(fitnesses)[:self.eliteSize]
                for idx in elite_indices:
                    new_population.append(population[idx])
                
                # Fill rest of population with offspring
                while len(new_population) < self.populationSize:
                    # Select parents
                    parent1 = self._tournament_selection(population, fitnesses)
                    parent2 = self._tournament_selection(population, fitnesses)
                    
                    # Create offspring
                    child = self._crossover(parent1, parent2)
                    child = self._mutate(child)
                    
                    new_population.append(child)
                
                population = new_population
            
            # Optimize the best well's parameters
            self._optimize_well_parameters([best_well], current_terrain)
            
            # Check if we've exceeded limits
            if monetaryCost + best_well.monetaryCost() > self.monetaryLimit or timeCost + best_well.time_cost() > self.timeLimit:
                break
            
            # Add well to list
            wells.append(best_well)
            
            # Update costs
            monetaryCost += best_well.monetaryCost()
            timeCost += best_well.time_cost()
            
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
                fidelity=fidelity
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