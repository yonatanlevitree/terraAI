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
                 terrain_size: int,
                 noise: float,
                 smoothness: float,
                 max_iterations: int,
                 depth_bounds: Tuple[float, float],
                 volume_bounds: Tuple[float, float],
                 monetary_limit: float,
                 time_limit: float,
                 target_fidelity: float,
                 seed: int = None,
                 population_size: int = 50,
                 mutation_rate: float = 0.1,
                 tournament_size: int = 3,
                 elite_size: int = 2):
        """
        Initialize the genetic algorithm optimizer
        
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
            population_size: Number of solutions in each generation
            mutation_rate: Probability of mutation for each well
            tournament_size: Number of solutions to compare in tournament selection
            elite_size: Number of best solutions to preserve in each generation
        """
        super().__init__(
            terrain_size=terrain_size,
            noise=noise,
            smoothness=smoothness,
            max_iterations=max_iterations,
            depth_bounds=depth_bounds,
            volume_bounds=volume_bounds,
            monetary_limit=monetary_limit,
            time_limit=time_limit,
            target_fidelity=target_fidelity,
            seed=seed
        )
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        
    def _initialize_population(self) -> List[Well]:
        """Initialize a population of random wells"""
        population = []
        for _ in range(self.population_size):
            x = int(np.random.randint(0, self.terrain_size))
            y = int(np.random.randint(0, self.terrain_size))
            depth = float(np.random.uniform(self.depth_bounds[0], self.depth_bounds[1]))
            volume = float(np.random.uniform(self.volume_bounds[0], self.volume_bounds[1]))
            population.append(Well(x, y, depth, volume))
        return population
    
    def _calculate_fitness(self, well: Well, current_terrain: torch.Tensor) -> float:
        """Calculate fitness (MSE) for a well placement"""
        # Apply the well to the current terrain
        modified_terrain = self.terrain.apply_wells([well])
        
        # Calculate MSE between modified terrain and goal terrain
        mse = float(torch.mean((self.terrain.goal_terrain - modified_terrain) ** 2))
        
        # Add penalty for costs
        if well.monetary_cost() > self.monetary_limit or well.time_cost() > self.time_limit:
            mse = float('inf')
            
        return mse
    
    def _tournament_selection(self, population: List[Well], 
                            fitnesses: List[float]) -> Well:
        """Select a well using tournament selection"""
        tournament_indices = np.random.choice(len(population), self.tournament_size, replace=False)
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
        
        if np.random.random() < self.mutation_rate:
            # Mutate x within neighborhood
            x_min = max(0, well.x0 - neighborhood_size)
            x_max = min(self.terrain_size - 1, well.x0 + neighborhood_size)
            x = int(np.random.randint(x_min, x_max + 1))
        else:
            x = well.x0
            
        if np.random.random() < self.mutation_rate:
            # Mutate y within neighborhood
            y_min = max(0, well.y0 - neighborhood_size)
            y_max = min(self.terrain_size - 1, well.y0 + neighborhood_size)
            y = int(np.random.randint(y_min, y_max + 1))
        else:
            y = well.y0
            
        if np.random.random() < self.mutation_rate:
            depth = float(np.random.uniform(self.depth_bounds[0], self.depth_bounds[1]))
        else:
            depth = well.depth
            
        if np.random.random() < self.mutation_rate:
            volume = float(np.random.uniform(self.volume_bounds[0], self.volume_bounds[1]))
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
        monetary_cost = 0.0
        time_cost = 0.0
        
        while iteration < self.max_iterations:
            # Initialize population for this iteration
            population = self._initialize_population()
            best_well = None
            best_fitness = float('inf')
            
            # Evolve population for this well placement
            for generation in range(50):  # Evolve for 50 generations per well
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
                elite_indices = np.argsort(fitnesses)[:self.elite_size]
                for idx in elite_indices:
                    new_population.append(population[idx])
                
                # Fill rest of population with offspring
                while len(new_population) < self.population_size:
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
            if monetary_cost + best_well.monetary_cost() > self.monetary_limit or time_cost + best_well.time_cost() > self.time_limit:
                break
            
            # Add well to list
            wells.append(best_well)
            
            # Update costs
            monetary_cost += best_well.monetary_cost()
            time_cost += best_well.time_cost()
            
            # Update terrain
            current_terrain = self.terrain.apply_wells(wells)
            
            # Calculate metrics
            mse = float(torch.mean((goal_terrain - current_terrain) ** 2))
            fidelity = 1.0 - mse
            
            # Update metrics
            self.update_metrics(
                iteration=iteration,
                wells_placed=len(wells),
                mse=mse,
                monetary_cost=monetary_cost,
                time_cost=time_cost,
                fidelity=fidelity
            )
            
            # Check if we've reached target fidelity
            if fidelity >= self.target_fidelity:
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
        bounds = [(self.depth_bounds[0], self.depth_bounds[1]), 
                 (self.volume_bounds[0], self.volume_bounds[1])] * len(wells)
        
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