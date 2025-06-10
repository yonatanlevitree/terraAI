import torch
import numpy as np
from typing import List, Tuple, Dict, Any
from .base import BaseOptimizer
from ..system import Terrain, Well
from scipy.optimize import minimize
import wandb

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
                 numGenerations: int = 150,
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
            numGenerations: Number of generations per well placement
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
        self.numGenerations = numGenerations
        self.algorithm = algorithm
        self.progress_callback = progress_callback
        
    def _initialize_population(self) -> List[List[Well]]:
        """Initialize a population of individuals, where each individual is a list of 1-20 random wells."""
        population = []
        for _ in range(self.populationSize):
            num_wells = np.random.randint(1, 21)  # Each individual has 1 to 20 wells
            individual = []
            for _ in range(num_wells):
                x = int(np.random.randint(0, self.terrainSize))
                y = int(np.random.randint(0, self.terrainSize))
                depth = float(np.random.randint(self.depthBounds[0], self.depthBounds[1]))
                volume = float(np.random.randint(self.volumeBounds[0], self.volumeBounds[1]))
                individual.append(Well(x, y, depth, volume))
            population.append(individual)
        return population
    
    def _calculate_fitness(self, individual: list, current_terrain: torch.Tensor, penalty_weight: float = 0.01) -> float:
        """Calculate fitness as error + penalty_weight * number of wells."""
        # Apply all wells in the individual's genome to the terrain
        modified_terrain = self.terrain.apply_wells(individual)
        discrepancy = modified_terrain - self.terrain.goal_terrain
        overshoot_weight = 3
        undershoot_weight = 1

        loss = torch.where(discrepancy > 0,
                        overshoot_weight * discrepancy ** 2,
                        undershoot_weight * discrepancy ** 2)
        mse = float(torch.mean(loss))
        # Add penalty for number of wells
        num_wells = len(individual)
        fitness = mse + penalty_weight * num_wells
        # Optionally, add constraints (e.g., cost/time limits)
        # if ...: fitness = float('inf')
        return fitness
    
    def _tournament_selection(self, population: list, fitnesses: list) -> list:
        """Select an individual using tournament selection."""
        tournament_indices = np.random.choice(len(population), self.tournamentSize, replace=False)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitnesses)]  # Lower fitness is better
        # Deep copy to avoid mutation side effects
        return [Well(w.x0, w.y0, w.depth, w.volume) for w in population[winner_idx]]
    
    def _crossover(self, parent1: list, parent2: list) -> list:
        """One-point crossover for lists of wells."""
        if len(parent1) == 0 or len(parent2) == 0:
            # If either parent is empty, return a copy of the other
            return parent1.copy() if len(parent2) == 0 else parent2.copy()
        # Choose crossover points
        point1 = np.random.randint(1, len(parent1)+1)
        point2 = np.random.randint(1, len(parent2)+1)
        # Combine segments from both parents
        child = parent1[:point1] + parent2[point2:]
        # Enforce max length
        if len(child) > 20:
            child = child[:20]
        return [Well(w.x0, w.y0, w.depth, w.volume) for w in child]  # Deep copy

    def _mutate(self, individual: list) -> list:
        """Mutate an individual by mutating wells, adding, or removing wells."""
        # Mutate well parameters
        for i in range(len(individual)):
            well = individual[i]
            if np.random.random() < self.mutationRate:
                # Mutate x
                x_min = max(0, well.x0 - 5)
                x_max = min(self.terrainSize - 1, well.x0 + 5)
                well.x0 = int(np.random.randint(x_min, x_max + 1))
            if np.random.random() < self.mutationRate:
                # Mutate y
                y_min = max(0, well.y0 - 5)
                y_max = min(self.terrainSize - 1, well.y0 + 5)
                well.y0 = int(np.random.randint(y_min, y_max + 1))
            if np.random.random() < self.mutationRate:
                well.depth = float(np.random.randint(self.depthBounds[0], self.depthBounds[1]))
            if np.random.random() < self.mutationRate:
                well.volume = float(np.random.randint(self.volumeBounds[0], self.volumeBounds[1]))
        # Possibly add a new well
        if len(individual) < 20 and np.random.random() < self.mutationRate:
            x = int(np.random.randint(0, self.terrainSize))
            y = int(np.random.randint(0, self.terrainSize))
            depth = float(np.random.randint(self.depthBounds[0], self.depthBounds[1]))
            volume = float(np.random.randint(self.volumeBounds[0], self.volumeBounds[1]))
            individual.append(Well(x, y, depth, volume))
        # Possibly remove a well
        if len(individual) > 1 and np.random.random() < self.mutationRate:
            idx = np.random.randint(0, len(individual))
            individual.pop(idx)
        return individual
    
    def optimize(self) -> Dict:
        """Run the genetic optimization algorithm for a population of individuals (each a list of wells)."""
        try:
            initial_terrain = self.terrain.initial_terrain
            goal_terrain = self.terrain.goal_terrain
            best_individual = None
            best_fitness = float('inf')
            best_metrics = None
            best_summary = None
            population = self._initialize_population()
            for generation in range(self.numGenerations):
                # Evaluate fitness for all individuals
                fitnesses = [self._calculate_fitness(individual, initial_terrain) for individual in population]
                # Find best in this generation
                min_fitness_idx = np.argmin(fitnesses)
                if fitnesses[min_fitness_idx] < best_fitness:
                    best_fitness = fitnesses[min_fitness_idx]
                    best_individual = [Well(w.x0, w.y0, w.depth, w.volume) for w in population[min_fitness_idx]]
                    # Calculate metrics for the best individual
                    modified_terrain = self.terrain.apply_wells(best_individual)
                    mse = float(torch.mean((goal_terrain - modified_terrain) ** 2))
                    fidelity = 1.0 - mse
                    best_metrics = {
                        'generation': generation,
                        'wellsPlaced': len(best_individual),
                        'mse': mse,
                        'fidelity': fidelity
                    }
                    best_summary = {
                        'finalMSE': mse,
                        'finalFidelity': fidelity,
                        'totalWells': len(best_individual)
                    }
                    # Update progress callback for live progress
                    if self.progress_callback:
                        self.progress_callback(best_metrics)
                # Elitism: keep best solutions
                new_population = []
                elite_indices = np.argsort(fitnesses)[:self.eliteSize]
                for idx in elite_indices:
                    elite = [Well(w.x0, w.y0, w.depth, w.volume) for w in population[idx]]
                    new_population.append(elite)
                # Fill rest of population with offspring
                while len(new_population) < self.populationSize:
                    parent1 = self._tournament_selection(population, fitnesses)
                    parent2 = self._tournament_selection(population, fitnesses)
                    child = self._crossover(parent1, parent2)
                    child = self._mutate(child)
                    new_population.append(child)
                population = new_population
            # Prepare output
            wells_dict = [
                {
                    'x': int(well.x0),
                    'y': int(well.y0),
                    'depth': float(well.depth),
                    'volume': float(well.volume)
                }
                for well in best_individual
            ]
            return {
                'wells': wells_dict,
                'metrics': best_metrics,
                'terrain_summary': best_summary
            }
        except Exception as e:
            print(f"Error in genetic optimization: {str(e)}")
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
            overshoot_weight = 5  # Higher weight for overshooting
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