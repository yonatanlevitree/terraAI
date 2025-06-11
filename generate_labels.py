import os
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from tqdm import tqdm
import logging
from functions.v1.optimizers.genetic import GeneticOptimizer
from functions.v1.optimizers.greedy import GreedyOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TerrainMatrixLoader:
    """Loads and preprocesses terrain matrices from various formats."""
    
    def __init__(self, data_dir: str, target_size: int = 100):
        """
        Initialize the terrain matrix loader.
        
        Args:
            data_dir: Directory containing terrain matrices
            target_size: Size to resize matrices to (will be square)
        """
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.supported_extensions = {'.npy', '.tif', '.tiff'}
        
    def load_matrix(self, file_path: Path) -> np.ndarray:
        """Load a single terrain matrix and preprocess it."""
        if file_path.suffix == '.npy':
            matrix = np.load(file_path)
        elif file_path.suffix in {'.tif', '.tiff'}:
            import rasterio
            with rasterio.open(file_path) as src:
                matrix = src.read(1)  # Read first band
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
        # Normalize to [0, 1] range
        matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min())
        
        # Resize if needed
        if matrix.shape != (self.target_size, self.target_size):
            from skimage.transform import resize
            matrix = resize(matrix, (self.target_size, self.target_size))
            
        return matrix
    
    def get_all_matrices(self) -> List[Tuple[Path, np.ndarray]]:
        """Load all supported terrain matrices from the data directory."""
        matrices = []
        for ext in self.supported_extensions:
            for file_path in self.data_dir.glob(f"*{ext}"):
                try:
                    matrix = self.load_matrix(file_path)
                    matrices.append((file_path, matrix))
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {str(e)}")
        return matrices

class LabelGenerator:
    """Generates training labels using optimization algorithms."""
    
    def __init__(
        self,
        output_dir: str,
        optimizer_type: str = 'genetic',
        top_n_wells: int = 5,
        optimizer_params: Optional[Dict] = None
    ):
        """
        Initialize the label generator.
        
        Args:
            output_dir: Directory to save generated labels
            optimizer_type: 'genetic' or 'greedy'
            top_n_wells: Number of best wells to save per terrain
            optimizer_params: Optional parameters to override defaults
        """
        self.output_dir = Path(output_dir)
        self.optimizer_type = optimizer_type
        self.top_n_wells = top_n_wells
        self.optimizer_params = optimizer_params or {}
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'matrices').mkdir(exist_ok=True)
        (self.output_dir / 'labels').mkdir(exist_ok=True)
        (self.output_dir / 'metadata').mkdir(exist_ok=True)
        
    def _get_optimizer(self, terrain_matrix: np.ndarray) -> Tuple[GeneticOptimizer, GreedyOptimizer]:
        """Initialize the appropriate optimizer with default or custom parameters."""
        base_params = {
            'terrainSize': terrain_matrix.shape[0],
            'maxIterations': 100,
            'depthBounds': (0.1, 1.0),
            'volumeBounds': (0.1, 1.0),
            'monetaryLimit': float('inf'),
            'timeLimit': float('inf'),     
            'fidelity': 0.95,
            'seed': 42
        }
        
        # Update with custom parameters
        base_params.update(self.optimizer_params)
        
        if self.optimizer_type == 'genetic':
            genetic_params = {
                'populationSize': 50,
                'mutationRate': 0.1,
                'tournamentSize': 6,
                'eliteSize': 3,
                'numGenerations': 150
            }
            genetic_params.update(self.optimizer_params)
            return GeneticOptimizer(**genetic_params)
        else:
            return GreedyOptimizer(**base_params)
    
    def generate_labels(self, matrix_path: Path, terrain_matrix: np.ndarray) -> Dict:
        """Generate labels for a single terrain matrix."""
        try:
            # Initialize optimizer
            optimizer = self._get_optimizer(terrain_matrix)
            
            # Run optimization
            result = optimizer.optimize()
            
            # Extract top N wells
            wells = result['wells'][:self.top_n_wells]
            
            # Prepare output
            matrix_id = matrix_path.stem
            output = {
                'matrix_id': matrix_id,
                'matrix_shape': terrain_matrix.shape,
                'wells': wells,
                'metrics': result['metrics'],
                'terrain_summary': result['terrain_summary']
            }
            
            # Save matrix
            np.save(self.output_dir / 'matrices' / f"{matrix_id}.npy", terrain_matrix)
            
            # Save labels
            with open(self.output_dir / 'labels' / f"{matrix_id}.json", 'w') as f:
                json.dump(output, f, indent=2)
            
            return output
            
        except Exception as e:
            logger.error(f"Error generating labels for {matrix_path}: {str(e)}")
            return None

def main():
    """Main function to generate training data."""
    import argparse
    parser = argparse.ArgumentParser(description='Generate training data for well injection CNN')
    parser.add_argument('--data_dir', required=True, help='Directory containing terrain matrices')
    parser.add_argument('--output_dir', required=True, help='Directory to save generated labels')
    parser.add_argument('--optimizer', choices=['genetic', 'greedy'], default='genetic',
                      help='Optimizer to use for generating labels')
    parser.add_argument('--top_n', type=int, default=5, help='Number of wells to save per terrain')
    parser.add_argument('--matrix_size', type=int, default=100, help='Size to resize matrices to')
    args = parser.parse_args()
    
    # Initialize components
    loader = TerrainMatrixLoader(args.data_dir, args.matrix_size)
    generator = LabelGenerator(
        args.output_dir,
        optimizer_type=args.optimizer,
        top_n_wells=args.top_n
    )
    
    # Load matrices
    logger.info(f"Loading matrices from {args.data_dir}")
    matrices = loader.get_all_matrices()
    logger.info(f"Found {len(matrices)} matrices")
    
    # Generate labels
    results = []
    for matrix_path, matrix in tqdm(matrices, desc="Generating labels"):
        result = generator.generate_labels(matrix_path, matrix)
        if result:
            results.append(result)
    
    # Save metadata
    metadata = {
        'total_matrices': len(matrices),
        'successful_generations': len(results),
        'optimizer_type': args.optimizer,
        'matrix_size': args.matrix_size,
        'top_n_wells': args.top_n
    }
    
    with open(generator.output_dir / 'metadata' / 'generation_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Generated labels for {len(results)} matrices")
    logger.info(f"Results saved to {args.output_dir}")

if __name__ == '__main__':
    main() 