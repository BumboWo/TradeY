from typing import List, Dict, Tuple
import numpy as np
from src.core.model import TradingModel
import logging
import random
from datetime import datetime

class ModelPopulation:
    def __init__(self, input_shape: Tuple[int, int], population_size: int = 20,
                 mutation_rate: float = 0.1, mutation_scale: float = 0.1):
        self.input_shape = input_shape
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.generation = 0
        self.population: List[TradingModel] = []
        self.fitness_scores: Dict[int, float] = {}
        self.best_model = None
        self.best_fitness = float('-inf')
        
        self._initialize_population()

    def _initialize_population(self):
        """Initialize the population with random models"""
        self.population = [
            TradingModel(self.input_shape) for _ in range(self.population_size)
        ]

    def evaluate_fitness(self, X: np.ndarray, y: np.ndarray) -> Dict[int, float]:
        """Evaluate fitness of all models in population"""
        self.fitness_scores = {}
        
        for i, model in enumerate(self.population):
            # Train the model
            history = model.train(X, y, epochs=50, batch_size=32)
            
            # Calculate fitness score based on validation loss
            val_loss = min(history.history['val_loss'])
            fitness_score = 1.0 / (1.0 + val_loss)  # Convert loss to fitness
            
            self.fitness_scores[i] = fitness_score
            
            # Update best model
            if fitness_score > self.best_fitness:
                self.best_fitness = fitness_score
                self.best_model = model.clone()
        
        return self.fitness_scores

    def select_parents(self) -> List[TradingModel]:
        """Select parents for next generation using tournament selection"""
        parents = []
        tournament_size = max(2, self.population_size // 5)
        
        while len(parents) < self.population_size:
            # Select random candidates for tournament
            tournament_candidates = random.sample(range(self.population_size), tournament_size)
            
            # Select winner based on fitness
            winner_idx = max(tournament_candidates, 
                           key=lambda idx: self.fitness_scores.get(idx, float('-inf')))
            
            parents.append(self.population[winner_idx].clone())
        
        return parents

    def crossover(self, parents: List[TradingModel]) -> List[TradingModel]:
        """Create new generation through crossover"""
        offspring = []
        
        while len(offspring) < self.population_size:
            # Select two parents
            parent1, parent2 = random.sample(parents, 2)
            
            # Create child through weight averaging
            child = TradingModel(self.input_shape)
            weights1 = parent1.model.get_weights()
            weights2 = parent2.model.get_weights()
            
            # Crossover weights
            child_weights = []
            for w1, w2 in zip(weights1, weights2):
                # Random mixing ratio for each layer
                mix_ratio = np.random.random()
                child_weights.append(w1 * mix_ratio + w2 * (1 - mix_ratio))
            
            child.model.set_weights(child_weights)
            offspring.append(child)
        
        return offspring

    def mutate(self, models: List[TradingModel]):
        """Apply mutations to the population"""
        for model in models:
            if random.random() < self.mutation_rate:
                model.mutate(self.mutation_rate, self.mutation_scale)

    def evolve(self, X: np.ndarray, y: np.ndarray):
        """Perform one generation of evolution"""
        # Evaluate current population
        self.evaluate_fitness(X, y)
        
        # Select parents
        parents = self.select_parents()
        
        # Create new generation through crossover
        offspring = self.crossover(parents)
        
        # Apply mutations
        self.mutate(offspring)
        
        # Update population
        self.population = offspring
        self.generation += 1
        
        logging.info(f"Generation {self.generation} completed. "
                    f"Best fitness: {self.best_fitness:.4f}")

    def get_best_model(self) -> TradingModel:
        """Return the best performing model"""
        return self.best_model

    def get_population_stats(self) -> Dict:
        """Get statistics about the current population"""
        fitness_values = list(self.fitness_scores.values())
        return {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'mean_fitness': np.mean(fitness_values),
            'std_fitness': np.std(fitness_values),
            'population_size': len(self.population),
            'timestamp': datetime.now().isoformat()
        }

    def save_population(self, path: str):
        """Save the entire population"""
        population_data = {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'mutation_rate': self.mutation_rate,
            'mutation_scale': self.mutation_scale,
            'input_shape': self.input_shape,
            'timestamp': datetime.now().isoformat()
        }
        
        np.save(f"{path}/population_meta.npy", population_data)
        
        # Save best model
        if self.best_model is not None:
            self.best_model.save(f"{path}/best_model.h5")
        
        # Save all models
        for i, model in enumerate(self.population):
            model.save(f"{path}/model_{i}.h5")

    @classmethod
    def load_population(cls, path: str) -> 'ModelPopulation':
        """Load a saved population"""
        population_data = np.load(f"{path}/population_meta.npy", allow_pickle=True).item()
        
        # Create new population instance
        population = cls(
            input_shape=population_data['input_shape'],
            population_size=population_data.get('population_size', 20),
            mutation_rate=population_data['mutation_rate'],
            mutation_scale=population_data['mutation_scale']
        )
        
        population.generation = population_data['generation']
        population.best_fitness = population_data['best_fitness']
        
        # Load best model if exists
        try:
            population.best_model = TradingModel.load(f"{path}/best_model.h5")
        except:
            logging.warning("Best model not found in saved population")
        
        # Load all models
        population.population = []
        i = 0
        while True:
            try:
                model = TradingModel.load(f"{path}/model_{i}.h5")
                population.population.append(model)
                i += 1
            except:
                break
        
        return population
