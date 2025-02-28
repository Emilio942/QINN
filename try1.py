import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time
from scipy.optimize import minimize
from typing import List, Tuple, Callable, Dict, Any, Optional
import random

class QuantumInspiredNeuron:
    """Quantum-inspiriertes Neuron, das Superposition und Entanglement simuliert."""
    
    def __init__(self, input_size: int, activation: str = 'tanh'):
        """
        Initialisiert ein Quantum-inspiriertes Neuron.
        
        Args:
            input_size: Anzahl der Eingabe-Features
            activation: Aktivierungsfunktion ('tanh', 'relu', 'sigmoid')
        """
        # Quanteninspirierte Gewichte - komplexe Zahlen für Amplituden und Phasen
        self.weights = np.random.uniform(-1, 1, input_size) + 1j * np.random.uniform(-1, 1, input_size)
        self.bias = np.random.uniform(-1, 1) + 1j * np.random.uniform(-1, 1)
        self.activation = activation
        self.entanglement_factors = np.random.uniform(0, 1, input_size)
        self.input_size = input_size
    
    def forward(self, x: np.ndarray) -> complex:
        """
        Forward-Pass durch das Neuron mit quanteninspirierten Operationen.
        
        Args:
            x: Eingabe-Vektor
            
        Returns:
            Ausgabe des Neurons
        """
        # Prüfen, ob die Dimensionen übereinstimmen
        if len(x) != self.input_size:
            # Anpassung der Dimensionen, falls notwendig (durch Abschneiden oder Padding)
            if len(x) > self.input_size:
                x = x[:self.input_size]  # Abschneiden
            else:
                # Padding mit Nullen
                x_padded = np.zeros(self.input_size, dtype=x.dtype)
                x_padded[:len(x)] = x
                x = x_padded
                
        # Simuliere Superposition durch komplexe Gewichtung
        z = np.sum(self.weights * x * (1 + 1j * self.entanglement_factors)) + self.bias
        
        # Anwenden der Aktivierungsfunktion auf den Realteil und Imaginärteil
        if self.activation == 'tanh':
            return np.tanh(z.real) + 1j * np.tanh(z.imag)
        elif self.activation == 'relu':
            return max(0, z.real) + 1j * max(0, z.imag)
        elif self.activation == 'sigmoid':
            # Clipping zur Vermeidung von Overflow
            z_real = np.clip(z.real, -100, 100)
            z_imag = np.clip(z.imag, -100, 100)
            return 1/(1 + np.exp(-z_real)) + 1j * 1/(1 + np.exp(-z_imag))
        else:
            return z
    
    def update(self, grad: complex, learning_rate: float = 0.01):
        """
        Aktualisiert die Gewichte basierend auf dem Gradienten.
        
        Args:
            grad: Komplexer Gradient (Skalarwert für vereinfachte Backpropagation)
            learning_rate: Lernrate
        """
        # Aktualisiere Gewichte und Bias mit dem skalaren Gradienten
        # Korrigierte Version verwendet Broadcast für alle Gewichte
        self.weights = self.weights - learning_rate * grad
        self.bias = self.bias - learning_rate * grad
        
        # Aktualisiere Entanglement-Faktoren
        self.entanglement_factors = np.clip(
            self.entanglement_factors + learning_rate * np.random.uniform(-0.1, 0.1, len(self.entanglement_factors)),
            0, 1
        )

class QuantumInspiredLayer:
    """Eine Schicht aus Quantum-inspirierten Neuronen."""
    
    def __init__(self, input_size: int, output_size: int, activation: str = 'tanh'):
        """
        Initialisiert eine Schicht von Quantum-inspirierten Neuronen.
        
        Args:
            input_size: Anzahl der Eingabe-Features
            output_size: Anzahl der Neuronen in der Schicht
            activation: Aktivierungsfunktion für die Neuronen
        """
        self.neurons = [QuantumInspiredNeuron(input_size, activation) for _ in range(output_size)]
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward-Pass durch die Schicht.
        
        Args:
            x: Eingabe-Vektor
            
        Returns:
            Ausgabe der Schicht
        """
        return np.array([neuron.forward(x) for neuron in self.neurons])
    
    def update(self, grads: np.ndarray, learning_rate: float = 0.01):
        """
        Aktualisiert alle Neuronen in der Schicht.
        
        Args:
            grads: Gradienten für jedes Neuron (one-to-one mapping)
            learning_rate: Lernrate
        """
        # Stellen wir sicher, dass die Anzahl der Gradienten mit der Anzahl der Neuronen übereinstimmt
        if len(grads) != len(self.neurons):
            # Fallback: Kopiere den ersten Gradienten oder verwende Durchschnitt
            if len(grads) > 0:
                grads = np.full(len(self.neurons), np.mean(grads))
            else:
                grads = np.full(len(self.neurons), 0.01) # Kleiner Default-Wert
                
        # Aktualisiere jedes Neuron mit seinem entsprechenden Gradienten
        for neuron, grad in zip(self.neurons, grads):
            neuron.update(grad, learning_rate)


class QINN:
    """Quantum-Inspired Neural Network für Optimierungsprobleme."""
    
    def __init__(self, layer_sizes: List[int], activations: List[str]):
        """
        Initialisiert ein QINN mit mehreren Schichten.
        
        Args:
            layer_sizes: Liste der Größen jeder Schicht (erstes Element ist Input-Größe)
            activations: Liste der Aktivierungsfunktionen für jede Schicht
        """
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(QuantumInspiredLayer(layer_sizes[i], layer_sizes[i+1], activations[i]))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward-Pass durch das gesamte Netzwerk.
        
        Args:
            x: Eingabe-Vektor
            
        Returns:
            Ausgabe des Netzwerks
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def _compute_gradient(self, x: np.ndarray, y: np.ndarray, output: np.ndarray) -> List[np.ndarray]:
        """
        Berechnet den Gradienten für das Netzwerk.
        
        Args:
            x: Eingabe-Daten
            y: Ziel-Daten
            output: Ausgabe des Netzwerks
            
        Returns:
            Gradienten für jede Schicht
        """
        # Initialer Fehler am Ausgang
        error = output - y
        
        # Liste für die Gradienten jeder Schicht
        gradients = []
        
        # Grundlegende Backpropagation für jede Schicht
        # Im QINN-Kontext vereinfacht, für jedes Neuron individuell
        for layer_idx, layer in enumerate(reversed(self.layers)):
            layer_grads = []
            
            # Für jedes Neuron in der Schicht einen individuellen Gradienten berechnen
            for neuron_idx, neuron in enumerate(layer.neurons):
                # Für jedes Neuron einen skalarwertigen Gradienten basierend auf dem Fehler
                if layer_idx == 0:  # Ausgabeschicht
                    neuron_grad = error[neuron_idx]
                else:
                    # Für versteckte Schichten könnten wir eine komplexere Backpropagation machen,
                    # aber für die Demonstration verwenden wir einen vereinfachten Ansatz
                    neuron_grad = np.mean(error) * 0.5  # Reduzierte Fehlerübertragung
                
                layer_grads.append(neuron_grad)
            
            gradients.insert(0, np.array(layer_grads))
            
            # Fehler für die nächste Schicht aktualisieren (vereinfacht)
            if layer_idx < len(self.layers) - 1:
                error = error * 0.8  # Reduzierter Fehler für tiefere Schichten
        
        return gradients
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, learning_rate: float = 0.01,
              batch_size: int = 32, verbose: bool = True) -> List[float]:
        """
        Trainiert das QINN auf den gegebenen Daten.
        
        Args:
            X: Trainings-Features
            y: Trainings-Labels
            epochs: Anzahl der Trainings-Epochen
            learning_rate: Lernrate
            batch_size: Größe der Batches
            verbose: Ob Trainingsfortschritt ausgegeben werden soll
            
        Returns:
            Liste der Verluste während des Trainings
        """
        losses = []
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Shuffle und Batching
            indices = np.random.permutation(n_samples)
            epoch_loss = 0
            
            for start_idx in range(0, n_samples, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                X_batch, y_batch = X[batch_indices], y[batch_indices]
                
                batch_loss = 0
                for x_sample, y_sample in zip(X_batch, y_batch):
                    # Forward-Pass
                    output = self.forward(x_sample)
                    
                    # Berechne Verlust (Verwende Magnitude für komplexe Zahlen)
                    loss = np.sum(np.abs(output - y_sample)**2)
                    batch_loss += loss
                    
                    # Backpropagation
                    gradients = self._compute_gradient(x_sample, y_sample, output)
                    
                    # Update Gewichte (für jede Schicht separat)
                    for layer_idx, layer in enumerate(self.layers):
                        for neuron_idx, neuron in enumerate(layer.neurons):
                            # Skalarwertiger Gradient für jedes Neuron
                            grad_value = gradients[layer_idx][neuron_idx]
                            # Update des Neurons mit dem skalaren Gradienten
                            neuron.update(grad_value, learning_rate)
                
                epoch_loss += batch_loss / len(X_batch)
            
            avg_epoch_loss = epoch_loss / (n_samples / batch_size)
            losses.append(avg_epoch_loss)
            
            if verbose and (epoch + 1) % (epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
        
        return losses
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Macht Vorhersagen mit dem trainierten Netzwerk.
        
        Args:
            X: Eingabe-Features
            
        Returns:
            Vorhersagen des Netzwerks
        """
        predictions = []
        for x in X:
            output = self.forward(x)
            # Verwende Magnitude für komplexe Zahlen
            predictions.append(np.abs(output))
        return np.array(predictions)


class QuantumInspiredOptimizer:
    """Wrapper-Klasse für verschiedene Optimierungsprobleme mit QINN."""
    
    def __init__(self, problem_type: str, problem_params: Dict[str, Any], 
                 num_processors: Optional[int] = None):
        """
        Initialisiert den Optimierer für ein bestimmtes Problem.
        
        Args:
            problem_type: Art des Problems ('tsp', 'max_cut', 'portfolio', 'chemistry')
            problem_params: Parameter des Problems
            num_processors: Anzahl der zu verwendenden Prozessoren (None = alle verfügbaren)
        """
        self.problem_type = problem_type
        self.problem_params = problem_params
        self.qinn = None
        
        # Automatisch verfügbare Prozessoren erkennen
        self.num_processors = num_processors if num_processors else multiprocessing.cpu_count()
        print(f"Verwende {self.num_processors} Prozessoren für parallele Verarbeitung")
        
        # Problem initialisieren
        self._initialize_problem()
    
    def _initialize_problem(self):
        """Initialisiert die Problemspezifikation und das QINN-Modell."""
        if self.problem_type == 'tsp':
            # Traveling Salesman Problem
            n_cities = self.problem_params.get('n_cities', 10)
            input_size = n_cities**2  # Adjazenzmatrix flachgedrückt
            self.qinn = QINN(
                layer_sizes=[input_size, 128, 64, n_cities], 
                activations=['tanh', 'tanh', 'tanh']
            )
            
            # Zufällige Städtekoordinaten generieren, wenn nicht gegeben
            if 'city_coordinates' not in self.problem_params:
                self.problem_params['city_coordinates'] = np.random.rand(n_cities, 2)
                
        elif self.problem_type == 'max_cut':
            # Maximum Cut Problem auf Graphen
            n_nodes = self.problem_params.get('n_nodes', 10)
            input_size = n_nodes**2  # Adjazenzmatrix flachgedrückt
            self.qinn = QINN(
                layer_sizes=[input_size, 64, 32, n_nodes], 
                activations=['tanh', 'tanh', 'tanh']
            )
            
            # Zufälligen Graphen generieren, wenn nicht gegeben
            if 'adjacency_matrix' not in self.problem_params:
                adj_matrix = np.random.randint(0, 2, (n_nodes, n_nodes))
                # Symmetrisch machen
                adj_matrix = np.triu(adj_matrix) + np.triu(adj_matrix, 1).T
                np.fill_diagonal(adj_matrix, 0)  # Keine Selbstverbindungen
                self.problem_params['adjacency_matrix'] = adj_matrix
                
        elif self.problem_type == 'portfolio':
            # Portfolio-Optimierung
            n_assets = self.problem_params.get('n_assets', 10)
            input_size = n_assets * 2  # Returns und Volatilitäten
            self.qinn = QINN(
                layer_sizes=[input_size, 32, 16, n_assets], 
                activations=['tanh', 'relu', 'sigmoid']
            )
            
            # Zufällige Asset-Daten generieren, wenn nicht gegeben
            if 'returns' not in self.problem_params or 'covariance' not in self.problem_params:
                returns = np.random.normal(0.05, 0.1, n_assets)  # Jährliche Returns
                # Positiv definite Kovarianzmatrix
                temp = np.random.randn(n_assets, n_assets)
                cov = np.dot(temp, temp.transpose()) / 10
                self.problem_params['returns'] = returns
                self.problem_params['covariance'] = cov
                
        elif self.problem_type == 'chemistry':
            # Molekulare Konfigurationsoptimierung
            n_atoms = self.problem_params.get('n_atoms', 10)
            input_size = n_atoms * 4  # 3D-Koordinaten + Atomtyp (dies war der Fehler)
            self.qinn = QINN(
                layer_sizes=[input_size, 64, 32, n_atoms * 3], 
                activations=['tanh', 'tanh', 'tanh']
            )
            
            # Zufällige Moleküldaten generieren, wenn nicht gegeben
            if 'atom_types' not in self.problem_params:
                atom_types = np.random.choice(['H', 'C', 'N', 'O'], n_atoms)
                initial_positions = np.random.randn(n_atoms, 3)
                self.problem_params['atom_types'] = atom_types
                self.problem_params['initial_positions'] = initial_positions
        
        else:
            raise ValueError(f"Unbekannter Problemtyp: {self.problem_type}")
    
    def _prepare_data_for_tsp(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bereitet Daten für das TSP-Problem vor.
        
        Returns:
            Tuple aus X (Adjazenzmatrizen) und y (optimale Touren)
        """
        n_cities = self.problem_params.get('n_cities', 10)
        coords = self.problem_params['city_coordinates']
        
        # Distanzmatrix berechnen
        dist_matrix = np.zeros((n_cities, n_cities))
        for i in range(n_cities):
            for j in range(n_cities):
                dist_matrix[i, j] = np.sqrt(np.sum((coords[i] - coords[j])**2))
        
        # Trainingsbeispiele generieren mit parallelem Simulated Annealing
        X, y = self._parallel_tsp_solutions(dist_matrix, n_samples=100)
        
        return X, y
    
    def _run_sa_for_tsp(self, seed, dist_matrix, n_cities):
        """Ausführung von Simulated Annealing für TSP mit gegebenem Seed.
        Als separate Methode für die Parallelisierung."""
        np.random.seed(seed)
        # Zufällige Tour initialisieren
        tour = np.random.permutation(n_cities)
        best_tour = tour.copy()
        best_dist = self._tour_distance(tour, dist_matrix)
        
        # Simulated Annealing Parameter
        temp = 1.0
        cooling_rate = 0.995
        n_iters = 1000
        
        for i in range(n_iters):
            # 2-opt Swap
            idx1, idx2 = np.random.choice(n_cities, 2, replace=False)
            if idx1 > idx2:
                idx1, idx2 = idx2, idx1
                
            new_tour = tour.copy()
            new_tour[idx1:idx2+1] = np.flip(new_tour[idx1:idx2+1])
            
            # Evaluiere neue Tour
            current_dist = self._tour_distance(tour, dist_matrix)
            new_dist = self._tour_distance(new_tour, dist_matrix)
            
            # Akzeptiere basierend auf Temperatur und Distanzänderung
            if new_dist < current_dist or np.random.random() < np.exp((current_dist - new_dist) / temp):
                tour = new_tour
                if new_dist < best_dist:
                    best_tour = new_tour.copy()
                    best_dist = new_dist
            
            # Temperatur abkühlen
            temp *= cooling_rate
        
        # Adjazenzmatrix aus Tour erzeugen
        adj_matrix = np.zeros((n_cities, n_cities))
        for i in range(n_cities):
            adj_matrix[best_tour[i], best_tour[(i+1) % n_cities]] = 1
        
        # One-hot-Darstellung der Tour
        tour_one_hot = np.zeros(n_cities)
        for i, city in enumerate(best_tour):
            tour_one_hot[city] = i / n_cities
            
        return adj_matrix.flatten(), tour_one_hot
    
    def _parallel_tsp_solutions(self, dist_matrix: np.ndarray, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generiert TSP-Lösungen mit parallelem Simulated Annealing.
        
        Args:
            dist_matrix: Matrix der Distanzen zwischen Städten
            n_samples: Anzahl der zu generierenden Beispiele
            
        Returns:
            Tuple aus X (Adjazenzmatrizen) und y (optimale Touren)
        """
        n_cities = dist_matrix.shape[0]
        
        # Wir verwenden multiprocessing.Pool direkt mit starmap statt ProcessPoolExecutor
        with multiprocessing.Pool(processes=self.num_processors) as pool:
            # Vorbereiten der Argumente für jede Ausführung
            args = [(seed, dist_matrix, n_cities) for seed in range(n_samples)]
            
            # Parallele Ausführung mit expliziten Argumenten
            results = pool.starmap(self._run_sa_for_tsp, args)
        
        X, y = zip(*results)
        return np.array(X), np.array(y)
    
    def _tour_distance(self, tour: np.ndarray, dist_matrix: np.ndarray) -> float:
        """
        Berechnet die Gesamtdistanz einer Tour.
        
        Args:
            tour: Reihenfolge der Städte
            dist_matrix: Distanzmatrix zwischen Städten
            
        Returns:
            Gesamtdistanz der Tour
        """
        n_cities = len(tour)
        total_dist = 0
        for i in range(n_cities):
            total_dist += dist_matrix[tour[i], tour[(i+1) % n_cities]]
        return total_dist
    
    def _prepare_data_for_max_cut(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bereitet Daten für das Max-Cut-Problem vor.
        
        Returns:
            Tuple aus X (Adjazenzmatrizen) und y (optimale Schnitte)
        """
        n_nodes = self.problem_params.get('n_nodes', 10)
        adj_matrix = self.problem_params['adjacency_matrix']
        
        # Trainingsbeispiele mit parallelem Greedy-Algorithmus generieren
        X, y = self._parallel_max_cut_solutions(adj_matrix, n_samples=100)
        
        return X, y
    
    def _run_greedy_for_max_cut(self, seed, adj_matrix, n_nodes):
        """Ausführung eines randomisierten Greedy-Algorithmus mit gegebenem Seed.
        Als separate Methode für die Parallelisierung."""
        np.random.seed(seed)
        # Zufällige Partition initialisieren
        partition = np.random.randint(0, 2, n_nodes)  # 0 oder 1
        best_partition = partition.copy()
        best_cut = self._cut_value(partition, adj_matrix)
        
        # Local Search Verbesserung
        improved = True
        while improved:
            improved = False
            # Für jeden Knoten prüfen, ob Wechsel der Partition den Cut verbessert
            nodes_order = np.random.permutation(n_nodes)
            for node in nodes_order:
                partition[node] = 1 - partition[node]  # Partition wechseln
                new_cut = self._cut_value(partition, adj_matrix)
                
                if new_cut > best_cut:
                    best_cut = new_cut
                    best_partition = partition.copy()
                    improved = True
                else:
                    partition[node] = 1 - partition[node]  # Zurücksetzen
        
        return adj_matrix.flatten(), best_partition
    
    def _parallel_max_cut_solutions(self, adj_matrix: np.ndarray, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generiert Max-Cut-Lösungen mit parallelen Greedy-Algorithmen.
        
        Args:
            adj_matrix: Adjazenzmatrix des Graphen
            n_samples: Anzahl der zu generierenden Beispiele
            
        Returns:
            Tuple aus X (Adjazenzmatrizen) und y (optimale Schnitte)
        """
        n_nodes = adj_matrix.shape[0]
        
        # Multiprocessing Pool mit starmap
        with multiprocessing.Pool(processes=self.num_processors) as pool:
            # Vorbereiten der Argumente für jede Ausführung
            args = [(seed, adj_matrix, n_nodes) for seed in range(n_samples)]
            
            # Parallele Ausführung mit expliziten Argumenten
            results = pool.starmap(self._run_greedy_for_max_cut, args)
        
        X, y = zip(*results)
        return np.array(X), np.array(y)
    
    def _cut_value(self, partition: np.ndarray, adj_matrix: np.ndarray) -> int:
        """
        Berechnet den Wert eines Schnitts.
        
        Args:
            partition: Partition der Knoten (0 oder 1)
            adj_matrix: Adjazenzmatrix des Graphen
            
        Returns:
            Wert des Schnitts
        """
        n_nodes = len(partition)
        cut_value = 0
        
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if adj_matrix[i, j] > 0 and partition[i] != partition[j]:
                    cut_value += adj_matrix[i, j]
        
        return cut_value
    
    def _prepare_data_for_portfolio(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bereitet Daten für das Portfolio-Optimierungsproblem vor.
        
        Returns:
            Tuple aus X (Asset-Daten) und y (optimale Gewichtungen)
        """
        n_assets = self.problem_params.get('n_assets', 10)
        returns = self.problem_params['returns']
        cov_matrix = self.problem_params['covariance']
        
        # Trainingsbeispiele mit verschiedenen Risikoaversions-Parametern generieren
        X, y = self._parallel_portfolio_solutions(returns, cov_matrix, n_samples=100)
        
        return X, y
    
    def _run_portfolio_opt(self, seed, returns, cov_matrix, n_assets):
        """Führt Portfolio-Optimierung mit verschiedenen Parametern durch.
        Als separate Methode für die Parallelisierung."""
        np.random.seed(seed)
        
        # Verschiedene Risikoaversions-Parameter (λ) ausprobieren
        risk_aversion = np.random.uniform(0.5, 5.0)
        
        # Objektfunktion für die Optimierung
        def objective(w):
            portfolio_return = np.sum(w * returns)
            portfolio_risk = np.sqrt(w.T @ cov_matrix @ w)
            # Nutzen = Return - λ * Risiko
            utility = portfolio_return - risk_aversion * portfolio_risk
            return -utility  # Minimierung, daher negativ
        
        # Nebenbedingungen: Summe der Gewichte = 1, keine negativen Gewichte
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Startgewichte
        w0 = np.ones(n_assets) / n_assets
        
        # Optimierung
        result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
        optimal_weights = result.x
        
        # Feature-Vektor: returns & volatilities
        volatilities = np.sqrt(np.diag(cov_matrix))
        features = np.concatenate([returns, volatilities])
        
        return features, optimal_weights
    
    def _parallel_portfolio_solutions(self, returns: np.ndarray, cov_matrix: np.ndarray, 
                                     n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generiert Portfolio-Optimierungs-Lösungen mit paralleler Optimierung.
        
        Args:
            returns: Erwartete Returns der Assets
            cov_matrix: Kovarianzmatrix der Assets
            n_samples: Anzahl der zu generierenden Beispiele
            
        Returns:
            Tuple aus X (Asset-Daten) und y (optimale Gewichtungen)
        """
        n_assets = len(returns)
        
        # Multiprocessing Pool mit starmap
        with multiprocessing.Pool(processes=self.num_processors) as pool:
            # Vorbereiten der Argumente für jede Ausführung
            args = [(seed, returns, cov_matrix, n_assets) for seed in range(n_samples)]
            
            # Parallele Ausführung mit expliziten Argumenten
            results = pool.starmap(self._run_portfolio_opt, args)
        
        X, y = zip(*results)
        return np.array(X), np.array(y)
    
    def _prepare_data_for_chemistry(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bereitet Daten für das Molekulare Konfigurationsproblem vor.
        
        Returns:
            Tuple aus X (Moleküldaten) und y (optimale Konfigurationen)
        """
        n_atoms = self.problem_params.get('n_atoms', 10)
        atom_types = self.problem_params['atom_types']
        initial_positions = self.problem_params['initial_positions']
        
        # Trainingsdaten generieren (Lennard-Jones wird jetzt intern in _run_molecular_opt definiert)
        X, y = self._parallel_molecular_optimizations(atom_types, initial_positions, 
                                                    None, n_samples=100)
        
        return X, y
    
    def _run_molecular_opt(self, seed, atom_types, initial_positions, n_atoms, atom_numbers):
        """Führt molekulare Optimierung mit verschiedenen Parametern durch.
        Als separate Methode für die Parallelisierung."""
        np.random.seed(seed)
        
        # Leichte Variation der Anfangspositionen
        positions = initial_positions + np.random.normal(0, 0.5, initial_positions.shape)
        
        # Interne Hilfsfunktion für Lennard-Jones-Potenzial
        def lennard_jones(r, epsilon=1.0, sigma=1.0):
            """Lennard-Jones Potenzial."""
            return 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)
        
        # Hilfsfunktion für die Gesamtenergie des Systems
        def total_energy(pos_flat):
            pos = pos_flat.reshape(-1, 3)
            energy = 0
            
            for i in range(n_atoms):
                for j in range(i+1, n_atoms):
                    # Abstand zwischen Atomen
                    r = np.sqrt(np.sum((pos[i] - pos[j])**2))
                    if r < 0.1:  # Vermeidung von Singularitäten
                        r = 0.1
                    
                    # Skalierungsparameter basierend auf Atomtypen
                    scale = atom_numbers[i] * atom_numbers[j] / 36  # Normalisiert mit C-C
                    
                    # Energie aus Potenzialfunktion
                    energy += scale * lennard_jones(r)
            
            return energy
        
        # Startpositionen flachdrücken für Optimierung
        pos_flat = positions.flatten()
        
        # Optimierung mit L-BFGS-B
        result = minimize(total_energy, pos_flat, method='L-BFGS-B')
        optimized_positions = result.x.reshape(-1, 3)
        
        # Feature-Vektor: Anfangspositionen & Atomtypen
        features = np.concatenate([positions.flatten(), atom_numbers / 10])
        
        return features, optimized_positions.flatten()
    
    def _parallel_molecular_optimizations(self, atom_types: np.ndarray, initial_positions: np.ndarray,
                                         potential_func: Callable, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generiert optimierte molekulare Konfigurationen parallel.
        
        Args:
            atom_types: Typen der Atome
            initial_positions: Anfangspositionen der Atome
            potential_func: Potenzialfunktion für die Interaktion zwischen Atomen
            n_samples: Anzahl der zu generierenden Beispiele
            
        Returns:
            Tuple aus X (Moleküldaten) und y (optimale Konfigurationen)
        """
        n_atoms = len(atom_types)
        
        # Abbildung von Atomtypen auf numerische Werte für die Optimierung
        atom_type_map = {'H': 1, 'C': 6, 'N': 7, 'O': 8}
        atom_numbers = np.array([atom_type_map.get(atom, 0) for atom in atom_types])
        
        # Für die molekulare Optimierung können wir funktionale Argumente nicht direkt übergeben,
        # da Pool.starmap Probleme mit der Serialisierung haben könnte
        # Daher erstellen wir eine lennard_jones-Funktion direkt in _run_molecular_opt
        
        # Multiprocessing Pool mit starmap
        with multiprocessing.Pool(processes=self.num_processors) as pool:
            # Vorbereiten der Argumente für jede Ausführung (ohne potential_func)
            args = [(seed, atom_types, initial_positions, n_atoms, atom_numbers) for seed in range(n_samples)]
            
            # Parallele Ausführung mit expliziten Argumenten
            results = pool.starmap(self._run_molecular_opt, args)
        
        X, y = zip(*results)
        return np.array(X), np.array(y)
    
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bereitet die Daten für das spezifische Problem vor.
        
        Returns:
            Tuple aus X (Features) und y (optimale Lösungen)
        """
        if self.problem_type == 'tsp':
            return self._prepare_data_for_tsp()
        elif self.problem_type == 'max_cut':
            return self._prepare_data_for_max_cut()
        elif self.problem_type == 'portfolio':
            return self._prepare_data_for_portfolio()
        elif self.problem_type == 'chemistry':
            return self._prepare_data_for_chemistry()
        else:
            raise ValueError(f"Datenaufbereitung für {self.problem_type} nicht implementiert")
    
    def train(self, epochs: int = 500, learning_rate: float = 0.01,
              batch_size: int = 32, verbose: bool = True) -> List[float]:
        """
        Trainiert das QINN-Modell für das Optimierungsproblem.
        
        Args:
            epochs: Anzahl der Trainings-Epochen
            learning_rate: Lernrate
            batch_size: Größe der Batches
            verbose: Ob Trainingsfortschritt ausgegeben werden soll
            
        Returns:
            Liste der Verluste während des Trainings
        """
        print(f"Bereite Daten für {self.problem_type}-Problem vor...")
        X, y = self.prepare_data()
        
        print(f"Starte Training mit {len(X)} Beispielen...")
        start_time = time.time()
        losses = self.qinn.train(X, y, epochs=epochs, learning_rate=learning_rate,
                                batch_size=batch_size, verbose=verbose)
        training_time = time.time() - start_time
        
        print(f"Training abgeschlossen in {training_time:.2f} Sekunden")
        return losses
    
    def solve(self, problem_instance: np.ndarray) -> np.ndarray:
        """
        Löst eine neue Instanz des Optimierungsproblems.
        
        Args:
            problem_instance: Neue Probleminstanz als Eingabe für das QINN
            
        Returns:
            Lösung des Problems
        """
        # Stelle sicher, dass die Instanz die richtige Dimension hat
        # für die forward-Methode des QINN
        problem_instance = problem_instance.flatten()
        
        prediction = self.qinn.predict(problem_instance.reshape(1, -1))[0]
        
        # Nachbearbeitung der Vorhersage basierend auf dem Problem
        if self.problem_type == 'tsp':
            # Konvertiere Vorhersage in eine gültige Tour
            n_cities = len(prediction)
            # Sortieren nach den vorhergesagten Werten, um Reihenfolge zu erhalten
            tour = np.argsort(prediction)
            return tour
            
        elif self.problem_type == 'max_cut':
            # Konvertiere Vorhersage in eine gültige Partition (Binarisierung)
            partition = (prediction > 0.5).astype(int)
            return partition
            
        elif self.problem_type == 'portfolio':
            # Normalisiere die Gewichte, um sicherzustellen, dass sie aufsummieren zu 1
            weights = np.maximum(0, prediction)  # Keine negativen Gewichte
            weights /= np.sum(weights)
            return weights
            
        elif self.problem_type == 'chemistry':
            # Die vorhergesagten Positionen zurückgeben
            n_atoms = self.problem_params.get('n_atoms', 8)
            
            # Stelle sicher, dass die Vorhersage die richtige Länge hat
            if len(prediction) != n_atoms * 3:
                if len(prediction) > n_atoms * 3:
                    prediction = prediction[:n_atoms * 3]
                else:
                    padded = np.zeros(n_atoms * 3)
                    padded[:len(prediction)] = prediction
                    prediction = padded
            
            # In die richtige Form bringen
            positions = prediction.reshape(n_atoms, 3)
            return positions
            
        else:
            return prediction
    
    def evaluate(self, n_samples: int = 10, comparison: bool = True) -> Dict[str, Any]:
        """
        Evaluiert die Leistung des QINN im Vergleich zu klassischen Algorithmen.
        
        Args:
            n_samples: Anzahl der zu evaluierenden Beispiele
            comparison: Ob ein Vergleich mit klassischen Algorithmen durchgeführt werden soll
            
        Returns:
            Dictionary mit Evaluationsergebnissen
        """
        print(f"Evaluiere QINN für {self.problem_type} mit {n_samples} Beispielen...")
        
        # Generiere neue Probleminstanzen
        X_test, y_test = self.prepare_data()
        X_test, y_test = X_test[:n_samples], y_test[:n_samples]
        
        # QINN-Lösungen mit Timing
        qinn_start = time.time()
        qinn_solutions = [self.solve(x) for x in X_test]
        qinn_time = time.time() - qinn_start
        
        # Evaluierungsfunktion je nach Problem
        if self.problem_type == 'tsp':
            def evaluate_solution(tour, instance):
                n_cities = int(np.sqrt(len(instance)))
                dist_matrix = instance.reshape(n_cities, n_cities)
                return self._tour_distance(tour, dist_matrix)
                
        elif self.problem_type == 'max_cut':
            def evaluate_solution(partition, instance):
                n_nodes = int(np.sqrt(len(instance)))
                adj_matrix = instance.reshape(n_nodes, n_nodes)
                return self._cut_value(partition, adj_matrix)
                
        elif self.problem_type == 'portfolio':
            def evaluate_solution(weights, instance):
                n_assets = len(weights)
                returns = instance[:n_assets]
                vols = instance[n_assets:]
                # Sharpe-Ratio als Metrik
                portfolio_return = np.sum(weights * returns)
                portfolio_risk = np.sqrt(np.sum((weights * vols)**2))
                return portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
                
        elif self.problem_type == 'chemistry':
            # Für Chemistry müssen wir die korrekte Anzahl an Atomen verwenden
            n_atoms = self.problem_params.get('n_atoms', 8)
            
            def evaluate_solution(positions, instance):
                # Einfache Bewertung: Summe der quadratischen Abweichungen
                # Stellen wir sicher, dass positions die richtige Form hat
                
                # Prüfen, ob positions bereits ein 2D-Array ist
                try:
                    if isinstance(positions, np.ndarray) and positions.ndim > 1:
                        # Flachdrücken, falls es schon die Form (n_atoms, 3) hat
                        positions = positions.flatten()
                except:
                    pass
                
                # Stellen wir sicher, dass positions die korrekte Länge hat
                if len(positions) != n_atoms * 3:
                    # Anpassung, wenn die Größe nicht übereinstimmt
                    if len(positions) > n_atoms * 3:
                        positions = positions[:n_atoms * 3]
                    else:
                        # Padding mit Nullen
                        padded = np.zeros(n_atoms * 3)
                        padded[:len(positions)] = positions
                        positions = padded
                
                # Extrahieren der Zielposition aus der Instanz
                # Stellen wir sicher, dass wir genug Daten haben
                instance_size = min(len(instance), n_atoms*3) 
                if instance_size < n_atoms*3:
                    target_data = np.zeros(n_atoms*3)
                    target_data[:instance_size] = instance[:instance_size]
                else:
                    target_data = instance[:n_atoms*3]
                
                target_pos = target_data.reshape(n_atoms, 3)
                pos = positions.reshape(n_atoms, 3)
                
                return -np.sum((pos - target_pos)**2)  # Negativ, da höher besser ist
        
        # Evaluiere QINN-Lösungen
        qinn_scores = [evaluate_solution(sol, x) for sol, x in zip(qinn_solutions, X_test)]
        
        results = {
            'problem_type': self.problem_type,
            'qinn_time': qinn_time,
            'qinn_avg_time': qinn_time / n_samples,
            'qinn_scores': qinn_scores,
            'qinn_avg_score': np.mean(qinn_scores)
        }
        
        # Vergleich mit klassischen Algorithmen, wenn gewünscht
        if comparison:
            # Klassische Lösungen
            classic_start = time.time()
            classic_solutions = []
            
            if self.problem_type == 'tsp':
                # Verwende Simulated Annealing (sequentiell zum besseren Vergleich)
                for x in X_test:
                    n_cities = int(np.sqrt(len(x)))
                    dist_matrix = x.reshape(n_cities, n_cities)
                    
                    # Simulated Annealing
                    tour = np.random.permutation(n_cities)
                    best_tour = tour.copy()
                    best_dist = self._tour_distance(tour, dist_matrix)
                    
                    temp = 1.0
                    cooling_rate = 0.995
                    n_iters = 1000
                    
                    for i in range(n_iters):
                        idx1, idx2 = np.random.choice(n_cities, 2, replace=False)
                        if idx1 > idx2:
                            idx1, idx2 = idx2, idx1
                            
                        new_tour = tour.copy()
                        new_tour[idx1:idx2+1] = np.flip(new_tour[idx1:idx2+1])
                        
                        current_dist = self._tour_distance(tour, dist_matrix)
                        new_dist = self._tour_distance(new_tour, dist_matrix)
                        
                        if new_dist < current_dist or np.random.random() < np.exp((current_dist - new_dist) / temp):
                            tour = new_tour
                            if new_dist < best_dist:
                                best_tour = new_tour.copy()
                                best_dist = new_dist
                        
                        temp *= cooling_rate
                    
                    classic_solutions.append(best_tour)
                
            elif self.problem_type == 'max_cut':
                # Verwende Greedy-Algorithmus
                for x in X_test:
                    n_nodes = int(np.sqrt(len(x)))
                    adj_matrix = x.reshape(n_nodes, n_nodes)
                    
                    # Greedy-Algorithmus
                    partition = np.random.randint(0, 2, n_nodes)
                    best_partition = partition.copy()
                    best_cut = self._cut_value(partition, adj_matrix)
                    
                    improved = True
                    while improved:
                        improved = False
                        for node in range(n_nodes):
                            partition[node] = 1 - partition[node]
                            new_cut = self._cut_value(partition, adj_matrix)
                            
                            if new_cut > best_cut:
                                best_cut = new_cut
                                best_partition = partition.copy()
                                improved = True
                            else:
                                partition[node] = 1 - partition[node]
                    
                    classic_solutions.append(best_partition)
                
            elif self.problem_type == 'portfolio':
                # Verwende klassische Portfoliooptimierung
                for x in X_test:
                    n_assets = len(x) // 2
                    returns = x[:n_assets]
                    vols = x[n_assets:]
                    
                    # Einfache mean-variance Optimierung
                    def objective(w):
                        # Maximiere Sharpe-Ratio
                        portfolio_return = np.sum(w * returns)
                        portfolio_risk = np.sqrt(np.sum((w * vols)**2))
                        return -(portfolio_return / portfolio_risk) if portfolio_risk > 0 else 0
                    
                    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
                    bounds = [(0, 1) for _ in range(n_assets)]
                    w0 = np.ones(n_assets) / n_assets
                    
                    result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
                    classic_solutions.append(result.x)
                
            elif self.problem_type == 'chemistry':
                # Verwende L-BFGS-B für molekulare Optimierung
                for x in X_test:
                    n_atoms = len(x) // 4  # 3 für Positionen + 1 für Atomtypen
                    initial_pos = x[:n_atoms*3].reshape(n_atoms, 3)
                    atom_numbers = x[n_atoms*3:] * 10
                    
                    def energy(pos_flat):
                        pos = pos_flat.reshape(-1, 3)
                        energy = 0
                        
                        for i in range(n_atoms):
                            for j in range(i+1, n_atoms):
                                r = np.sqrt(np.sum((pos[i] - pos[j])**2))
                                if r < 0.1:
                                    r = 0.1
                                
                                scale = atom_numbers[i] * atom_numbers[j] / 36
                                
                                # Lennard-Jones
                                energy += scale * 4 * ((1/r)**12 - (1/r)**6)
                        
                        return energy
                    
                    result = minimize(energy, initial_pos.flatten(), method='L-BFGS-B')
                    classic_solutions.append(result.x)
            
            classic_time = time.time() - classic_start
            
            # Evaluiere klassische Lösungen
            classic_scores = [evaluate_solution(sol, x) for sol, x in zip(classic_solutions, X_test)]
            
            # Füge Vergleichsergebnisse hinzu
            results.update({
                'classic_time': classic_time,
                'classic_avg_time': classic_time / n_samples,
                'classic_scores': classic_scores,
                'classic_avg_score': np.mean(classic_scores),
                'speedup': classic_time / qinn_time if qinn_time > 0 else float('inf'),
                'quality_ratio': np.mean(qinn_scores) / np.mean(classic_scores) if np.mean(classic_scores) != 0 else float('inf')
            })
        
        return results
        
    def visualize(self, solution: np.ndarray = None, save_path: str = None):
        """
        Visualisiert die Probleminstanz und optional eine Lösung.
        
        Args:
            solution: Lösung des Problems (Optional)
            save_path: Pfad zum Speichern der Visualisierung (Optional)
        """
        plt.figure(figsize=(10, 8))
        
        if self.problem_type == 'tsp':
            coords = self.problem_params['city_coordinates']
            plt.scatter(coords[:, 0], coords[:, 1], c='blue', s=100, label='Städte')
            
            # Wenn eine Lösung gegeben ist, zeichne Tour
            if solution is not None:
                for i in range(len(solution)):
                    j = (i + 1) % len(solution)
                    plt.plot([coords[solution[i], 0], coords[solution[j], 0]], 
                             [coords[solution[i], 1], coords[solution[j], 1]], 'r-')
                
                plt.title('TSP-Lösung mit QINN')
            else:
                plt.title('TSP-Problem')
                
        elif self.problem_type == 'max_cut':
            adj_matrix = self.problem_params['adjacency_matrix']
            n_nodes = adj_matrix.shape[0]
            
            # Positioniere Knoten kreisförmig
            positions = {}
            for i in range(n_nodes):
                angle = 2 * np.pi * i / n_nodes
                positions[i] = (np.cos(angle), np.sin(angle))
            
            # Zeichne Kanten
            for i in range(n_nodes):
                for j in range(i+1, n_nodes):
                    if adj_matrix[i, j] > 0:
                        plt.plot([positions[i][0], positions[j][0]], 
                                [positions[i][1], positions[j][1]], 'k-', alpha=0.6)
            
            # Zeichne Knoten
            node_colors = ['blue'] * n_nodes
            
            # Wenn eine Lösung gegeben ist, färbe Knoten
            if solution is not None:
                for i in range(n_nodes):
                    node_colors[i] = 'red' if solution[i] == 1 else 'green'
                plt.title('Max-Cut-Lösung mit QINN')
            else:
                plt.title('Max-Cut-Problem')
                
            # Zeichne Knoten mit Farben
            for i in range(n_nodes):
                plt.scatter(positions[i][0], positions[i][1], c=node_colors[i], s=100)
                
        elif self.problem_type == 'portfolio':
            n_assets = self.problem_params.get('n_assets', 10)
            returns = self.problem_params['returns']
            cov_matrix = self.problem_params['covariance']
            
            # Berechne Volatilitäten
            vols = np.sqrt(np.diag(cov_matrix))
            
            # Wenn eine Lösung gegeben ist, visualisiere die Allokation
            if solution is not None:
                # Kreisdiagramm der Allokation
                plt.subplot(1, 2, 1)
                plt.pie(solution, labels=[f'Asset {i+1}' for i in range(n_assets)], autopct='%1.1f%%')
                plt.title('Portfolio-Allokation')
                
                # Risk-Return-Profil
                plt.subplot(1, 2, 2)
                plt.scatter(vols, returns, c='blue', s=100, alpha=0.7)
                
                # Markiere das gewählte Portfolio
                portfolio_return = np.sum(solution * returns)
                portfolio_risk = np.sqrt(solution @ cov_matrix @ solution)
                plt.scatter(portfolio_risk, portfolio_return, c='red', s=200, marker='*')
                
                plt.xlabel('Risiko (Volatilität)')
                plt.ylabel('Rendite')
                plt.title('Risk-Return-Profil')
                
            else:
                # Risk-Return-Profil aller Assets
                plt.scatter(vols, returns, c='blue', s=100, alpha=0.7)
                
                for i in range(n_assets):
                    plt.annotate(f'Asset {i+1}', (vols[i], returns[i]))
                
                plt.xlabel('Risiko (Volatilität)')
                plt.ylabel('Rendite')
                plt.title('Risk-Return-Profil der Assets')
                
        elif self.problem_type == 'chemistry':
            atom_types = self.problem_params['atom_types']
            positions = self.problem_params['initial_positions']
            
            from mpl_toolkits.mplot3d import Axes3D
            
            # 3D-Plot für Molekül
            ax = plt.figure().add_subplot(111, projection='3d')
            
            # Farbkodierung für Atomtypen
            color_map = {'H': 'white', 'C': 'black', 'N': 'blue', 'O': 'red'}
            size_map = {'H': 50, 'C': 100, 'N': 100, 'O': 100}
            
            # Positionen zu plotten (entweder initial oder optimiert)
            pos_to_plot = positions if solution is None else solution.reshape(-1, 3)
            
            # Plotte Atome
            for i, atom in enumerate(atom_types):
                ax.scatter(pos_to_plot[i, 0], pos_to_plot[i, 1], pos_to_plot[i, 2], 
                          c=color_map.get(atom, 'gray'), s=size_map.get(atom, 50), label=atom)
            
            # Entferne doppelte Labels in der Legende
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            
            # Plotte Bindungen (einfache Annäherung: verbinde nahe Atome)
            for i in range(len(atom_types)):
                for j in range(i+1, len(atom_types)):
                    # Berechne Abstand
                    dist = np.sqrt(np.sum((pos_to_plot[i] - pos_to_plot[j])**2))
                    
                    # Wenn Atome nahe genug, zeichne eine Bindung
                    if dist < 2.0:  # Vereinfachter Schwellenwert
                        ax.plot([pos_to_plot[i, 0], pos_to_plot[j, 0]],
                               [pos_to_plot[i, 1], pos_to_plot[j, 1]],
                               [pos_to_plot[i, 2], pos_to_plot[j, 2]], 'k-', alpha=0.6)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            if solution is None:
                plt.title('Molekulare Struktur')
            else:
                plt.title('Optimierte Molekulare Struktur mit QINN')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()


def main():
    """Hauptfunktion zum Testen des QINN für verschiedene Optimierungsprobleme."""
    
    # Ressourcennutzung anzeigen
    print(f"Verfügbare Prozessoren: {multiprocessing.cpu_count()}")
    
    # Testen verschiedener Problemtypen
    problem_types = ['tsp', 'max_cut', 'portfolio', 'chemistry']
    results = {}
    
    for problem_type in problem_types:
        print(f"\n{'='*50}")
        print(f"Testing QINN for {problem_type} problem")
        print(f"{'='*50}")
        
        # Problem-Parameter definieren
        if problem_type == 'tsp':
            n_cities = 10
            problem_params = {
                'n_cities': n_cities,
                'city_coordinates': np.random.rand(n_cities, 2)
            }
        elif problem_type == 'max_cut':
            n_nodes = 10
            problem_params = {'n_nodes': n_nodes}
        elif problem_type == 'portfolio':
            n_assets = 8
            problem_params = {'n_assets': n_assets}
        elif problem_type == 'chemistry':
            n_atoms = 8
            atom_types = np.random.choice(['H', 'C', 'N', 'O'], n_atoms)
            initial_positions = np.random.randn(n_atoms, 3)  # Zufällige 3D-Positionen
            problem_params = {
                'n_atoms': n_atoms,
                'atom_types': atom_types,
                'initial_positions': initial_positions  # Hinzugefügt
            }
        
        # Initialisiere Optimierer
        optimizer = QuantumInspiredOptimizer(problem_type, problem_params)
        
        # Transkurzes Training für Demo-Zwecke
        losses = optimizer.train(epochs=50, verbose=True)
        
        # Visualisiere Trainingsverlauf
        plt.figure()
        plt.plot(losses)
        plt.title(f'Trainingsverlauf für {problem_type}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
        
        # Evaluiere Performance
        eval_results = optimizer.evaluate(n_samples=5, comparison=True)
        results[problem_type] = eval_results
        
        # Generiere und visualisiere eine Lösung
        if problem_type == 'tsp':
            # Erzeugen einer neuen TSP-Instanz
            n_cities = problem_params['n_cities']
            coords = np.random.rand(n_cities, 2)
            
            # Distanzmatrix berechnen
            dist_matrix = np.zeros((n_cities, n_cities))
            for i in range(n_cities):
                for j in range(n_cities):
                    dist_matrix[i, j] = np.sqrt(np.sum((coords[i] - coords[j])**2))
            
            # Problem lösen
            solution = optimizer.solve(dist_matrix.flatten())
            
            # Problem-Parameter aktualisieren für Visualisierung
            optimizer.problem_params['city_coordinates'] = coords
            
        elif problem_type == 'max_cut':
            # Erzeugen einer neuen Max-Cut-Instanz
            n_nodes = problem_params['n_nodes']
            adj_matrix = np.random.randint(0, 2, (n_nodes, n_nodes))
            adj_matrix = np.triu(adj_matrix) + np.triu(adj_matrix, 1).T
            np.fill_diagonal(adj_matrix, 0)
            
            # Problem lösen
            solution = optimizer.solve(adj_matrix.flatten())
            
            # Problem-Parameter aktualisieren für Visualisierung
            optimizer.problem_params['adjacency_matrix'] = adj_matrix
            
        elif problem_type == 'portfolio':
            # Erzeugen einer neuen Portfolio-Instanz
            n_assets = problem_params['n_assets']
            returns = np.random.normal(0.05, 0.1, n_assets)
            temp = np.random.randn(n_assets, n_assets)
            cov = np.dot(temp, temp.transpose()) / 10
            
            # Features zusammenstellen
            features = np.concatenate([returns, np.sqrt(np.diag(cov))])
            
            # Problem lösen
            solution = optimizer.solve(features)
            
            # Problem-Parameter aktualisieren für Visualisierung
            optimizer.problem_params['returns'] = returns
            optimizer.problem_params['covariance'] = cov
            
        elif problem_type == 'chemistry':
            # Erzeugen einer neuen Molekül-Instanz
            n_atoms = problem_params['n_atoms']
            atom_types = problem_params['atom_types']
            positions = np.random.randn(n_atoms, 3)
            
            # Features zusammenstellen
            atom_type_map = {'H': 1, 'C': 6, 'N': 7, 'O': 8}
            atom_numbers = np.array([atom_type_map.get(atom, 0) for atom in atom_types])
            features = np.concatenate([positions.flatten(), atom_numbers / 10])
            
            # Problem lösen
            solution = optimizer.solve(features)
            
            # Problem-Parameter aktualisieren für Visualisierung
            optimizer.problem_params['initial_positions'] = positions
        
        # Visualisierung
        optimizer.visualize(solution)
    
    # Zusammenfassung der Ergebnisse
    print("\n\n" + "="*60)
    print("Performance-Zusammenfassung für QINN vs. Klassische Algorithmen")
    print("="*60)
    
    for problem_type, result in results.items():
        print(f"\n{problem_type.upper()}:")
        print(f"  QINN Zeit: {result['qinn_avg_time']:.4f}s pro Instanz")
        print(f"  Klassische Zeit: {result['classic_avg_time']:.4f}s pro Instanz")
        print(f"  Beschleunigung: {result['speedup']:.2f}x")
        print(f"  QINN Score: {result['qinn_avg_score']:.4f}")
        print(f"  Klassischer Score: {result['classic_avg_score']:.4f}")
        print(f"  Qualitätsverhältnis: {result['quality_ratio']:.2f}")


if __name__ == "__main__":
    main()
