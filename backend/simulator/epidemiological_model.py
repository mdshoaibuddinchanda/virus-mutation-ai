"""
Epidemiological Model - Population-level viral spread simulation
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random
from scipy.integrate import odeint

@dataclass
class Individual:
    """Represents an individual in the population"""
    id: int
    status: str  # 'S', 'I', 'R', 'V' (Susceptible, Infected, Recovered, Vaccinated)
    viral_strain: Optional[str] = None
    infection_time: Optional[int] = None
    recovery_time: Optional[int] = None
    immunity_level: float = 0.0

@dataclass
class ViralStrain:
    """Represents a viral strain with specific properties"""
    id: str
    sequence: str
    transmissibility: float
    virulence: float
    immune_escape: float
    generation: int

class EpidemiologicalModel:
    """Advanced epidemiological simulation with automatic GPU/CPU selection and mutation-aware dynamics"""
    
    def __init__(self, population_size: int = 10000, use_gpu: bool = True):
        self.population_size = population_size
        self.population = []
        self.viral_strains = {}
        self.time_step = 0
        self.history = []
        
        # Model parameters - adjusted for more realistic dynamics
        self.base_transmission_rate = 0.15  # Reduced from 0.3
        self.recovery_rate = 0.05           # Reduced from 0.1 (longer infectious period)
        self.vaccination_rate = 0.005       # Reduced from 0.01 (slower vaccination)
        self.mutation_rate = 0.001
        
        # Initialize universal GPU manager for automatic GPU/CPU selection
        try:
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from utils.gpu_utils import get_universal_gpu_manager
            
            self.gpu_manager = get_universal_gpu_manager()
            self.use_gpu = use_gpu and self.gpu_manager.gpu_available and population_size > 5000
            
            if self.use_gpu:
                # Check GPU for epidemiological simulation
                data_size_mb = population_size * 0.001  # Rough estimate
                self.device = self.gpu_manager.check_and_use_gpu("EpidemiologicalModel", data_size_mb)
                self.gpu_available = self.device.type == 'cuda'
                
                if self.gpu_available:
                    print(f"🦠 EpidemiologicalModel: GPU acceleration for {population_size:,} agents")
                else:
                    print(f"💻 EpidemiologicalModel: Using CPU for {population_size:,} agents")
            else:
                self.gpu_available = False
                print(f"💻 EpidemiologicalModel: Using CPU for {population_size:,} agents")
                
        except ImportError:
            print("⚠️ EpidemiologicalModel: GPU utilities not available, using CPU")
            self.gpu_manager = None
            self.use_gpu = False
            self.gpu_available = False
            self.device = None
        
        self._initialize_population()
    
    def _initialize_population(self):
        """Initialize population with susceptible individuals"""
        self.population = [
            Individual(id=i, status='S') 
            for i in range(self.population_size)
        ]
    
    def add_viral_strain(self, strain: ViralStrain):
        """Add a new viral strain to the simulation"""
        self.viral_strains[strain.id] = strain
    
    def introduce_infection(self, strain_id: str, num_infected: int = 1):
        """Introduce initial infections with specific strain"""
        susceptible = [ind for ind in self.population if ind.status == 'S']
        
        if len(susceptible) < num_infected:
            num_infected = len(susceptible)
        
        infected_individuals = random.sample(susceptible, num_infected)
        
        for individual in infected_individuals:
            individual.status = 'I'
            individual.viral_strain = strain_id
            individual.infection_time = self.time_step
    
    def calculate_transmission_probability(self, infected_strain: str, 
                                         target_individual: Individual) -> float:
        """Calculate transmission probability based on strain and individual immunity"""
        strain = self.viral_strains[infected_strain]
        base_prob = self.base_transmission_rate * strain.transmissibility
        
        # Adjust for immunity
        if target_individual.status == 'V':  # Vaccinated
            base_prob *= (1 - 0.7 + strain.immune_escape * 0.5)
        elif target_individual.immunity_level > 0:  # Previous infection
            base_prob *= (1 - target_individual.immunity_level + strain.immune_escape * 0.3)
        
        return min(1.0, base_prob)
    
    def simulate_step(self):
        """Simulate one time step of the epidemic"""
        new_infections = []
        recoveries = []
        vaccinations = []
        
        # Get current infected individuals
        infected = [ind for ind in self.population if ind.status == 'I']
        susceptible = [ind for ind in self.population if ind.status == 'S']
        
        # Process infections
        for infected_ind in infected:
            strain_id = infected_ind.viral_strain
            
            # Calculate number of contacts - more realistic distribution
            num_contacts = np.random.poisson(3)  # Reduced average contacts
            contacts = random.sample(self.population, 
                                   min(num_contacts, len(self.population)))
            
            for contact in contacts:
                if contact.status == 'S' or contact.status == 'V':
                    trans_prob = self.calculate_transmission_probability(
                        strain_id, contact
                    )
                    
                    if random.random() < trans_prob:
                        new_infections.append((contact, strain_id))
        
        # Apply new infections
        for individual, strain_id in new_infections:
            individual.status = 'I'
            individual.viral_strain = strain_id
            individual.infection_time = self.time_step
        
        # Process recoveries - more realistic recovery timing
        for individual in infected:
            days_infected = self.time_step - individual.infection_time
            if days_infected >= 5:  # Minimum 5 days infection period
                # Recovery probability increases with time
                recovery_prob = self.recovery_rate * (1 + days_infected * 0.1)
                if random.random() < recovery_prob:
                    recoveries.append(individual)
        
        # Apply recoveries
        for individual in recoveries:
            individual.status = 'R'
            individual.immunity_level = 0.8  # 80% immunity after recovery
            individual.viral_strain = None
            individual.recovery_time = self.time_step
        
        # Process vaccinations
        for individual in susceptible:
            if random.random() < self.vaccination_rate:
                vaccinations.append(individual)
        
        # Apply vaccinations
        for individual in vaccinations:
            individual.status = 'V'
            individual.immunity_level = 0.9  # 90% vaccine efficacy
        
        # Record statistics
        stats = self.get_current_stats()
        self.history.append(stats)
        
        self.time_step += 1
        
        return stats
    
    def get_current_stats(self) -> Dict:
        """Get current population statistics"""
        status_counts = {'S': 0, 'I': 0, 'R': 0, 'V': 0}
        strain_counts = {}
        
        for individual in self.population:
            status_counts[individual.status] += 1
            
            if individual.status == 'I' and individual.viral_strain:
                strain = individual.viral_strain
                strain_counts[strain] = strain_counts.get(strain, 0) + 1
        
        return {
            'time_step': self.time_step,
            'susceptible': status_counts['S'],
            'infected': status_counts['I'],
            'recovered': status_counts['R'],
            'vaccinated': status_counts['V'],
            'strain_distribution': strain_counts,
            'total_population': self.population_size
        }
    
    def run_simulation(self, num_steps: int = 365) -> List[Dict]:
        """Run complete epidemiological simulation"""
        for _ in range(num_steps):
            self.simulate_step()
            
            # Stop if no more infected
            if all(ind.status != 'I' for ind in self.population):
                break
        
        return self.history
    
    def sir_model_comparison(self, beta: float = 0.3, gamma: float = 0.1, 
                           days: int = 365) -> Tuple[np.ndarray, np.ndarray]:
        """Compare with classical SIR model"""
        def sir_equations(y, t, beta, gamma):
            S, I, R = y
            N = S + I + R
            dSdt = -beta * S * I / N
            dIdt = beta * S * I / N - gamma * I
            dRdt = gamma * I
            return dSdt, dIdt, dRdt
        
        # Initial conditions
        S0 = self.population_size - 1
        I0 = 1
        R0 = 0
        
        t = np.linspace(0, days, days)
        sol = odeint(sir_equations, [S0, I0, R0], t, args=(beta, gamma))
        
        return t, sol
    
    def calculate_r0(self, strain_id: str) -> float:
        """Calculate basic reproduction number for a strain"""
        strain = self.viral_strains[strain_id]
        
        # Simplified R0 calculation
        contacts_per_day = 5
        transmission_prob = self.base_transmission_rate * strain.transmissibility
        infectious_period = 1 / self.recovery_rate
        
        r0 = contacts_per_day * transmission_prob * infectious_period
        return r0