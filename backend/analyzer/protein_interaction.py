"""
Protein-Protein Interaction Analyzer - 3D docking and binding energy calculations with dynamic configuration
"""
import numpy as np
import os
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from Bio.PDB import PDBParser, NeighborSearch, Selection
import warnings
warnings.filterwarnings('ignore')

# Import dynamic configuration
try:
    from ..utils.dynamic_config import get_dynamic_config_manager
    from ..utils.constants import (get_dynamic_constants, get_amino_acid_properties,
                                   get_interaction_cutoffs)
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    try:
        from utils.dynamic_config import get_dynamic_config_manager
        from utils.constants import (get_dynamic_constants, get_amino_acid_properties,
                                     get_interaction_cutoffs)
    except ImportError:
        # Fallback functions if dynamic config not available
        get_dynamic_constants = lambda: None
        get_amino_acid_properties = lambda: {}
        get_interaction_cutoffs = lambda x: {'hydrogen_bond': 3.5, 'hydrophobic': 4.5, 'electrostatic': 6.0}

@dataclass
class InteractionSite:
    """Represents a protein-protein interaction site"""
    residue_id: int
    amino_acid: str
    interaction_type: str  # 'hydrogen_bond', 'hydrophobic', 'electrostatic'
    distance: float
    energy_contribution: float

@dataclass
class DockingResult:
    """Results from protein docking simulation"""
    binding_energy: float
    interaction_sites: List[InteractionSite]
    interface_area: float
    confidence_score: float
    rmsd: float

class ProteinInteractionAnalyzer:
    """Advanced protein-protein interaction analysis and docking with automatic GPU/CPU selection"""
    
    def __init__(self, use_gpu: bool = True):
        self.pdb_parser = PDBParser(QUIET=True)

        # Get dynamic configuration
        self.constants = get_dynamic_constants()

        # Use dynamic interaction cutoffs if available
        if self.constants:
            self.interaction_cutoffs = get_interaction_cutoffs("standard")
            aa_groups = self.constants.AMINO_ACID_GROUPS
            self.aa_properties = {
                'hydrophobic': aa_groups.get('hydrophobic', {'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W', 'P'}),
                'polar': aa_groups.get('polar', {'S', 'T', 'N', 'Q', 'C'}),
                'positive': aa_groups.get('positive', {'R', 'K', 'H'}),
                'negative': aa_groups.get('negative', {'D', 'E'}),
                'aromatic': aa_groups.get('aromatic', {'F', 'Y', 'W', 'H'})
            }
        else:
            # Fallback to static values
            self.interaction_cutoffs = {
                'hydrogen_bond': 3.5,
                'hydrophobic': 4.5,
                'electrostatic': 6.0
            }
            self.aa_properties = {
                'hydrophobic': {'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W', 'P'},
                'polar': {'S', 'T', 'N', 'Q', 'C'},
                'positive': {'R', 'K', 'H'},
                'negative': {'D', 'E'},
                'aromatic': {'F', 'Y', 'W', 'H'}
            }
        
        # Initialize universal GPU manager for automatic GPU/CPU selection
        try:
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from utils.gpu_utils import get_universal_gpu_manager
            
            self.gpu_manager = get_universal_gpu_manager()
            self.use_gpu = use_gpu and self.gpu_manager.gpu_available
            
            # Check GPU for protein interaction analysis
            self.device = self.gpu_manager.check_and_use_gpu("ProteinInteraction")
            self.gpu_available = self.device.type == 'cuda'
                
        except ImportError:
            print("⚠️ ProteinInteraction: GPU utilities not available, using CPU")
            self.gpu_manager = None
            self.use_gpu = False
            self.gpu_available = False
            self.device = None
    
    def load_protein_structure(self, pdb_file: str, chain_id: str = 'A'):
        """Load protein structure from PDB file"""
        try:
            structure = self.pdb_parser.get_structure("protein", pdb_file)
            return structure[0][chain_id]  # First model, specified chain
        except Exception as e:
            print(f"Error loading protein structure: {e}")
            return None
    
    def calculate_interface_residues(self, protein1, protein2, 
                                   cutoff: float = 5.0) -> Tuple[List, List]:
        """Identify interface residues between two proteins"""
        # Get all atoms from both proteins
        atoms1 = Selection.unfold_entities(protein1, 'A')
        atoms2 = Selection.unfold_entities(protein2, 'A')
        
        # Create neighbor search for protein2
        ns = NeighborSearch(atoms2)
        
        interface_residues1 = set()
        interface_residues2 = set()
        
        # Find interface residues
        for atom1 in atoms1:
            neighbors = ns.search(atom1.coord, cutoff)
            if neighbors:
                interface_residues1.add(atom1.get_parent())
                for atom2 in neighbors:
                    interface_residues2.add(atom2.get_parent())
        
        return list(interface_residues1), list(interface_residues2)
    
    def analyze_interaction_type(self, residue1, residue2, distance: float) -> str:
        """Determine interaction type between two residues"""
        aa1 = residue1.get_resname()
        aa2 = residue2.get_resname()
        
        # Convert 3-letter to 1-letter code
        aa_conversion = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
        }
        
        aa1_single = aa_conversion.get(aa1, 'X')
        aa2_single = aa_conversion.get(aa2, 'X')
        
        # Determine interaction type
        if distance <= self.interaction_cutoffs['hydrogen_bond']:
            if ((aa1_single in self.aa_properties['polar'] or 
                 aa1_single in self.aa_properties['positive'] or 
                 aa1_single in self.aa_properties['negative']) and
                (aa2_single in self.aa_properties['polar'] or 
                 aa2_single in self.aa_properties['positive'] or 
                 aa2_single in self.aa_properties['negative'])):
                return 'hydrogen_bond'
        
        if distance <= self.interaction_cutoffs['electrostatic']:
            if ((aa1_single in self.aa_properties['positive'] and 
                 aa2_single in self.aa_properties['negative']) or
                (aa1_single in self.aa_properties['negative'] and 
                 aa2_single in self.aa_properties['positive'])):
                return 'electrostatic'
        
        if distance <= self.interaction_cutoffs['hydrophobic']:
            if (aa1_single in self.aa_properties['hydrophobic'] and 
                aa2_single in self.aa_properties['hydrophobic']):
                return 'hydrophobic'
        
        return 'van_der_waals'
    
    def calculate_binding_energy(self, interaction_sites: List[InteractionSite]) -> float:
        """Calculate approximate binding energy from interaction sites"""
        energy_weights = {
            'hydrogen_bond': -2.5,
            'electrostatic': -3.0,
            'hydrophobic': -1.5,
            'van_der_waals': -0.5
        }
        
        total_energy = 0.0
        
        for site in interaction_sites:
            base_energy = energy_weights.get(site.interaction_type, -0.5)
            
            # Distance-dependent energy scaling
            distance_factor = max(0.1, 1.0 - (site.distance - 2.0) / 4.0)
            
            site.energy_contribution = base_energy * distance_factor
            total_energy += site.energy_contribution
        
        return total_energy
    
    def perform_docking_analysis(self, protein1_pdb: str, protein2_pdb: str,
                               chain1: str = 'A', chain2: str = 'B') -> DockingResult:
        """Perform protein-protein docking analysis"""
        
        # Load protein structures
        protein1 = self.load_protein_structure(protein1_pdb, chain1)
        protein2 = self.load_protein_structure(protein2_pdb, chain2)
        
        if not protein1 or not protein2:
            return DockingResult(0.0, [], 0.0, 0.0, 0.0)
        
        # Find interface residues
        interface1, interface2 = self.calculate_interface_residues(protein1, protein2)
        
        # Analyze interactions
        interaction_sites = []
        
        for res1 in interface1:
            for res2 in interface2:
                # Calculate minimum distance between residues
                min_distance = float('inf')
                for atom1 in res1:
                    for atom2 in res2:
                        distance = np.linalg.norm(atom1.coord - atom2.coord)
                        min_distance = min(min_distance, distance)
                
                if min_distance <= 6.0:  # Interaction cutoff
                    interaction_type = self.analyze_interaction_type(res1, res2, min_distance)
                    
                    site = InteractionSite(
                        residue_id=res1.id[1],
                        amino_acid=res1.get_resname(),
                        interaction_type=interaction_type,
                        distance=min_distance,
                        energy_contribution=0.0  # Will be calculated later
                    )
                    interaction_sites.append(site)
        
        # Calculate binding energy
        binding_energy = self.calculate_binding_energy(interaction_sites)
        
        # Calculate interface area (simplified)
        interface_area = len(interface1) * 20.0  # Approximate area per residue
        
        # Calculate confidence score based on number of interactions
        confidence_score = min(1.0, len(interaction_sites) / 20.0)
        
        # Mock RMSD calculation
        rmsd = np.random.uniform(1.0, 3.0)
        
        return DockingResult(
            binding_energy=binding_energy,
            interaction_sites=interaction_sites,
            interface_area=interface_area,
            confidence_score=confidence_score,
            rmsd=rmsd
        )
    
    def analyze_mutation_effect_on_binding(self, original_sequence: str,
                                         mutations: List[Tuple],
                                         protein1_pdb: str,
                                         protein2_pdb: str) -> Dict:
        """Analyze how mutations affect protein-protein binding"""
        
        # Perform original docking
        original_docking = self.perform_docking_analysis(protein1_pdb, protein2_pdb)
        
        # Simulate mutated binding (simplified approach)
        mutation_effects = []
        
        for pos, from_aa, to_aa in mutations:
            # Calculate mutation effect on binding
            effect = self._calculate_mutation_binding_effect(
                pos, from_aa, to_aa, original_docking.interaction_sites
            )
            
            mutation_effects.append({
                'position': pos,
                'from_aa': from_aa,
                'to_aa': to_aa,
                'binding_energy_change': effect['energy_change'],
                'affected_interactions': effect['affected_interactions'],
                'severity': effect['severity']
            })
        
        # Calculate overall effect
        total_energy_change = sum(m['binding_energy_change'] for m in mutation_effects)
        
        return {
            'original_binding_energy': original_docking.binding_energy,
            'predicted_binding_energy': original_docking.binding_energy + total_energy_change,
            'energy_change': total_energy_change,
            'mutation_effects': mutation_effects,
            'binding_affinity_change': self._classify_binding_change(total_energy_change)
        }
    
    def _calculate_mutation_binding_effect(self, position: int, from_aa: str, 
                                         to_aa: str, interaction_sites: List) -> Dict:
        """Calculate effect of single mutation on binding"""
        
        # Find affected interaction sites
        affected_sites = [site for site in interaction_sites 
                         if site.residue_id == position]
        
        if not affected_sites:
            return {
                'energy_change': 0.0,
                'affected_interactions': 0,
                'severity': 'None'
            }
        
        # Calculate energy change based on amino acid properties
        aa_conversion = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
        }
        
        from_single = aa_conversion.get(from_aa, from_aa)
        to_single = aa_conversion.get(to_aa, to_aa)
        
        energy_change = 0.0
        
        for site in affected_sites:
            # Calculate change based on interaction type and amino acid change
            if site.interaction_type == 'electrostatic':
                if ((from_single in self.aa_properties['positive'] and 
                     to_single not in self.aa_properties['positive']) or
                    (from_single in self.aa_properties['negative'] and 
                     to_single not in self.aa_properties['negative'])):
                    energy_change += 2.0  # Loss of electrostatic interaction
            
            elif site.interaction_type == 'hydrophobic':
                if ((from_single in self.aa_properties['hydrophobic'] and 
                     to_single not in self.aa_properties['hydrophobic']) or
                    (from_single not in self.aa_properties['hydrophobic'] and 
                     to_single in self.aa_properties['hydrophobic'])):
                    energy_change += 1.0  # Change in hydrophobic interaction
        
        # Classify severity
        if energy_change > 3.0:
            severity = 'High'
        elif energy_change > 1.0:
            severity = 'Medium'
        elif energy_change > 0.0:
            severity = 'Low'
        else:
            severity = 'None'
        
        return {
            'energy_change': energy_change,
            'affected_interactions': len(affected_sites),
            'severity': severity
        }
    
    def _classify_binding_change(self, energy_change: float) -> str:
        """Classify overall binding affinity change"""
        if energy_change > 5.0:
            return "Significantly Weakened"
        elif energy_change > 2.0:
            return "Moderately Weakened"
        elif energy_change > 0.5:
            return "Slightly Weakened"
        elif energy_change > -0.5:
            return "No Significant Change"
        elif energy_change > -2.0:
            return "Slightly Strengthened"
        else:
            return "Significantly Strengthened"
    
    def generate_interaction_network(self, docking_results: List[DockingResult]) -> Dict:
        """Generate protein interaction network from multiple docking results"""
        network = {
            'nodes': [],
            'edges': [],
            'statistics': {}
        }
        
        # Create nodes for each unique protein
        proteins = set()
        for i, result in enumerate(docking_results):
            proteins.add(f"Protein_{i}_A")
            proteins.add(f"Protein_{i}_B")
        
        network['nodes'] = [{'id': protein, 'type': 'protein'} for protein in proteins]
        
        # Create edges for interactions
        for i, result in enumerate(docking_results):
            if result.binding_energy < -1.0:  # Significant binding
                edge = {
                    'source': f"Protein_{i}_A",
                    'target': f"Protein_{i}_B",
                    'binding_energy': result.binding_energy,
                    'confidence': result.confidence_score,
                    'interaction_count': len(result.interaction_sites)
                }
                network['edges'].append(edge)
        
        # Calculate network statistics
        network['statistics'] = {
            'total_proteins': len(network['nodes']),
            'total_interactions': len(network['edges']),
            'average_binding_energy': np.mean([e['binding_energy'] for e in network['edges']]) if network['edges'] else 0,
            'network_density': len(network['edges']) / (len(network['nodes']) * (len(network['nodes']) - 1) / 2) if len(network['nodes']) > 1 else 0
        }
        
        return network