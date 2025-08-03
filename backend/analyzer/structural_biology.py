"""
Structural Biology Analyzer - AlphaFold, conservation, and stability analysis with dynamic configuration
"""
import numpy as np
import requests
import json
import os
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from Bio.PDB import PDBParser, DSSP, Selection
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import warnings
warnings.filterwarnings('ignore')

# Import dynamic configuration
try:
    from ..utils.dynamic_config import get_dynamic_config_manager
    from ..utils.constants import (get_dynamic_constants, get_amino_acid_properties,
                                   get_interaction_cutoffs, get_conservation_thresholds,
                                   get_stability_thresholds)
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    try:
        from utils.dynamic_config import get_dynamic_config_manager
        from utils.constants import (get_dynamic_constants, get_amino_acid_properties,
                                     get_interaction_cutoffs, get_conservation_thresholds,
                                     get_stability_thresholds)
    except ImportError:
        # Fallback functions if dynamic config not available
        get_dynamic_constants = lambda: None
        get_amino_acid_properties = lambda: {}
        get_interaction_cutoffs = lambda x: {'hydrogen_bond': 3.5, 'hydrophobic': 4.5, 'electrostatic': 6.0}
        get_conservation_thresholds = lambda x: {'high_conservation': 0.8, 'medium_conservation': 0.5, 'low_conservation': 0.2}
        get_stability_thresholds = lambda x: {'destabilizing': -0.8, 'highly_destabilizing': -1.5}

@dataclass
class StructuralFeature:
    """Represents structural features of a protein"""
    position: int
    amino_acid: str
    secondary_structure: str
    solvent_accessibility: float
    conservation_score: float
    stability_change: float

class StructuralBiologyAnalyzer:
    """Advanced structural biology analysis with automatic GPU/CPU selection and AlphaFold integration"""

    def __init__(self, use_gpu: bool = True):
        # Get dynamic configuration
        self.constants = get_dynamic_constants()

        # Use dynamic API URLs if available
        if self.constants:
            api_urls = self.constants.API_URLS
            self.alphafold_base_url = api_urls.get('alphafold_base', "https://alphafold.ebi.ac.uk/api/prediction/")
        else:
            self.alphafold_base_url = "https://alphafold.ebi.ac.uk/api/prediction/"

        self.pdb_parser = PDBParser(QUIET=True)
        self.conservation_matrix = self._load_conservation_matrix()
        
        # Initialize universal GPU manager for automatic GPU/CPU selection
        try:
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from utils.gpu_utils import get_universal_gpu_manager
            
            self.gpu_manager = get_universal_gpu_manager()
            self.use_gpu = use_gpu and self.gpu_manager.gpu_available
            
            # Check GPU for structural analysis tasks
            self.device = self.gpu_manager.check_and_use_gpu("StructuralBiology")
            self.gpu_available = self.device.type == 'cuda'
                
        except ImportError:
            print("⚠️ StructuralBiology: GPU utilities not available, using CPU")
            self.gpu_manager = None
            self.use_gpu = False
            self.gpu_available = False
            self.device = None
    
    def _load_conservation_matrix(self) -> Dict:
        """Load amino acid conservation scoring matrix"""
        # Simplified BLOSUM62-like matrix
        amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                      'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        
        # Create conservation matrix (simplified)
        matrix = {}
        for i, aa1 in enumerate(amino_acids):
            matrix[aa1] = {}
            for j, aa2 in enumerate(amino_acids):
                if i == j:
                    matrix[aa1][aa2] = 1.0
                else:
                    # Simple chemical similarity scoring
                    hydrophobic = {'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W'}
                    polar = {'S', 'T', 'N', 'Q'}
                    charged = {'R', 'K', 'D', 'E'}
                    
                    if (aa1 in hydrophobic and aa2 in hydrophobic) or \
                       (aa1 in polar and aa2 in polar) or \
                       (aa1 in charged and aa2 in charged):
                        matrix[aa1][aa2] = 0.5
                    else:
                        matrix[aa1][aa2] = 0.1
        
        return matrix
    
    def fetch_alphafold_structure(self, uniprot_id: str) -> Optional[Dict]:
        """Fetch protein structure from AlphaFold database"""
        try:
            response = requests.get(f"{self.alphafold_base_url}{uniprot_id}")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"AlphaFold structure not found for {uniprot_id}")
                return None
        except Exception as e:
            print(f"Error fetching AlphaFold structure: {e}")
            return None
    
    def download_pdb_structure(self, uniprot_id: str, output_path: str) -> bool:
        """Download PDB structure file from AlphaFold"""
        try:
            pdb_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
            response = requests.get(pdb_url)
            
            if response.status_code == 200:
                with open(output_path, 'w') as f:
                    f.write(response.text)
                return True
            else:
                return False
        except Exception as e:
            print(f"Error downloading PDB structure: {e}")
            return False
    
    def analyze_secondary_structure(self, pdb_file: str) -> Dict[int, str]:
        """Analyze secondary structure using DSSP"""
        try:
            structure = self.pdb_parser.get_structure("protein", pdb_file)
            model = structure[0]
            
            # Run DSSP (requires DSSP installation)
            dssp = DSSP(model, pdb_file)
            
            secondary_structure = {}
            for key in dssp.keys():
                residue_id = key[1][1]  # Residue number
                ss = dssp[key][2]  # Secondary structure
                
                # Convert DSSP notation to simplified format
                if ss in ['H', 'G', 'I']:
                    ss_simplified = 'H'  # Helix
                elif ss in ['B', 'E']:
                    ss_simplified = 'E'  # Sheet
                else:
                    ss_simplified = 'C'  # Coil
                
                secondary_structure[residue_id] = ss_simplified
            
            return secondary_structure
            
        except Exception as e:
            print(f"Error analyzing secondary structure: {e}")
            return {}
    
    def calculate_solvent_accessibility(self, pdb_file: str) -> Dict[int, float]:
        """Calculate solvent accessible surface area"""
        try:
            structure = self.pdb_parser.get_structure("protein", pdb_file)
            model = structure[0]
            
            # Run DSSP for accessibility
            dssp = DSSP(model, pdb_file)
            
            accessibility = {}
            for key in dssp.keys():
                residue_id = key[1][1]
                acc = dssp[key][3]  # Relative accessibility
                accessibility[residue_id] = acc
            
            return accessibility
            
        except Exception as e:
            print(f"Error calculating solvent accessibility: {e}")
            return {}
    
    def calculate_conservation_score(self, sequence: str, position: int, 
                                   mutation: str) -> float:
        """Calculate conservation score for a mutation"""
        if position >= len(sequence):
            return 0.0
        
        original_aa = sequence[position]
        
        if original_aa in self.conservation_matrix and \
           mutation in self.conservation_matrix[original_aa]:
            return self.conservation_matrix[original_aa][mutation]
        
        return 0.1  # Default low conservation score
    
    def predict_stability_change(self, sequence: str, position: int, 
                               mutation: str) -> float:
        """Predict protein stability change due to mutation"""
        if position >= len(sequence):
            return 0.0
        
        original_aa = sequence[position]
        
        # Simple stability prediction based on amino acid properties
        aa_properties = {
            'A': {'hydrophobicity': 1.8, 'volume': 67, 'flexibility': 0.357},
            'R': {'hydrophobicity': -4.5, 'volume': 148, 'flexibility': 0.529},
            'N': {'hydrophobicity': -3.5, 'volume': 96, 'flexibility': 0.463},
            'D': {'hydrophobicity': -3.5, 'volume': 91, 'flexibility': 0.511},
            'C': {'hydrophobicity': 2.5, 'volume': 86, 'flexibility': 0.346},
            'Q': {'hydrophobicity': -3.5, 'volume': 114, 'flexibility': 0.493},
            'E': {'hydrophobicity': -3.5, 'volume': 109, 'flexibility': 0.497},
            'G': {'hydrophobicity': -0.4, 'volume': 48, 'flexibility': 0.544},
            'H': {'hydrophobicity': -3.2, 'volume': 118, 'flexibility': 0.323},
            'I': {'hydrophobicity': 4.5, 'volume': 124, 'flexibility': 0.462},
            'L': {'hydrophobicity': 3.8, 'volume': 124, 'flexibility': 0.365},
            'K': {'hydrophobicity': -3.9, 'volume': 135, 'flexibility': 0.466},
            'M': {'hydrophobicity': 1.9, 'volume': 124, 'flexibility': 0.295},
            'F': {'hydrophobicity': 2.8, 'volume': 135, 'flexibility': 0.314},
            'P': {'hydrophobicity': -1.6, 'volume': 90, 'flexibility': 0.509},
            'S': {'hydrophobicity': -0.8, 'volume': 73, 'flexibility': 0.507},
            'T': {'hydrophobicity': -0.7, 'volume': 93, 'flexibility': 0.444},
            'W': {'hydrophobicity': -0.9, 'volume': 163, 'flexibility': 0.305},
            'Y': {'hydrophobicity': -1.3, 'volume': 141, 'flexibility': 0.420},
            'V': {'hydrophobicity': 4.2, 'volume': 105, 'flexibility': 0.386}
        }
        
        if original_aa not in aa_properties or mutation not in aa_properties:
            return 0.0
        
        orig_props = aa_properties[original_aa]
        mut_props = aa_properties[mutation]
        
        # Calculate property changes
        hydro_change = abs(orig_props['hydrophobicity'] - mut_props['hydrophobicity'])
        volume_change = abs(orig_props['volume'] - mut_props['volume']) / 100
        flex_change = abs(orig_props['flexibility'] - mut_props['flexibility'])
        
        # Combine changes (negative = destabilizing)
        stability_change = -(hydro_change * 0.4 + volume_change * 0.4 + flex_change * 0.2)
        
        return stability_change
    
    def analyze_protein_features(self, sequence: str, pdb_file: Optional[str] = None) -> List[StructuralFeature]:
        """Comprehensive protein feature analysis"""
        features = []
        
        # Get structural data if PDB file provided
        secondary_structure = {}
        solvent_accessibility = {}
        
        if pdb_file:
            secondary_structure = self.analyze_secondary_structure(pdb_file)
            solvent_accessibility = self.calculate_solvent_accessibility(pdb_file)
        
        # Analyze each position
        for i, aa in enumerate(sequence):
            feature = StructuralFeature(
                position=i,
                amino_acid=aa,
                secondary_structure=secondary_structure.get(i+1, 'C'),
                solvent_accessibility=solvent_accessibility.get(i+1, 0.5),
                conservation_score=1.0,  # Default high conservation
                stability_change=0.0
            )
            features.append(feature)
        
        return features
    
    def analyze_mutation_impact(self, sequence: str, mutations: List[Tuple], 
                              pdb_file: Optional[str] = None) -> Dict:
        """Analyze structural impact of mutations"""
        results = {
            'mutations': [],
            'overall_stability_change': 0.0,
            'conservation_impact': 0.0,
            'structural_regions_affected': []
        }
        
        # Get protein features
        features = self.analyze_protein_features(sequence, pdb_file)
        feature_dict = {f.position: f for f in features}
        
        total_stability_change = 0.0
        total_conservation_impact = 0.0
        
        for pos, from_aa, to_aa in mutations:
            if pos in feature_dict:
                feature = feature_dict[pos]
                
                # Calculate impacts
                stability_change = self.predict_stability_change(sequence, pos, to_aa)
                conservation_score = self.calculate_conservation_score(sequence, pos, to_aa)
                
                mutation_result = {
                    'position': pos,
                    'from_aa': from_aa,
                    'to_aa': to_aa,
                    'secondary_structure': feature.secondary_structure,
                    'solvent_accessibility': feature.solvent_accessibility,
                    'stability_change': stability_change,
                    'conservation_score': conservation_score,
                    'severity': self._classify_mutation_severity(stability_change, conservation_score)
                }
                
                results['mutations'].append(mutation_result)
                total_stability_change += stability_change
                total_conservation_impact += (1 - conservation_score)
        
        results['overall_stability_change'] = total_stability_change
        results['conservation_impact'] = total_conservation_impact
        
        return results
    
    def _classify_mutation_severity(self, stability_change: float, 
                                  conservation_score: float) -> str:
        """Classify mutation severity based on stability and conservation"""
        if stability_change < -1.0 or conservation_score < 0.2:
            return "High"
        elif stability_change < -0.5 or conservation_score < 0.5:
            return "Medium"
        else:
            return "Low"