"""
Protein 3D Visualization - py3Dmol integration for interactive protein visualization
"""
import py3Dmol
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import requests
from Bio.PDB import PDBParser
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MutationVisualization:
    """Configuration for mutation visualization"""
    position: int
    original_aa: str
    mutated_aa: str
    color: str
    size: float
    label: str

@dataclass
class VisualizationStyle:
    """3D visualization style configuration"""
    representation: str  # 'cartoon', 'stick', 'sphere', 'surface'
    color_scheme: str   # 'spectrum', 'chain', 'residue', 'custom'
    background_color: str
    highlight_mutations: bool
    show_labels: bool
    transparency: float

class Protein3DVisualizer:
    """Advanced 3D protein visualization with mutation highlighting and automatic GPU/CPU selection"""
    
    def __init__(self, use_gpu: bool = True):
        self.pdb_parser = PDBParser(QUIET=True)
        self.default_style = VisualizationStyle(
            representation='cartoon',
            color_scheme='spectrum',
            background_color='white',
            highlight_mutations=True,
            show_labels=True,
            transparency=0.0
        )
        
        # Initialize GPU support (optional)
        self.use_gpu = use_gpu
        self.gpu_manager = None
        self.gpu_available = False
        self.device = None
        
        if use_gpu:
            try:
                import sys
                import os
                sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
                from utils.gpu_utils import get_universal_gpu_manager
                
                self.gpu_manager = get_universal_gpu_manager()
                self.gpu_available = self.gpu_manager.gpu_available
                self.device = self.gpu_manager.check_and_use_gpu("Protein3D")
                
                if self.gpu_available:
                    print(f"🧬 Protein3D: Using GPU acceleration")
                else:
                    print("💻 Protein3D: Using CPU")
                    
            except Exception as e:
                print(f"⚠️ Protein3D: GPU utilities not available ({e}), using CPU")
                self.gpu_manager = None
                self.use_gpu = False
                self.gpu_available = False
                self.device = None
        else:
            print("💻 Protein3D: GPU disabled, using CPU")
        
        # Color schemes for different visualization types
        self.color_schemes = {
            'mutation_impact': {
                'high': '#FF0000',      # Red
                'medium': '#FFA500',    # Orange
                'low': '#FFFF00',       # Yellow
                'neutral': '#00FF00'    # Green
            },
            'conservation': {
                'highly_conserved': '#000080',  # Navy
                'conserved': '#0000FF',         # Blue
                'variable': '#00FFFF',          # Cyan
                'highly_variable': '#FF00FF'   # Magenta
            },
            'secondary_structure': {
                'helix': '#FF0000',     # Red
                'sheet': '#00FF00',     # Green
                'coil': '#0000FF'       # Blue
            }
        }
    
    def create_mock_pdb_structure(self, sequence: str, structure_type: str = 'helix') -> str:
        """Create a mock PDB structure for visualization"""
        pdb_lines = [
            "HEADER    MOCK PROTEIN STRUCTURE",
            f"TITLE     {structure_type.upper()} STRUCTURE FOR SEQUENCE LENGTH {len(sequence)}",
            "REMARK   This is a mock structure for visualization purposes"
        ]
        
        # Amino acid 3-letter codes
        aa_3letter = {
            'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
            'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
            'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
            'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
        }
        
        atom_serial = 1
        
        for i, aa in enumerate(sequence):
            residue_name = aa_3letter.get(aa, 'UNK')
            residue_num = i + 1
            
            # Generate coordinates based on structure type
            if structure_type == 'helix':
                # Alpha helix parameters
                x = 2.3 * np.cos(i * 100 * np.pi / 180)  # 100° per residue
                y = 2.3 * np.sin(i * 100 * np.pi / 180)
                z = i * 1.5  # 1.5 Å rise per residue
            
            elif structure_type == 'sheet':
                # Beta sheet parameters
                x = i * 3.5  # Extended conformation
                y = 0.0 if i % 2 == 0 else 2.0  # Alternating positions
                z = 0.0
            
            else:  # random coil
                x = np.random.uniform(-10, 10)
                y = np.random.uniform(-10, 10)
                z = np.random.uniform(-10, 10)
            
            # Add backbone atoms (N, CA, C, O)
            atoms = [
                ('N', x - 0.5, y, z, 'N'),
                ('CA', x, y, z, 'C'),
                ('C', x + 0.5, y, z + 0.5, 'C'),
                ('O', x + 1.0, y, z + 1.0, 'O')
            ]
            
            for atom_name, ax, ay, az, element in atoms:
                pdb_line = (f"ATOM  {atom_serial:5d}  {atom_name:<4s}{residue_name} A"
                           f"{residue_num:4d}    {ax:8.3f}{ay:8.3f}{az:8.3f}"
                           f"  1.00 20.00           {element:>2s}")
                pdb_lines.append(pdb_line)
                atom_serial += 1
        
        pdb_lines.append("END")
        return '\n'.join(pdb_lines)
    
    def create_basic_viewer(self, width: int = 800, height: int = 600) -> py3Dmol.view:
        """Create a basic py3Dmol viewer with proper initialization"""
        viewer = py3Dmol.view(width=width, height=height)
        viewer.setBackgroundColor(self.default_style.background_color)
        
        # Set up proper viewing parameters
        viewer.setViewStyle({'style': 'outline', 'color': 'black', 'width': 0.1})
        
        return viewer
    
    def add_protein_structure(self, viewer: py3Dmol.view, pdb_data: str,
                            style: Optional[VisualizationStyle] = None) -> py3Dmol.view:
        """Add protein structure to viewer with specified style"""
        if style is None:
            style = self.default_style
        
        # Add model to viewer
        viewer.addModel(pdb_data, 'pdb')
        
        # Apply visualization style
        style_dict = self._get_style_dict(style)
        viewer.setStyle(style_dict)
        
        # Ensure proper zoom and centering
        viewer.zoomTo()
        viewer.center()
        
        return viewer
    
    def highlight_mutations(self, viewer: py3Dmol.view, mutations: List[MutationVisualization],
                          model_index: int = 0) -> py3Dmol.view:
        """Highlight specific mutations in the protein structure"""
        
        for mutation in mutations:
            # Create selection for the mutated residue
            selection = {'resi': mutation.position, 'model': model_index}
            
            # Apply highlighting style
            highlight_style = {
                'stick': {
                    'color': mutation.color,
                    'radius': mutation.size
                },
                'sphere': {
                    'color': mutation.color,
                    'radius': mutation.size * 2
                }
            }
            
            viewer.addStyle(selection, highlight_style)
            
            # Add label if requested
            if mutation.label:
                viewer.addLabel(mutation.label, {
                    'position': selection,
                    'backgroundColor': mutation.color,
                    'fontColor': 'white',
                    'fontSize': 12
                })
        
        return viewer
    
    def create_mutation_comparison_view(self, original_sequence: str, 
                                      mutated_sequence: str,
                                      mutations: List[Tuple[int, str, str]],
                                      width: int = 1200, height: int = 600) -> py3Dmol.view:
        """Create side-by-side comparison of original and mutated structures"""
        
        viewer = py3Dmol.view(width=width, height=height)
        viewer.setBackgroundColor('white')
        
        # Create mock structures
        original_pdb = self.create_mock_pdb_structure(original_sequence)
        mutated_pdb = self.create_mock_pdb_structure(mutated_sequence)
        
        # Add original structure (left side)
        viewer.addModel(original_pdb, 'pdb')
        viewer.setStyle({'model': 0}, {'cartoon': {'color': 'blue', 'opacity': 0.8}})
        
        # Add mutated structure (right side, shifted)
        viewer.addModel(mutated_pdb, 'pdb')
        viewer.setStyle({'model': 1}, {'cartoon': {'color': 'red', 'opacity': 0.8}})
        
        # Shift the second model to the right
        viewer.translate(20, 0, 0, {'model': 1})
        
        # Highlight mutation sites
        mutation_visualizations = []
        for pos, from_aa, to_aa in mutations:
            mutation_viz = MutationVisualization(
                position=pos + 1,  # PDB numbering starts at 1
                original_aa=from_aa,
                mutated_aa=to_aa,
                color='yellow',
                size=0.5,
                label=f"{from_aa}{pos+1}{to_aa}"
            )
            mutation_visualizations.append(mutation_viz)
        
        # Highlight mutations in both models
        self.highlight_mutations(viewer, mutation_visualizations, model_index=0)
        self.highlight_mutations(viewer, mutation_visualizations, model_index=1)
        
        # Add title labels
        viewer.addLabel("Original", {
            'position': {'x': -10, 'y': 0, 'z': 20},
            'backgroundColor': 'blue',
            'fontColor': 'white',
            'fontSize': 16
        })
        
        viewer.addLabel("Mutated", {
            'position': {'x': 30, 'y': 0, 'z': 20},
            'backgroundColor': 'red',
            'fontColor': 'white',
            'fontSize': 16
        })
        
        viewer.zoomTo()
        return viewer
    
    def create_conservation_view(self, sequence: str, conservation_scores: List[float],
                               width: int = 800, height: int = 600,
                               structure_type: str = 'helix', color_scheme: str = 'conservation',
                               visual_style: str = 'cartoon') -> py3Dmol.view:
        """Create visualization colored by conservation scores"""

        viewer = py3Dmol.view(width=width, height=height)
        pdb_data = self.create_mock_pdb_structure(sequence, structure_type)
        viewer.addModel(pdb_data, 'pdb')

        # Color residues by conservation score
        for i, score in enumerate(conservation_scores):
            residue_num = i + 1

            # Map conservation score to color
            if score > 0.8:
                color = self.color_schemes['conservation']['highly_conserved']
            elif score > 0.6:
                color = self.color_schemes['conservation']['conserved']
            elif score > 0.4:
                color = self.color_schemes['conservation']['variable']
            else:
                color = self.color_schemes['conservation']['highly_variable']

            selection = {'resi': residue_num}
            style_dict = {visual_style: {'color': color}}
            viewer.setStyle(selection, style_dict)

        # Add color legend
        self._add_conservation_legend(viewer)

        viewer.zoomTo()
        return viewer
    
    def create_binding_site_view(self, sequence: str, binding_sites: List[Tuple[int, int]],
                               width: int = 800, height: int = 600,
                               structure_type: str = 'cartoon', color_scheme: str = 'chain') -> py3Dmol.view:
        """Create visualization highlighting binding sites"""

        viewer = py3Dmol.view(width=width, height=height)
        pdb_data = self.create_mock_pdb_structure(sequence, structure_type)
        viewer.addModel(pdb_data, 'pdb')

        # Apply base color scheme
        base_color = self._get_base_color_for_scheme(color_scheme)
        style_dict = {structure_type: {'color': base_color}}
        viewer.setStyle(style_dict)
        
        # Highlight binding sites
        for i, (start, end) in enumerate(binding_sites):
            color = ['red', 'blue', 'green', 'orange', 'purple'][i % 5]
            
            selection = {'resi': f"{start}-{end}"}
            viewer.addStyle(selection, {
                'cartoon': {'color': color},
                'stick': {'color': color, 'radius': 0.3}
            })
            
            # Add label for binding site
            center_pos = (start + end) // 2
            viewer.addLabel(f"Binding Site {i+1}", {
                'position': {'resi': center_pos},
                'backgroundColor': color,
                'fontColor': 'white',
                'fontSize': 12
            })
        
        viewer.zoomTo()
        return viewer

    def _get_base_color_for_scheme(self, color_scheme: str) -> str:
        """Get base color based on color scheme"""
        color_map = {
            'spectrum': 'spectrum',
            'chain': 'lightblue',
            'residue': 'green',
            'hydrophobicity': 'orange',
            'charge': 'red'
        }
        return color_map.get(color_scheme, 'lightgray')

    def create_secondary_structure_view(self, sequence: str,
                                      secondary_structure: List[str],
                                      width: int = 800, height: int = 600) -> py3Dmol.view:
        """Create visualization colored by secondary structure"""
        
        viewer = py3Dmol.view(width=width, height=height)
        pdb_data = self.create_mock_pdb_structure(sequence)
        viewer.addModel(pdb_data, 'pdb')
        
        # Color by secondary structure
        for i, ss in enumerate(secondary_structure):
            residue_num = i + 1
            color = self.color_schemes['secondary_structure'].get(ss.lower(), 'gray')
            
            selection = {'resi': residue_num}
            viewer.setStyle(selection, {'cartoon': {'color': color}})
        
        # Add legend
        self._add_secondary_structure_legend(viewer)
        
        viewer.zoomTo()
        return viewer
    
    def create_interactive_mutation_explorer(self, sequence: str,
                                           all_mutations: List[Dict],
                                           width: int = 1000, height: int = 700,
                                           structure_type: str = 'cartoon', color_scheme: str = 'spectrum') -> py3Dmol.view:
        """Create interactive explorer for multiple mutations"""

        viewer = py3Dmol.view(width=width, height=height)
        pdb_data = self.create_mock_pdb_structure(sequence, structure_type)
        viewer.addModel(pdb_data, 'pdb')

        # Base structure with color scheme
        base_color = self._get_base_color_for_scheme(color_scheme)
        style_dict = {structure_type: {'color': base_color, 'opacity': 0.7}}
        viewer.setStyle(style_dict)
        
        # Add mutations with different colors based on impact
        for mutation in all_mutations:
            pos = mutation.get('position', 0)
            impact = mutation.get('impact', 'low')
            from_aa = mutation.get('from_aa', 'X')
            to_aa = mutation.get('to_aa', 'X')
            
            color = self.color_schemes['mutation_impact'].get(impact, 'gray')
            
            selection = {'resi': pos + 1}
            viewer.addStyle(selection, {
                'stick': {'color': color, 'radius': 0.4},
                'sphere': {'color': color, 'radius': 1.0, 'opacity': 0.6}
            })
            
            # Add clickable label
            viewer.addLabel(f"{from_aa}{pos+1}{to_aa}", {
                'position': {'resi': pos + 1},
                'backgroundColor': color,
                'fontColor': 'white',
                'fontSize': 10,
                'clickable': True
            })
        
        # Add impact legend
        self._add_impact_legend(viewer)
        
        viewer.zoomTo()
        return viewer
    
    def create_protein_surface_view(self, sequence: str,
                                  surface_properties: Optional[Dict] = None,
                                  width: int = 800, height: int = 600,
                                  structure_type: str = 'cartoon', color_scheme: str = 'spectrum') -> py3Dmol.view:
        """Create protein surface visualization"""

        viewer = py3Dmol.view(width=width, height=height)
        pdb_data = self.create_mock_pdb_structure(sequence, structure_type)
        viewer.addModel(pdb_data, 'pdb')
        
        # Surface representation
        surface_style = {
            'surface': {
                'opacity': 0.8,
                'colorscheme': surface_properties.get('colorscheme', 'RdYlBu') if surface_properties else 'RdYlBu'
            }
        }
        
        viewer.setStyle(surface_style)
        
        # Add cartoon backbone for reference
        viewer.addStyle({'cartoon': {'color': 'gray', 'opacity': 0.3}})
        
        viewer.zoomTo()
        return viewer
    
    def _get_style_dict(self, style: VisualizationStyle) -> Dict:
        """Convert VisualizationStyle to py3Dmol style dictionary"""
        style_dict = {}
        
        if style.representation == 'cartoon':
            style_dict['cartoon'] = {
                'color': style.color_scheme,
                'opacity': 1.0 - style.transparency
            }
        elif style.representation == 'stick':
            style_dict['stick'] = {
                'color': style.color_scheme,
                'radius': 0.3
            }
        elif style.representation == 'sphere':
            style_dict['sphere'] = {
                'color': style.color_scheme,
                'radius': 1.0
            }
        elif style.representation == 'surface':
            style_dict['surface'] = {
                'color': style.color_scheme,
                'opacity': 1.0 - style.transparency
            }
        
        return style_dict
    
    def _add_conservation_legend(self, viewer: py3Dmol.view):
        """Add conservation color legend to viewer"""
        legend_items = [
            ("Highly Conserved", self.color_schemes['conservation']['highly_conserved']),
            ("Conserved", self.color_schemes['conservation']['conserved']),
            ("Variable", self.color_schemes['conservation']['variable']),
            ("Highly Variable", self.color_schemes['conservation']['highly_variable'])
        ]
        
        for i, (label, color) in enumerate(legend_items):
            viewer.addLabel(label, {
                'position': {'x': -20, 'y': 15 - i * 3, 'z': 0},
                'backgroundColor': color,
                'fontColor': 'white',
                'fontSize': 10
            })
    
    def _add_secondary_structure_legend(self, viewer: py3Dmol.view):
        """Add secondary structure color legend to viewer"""
        legend_items = [
            ("Helix", self.color_schemes['secondary_structure']['helix']),
            ("Sheet", self.color_schemes['secondary_structure']['sheet']),
            ("Coil", self.color_schemes['secondary_structure']['coil'])
        ]
        
        for i, (label, color) in enumerate(legend_items):
            viewer.addLabel(label, {
                'position': {'x': -20, 'y': 10 - i * 3, 'z': 0},
                'backgroundColor': color,
                'fontColor': 'white',
                'fontSize': 10
            })
    
    def _add_impact_legend(self, viewer: py3Dmol.view):
        """Add mutation impact color legend to viewer"""
        legend_items = [
            ("High Impact", self.color_schemes['mutation_impact']['high']),
            ("Medium Impact", self.color_schemes['mutation_impact']['medium']),
            ("Low Impact", self.color_schemes['mutation_impact']['low']),
            ("Neutral", self.color_schemes['mutation_impact']['neutral'])
        ]
        
        for i, (label, color) in enumerate(legend_items):
            viewer.addLabel(label, {
                'position': {'x': -25, 'y': 15 - i * 3, 'z': 0},
                'backgroundColor': color,
                'fontColor': 'white',
                'fontSize': 10
            })
    
    def export_visualization_html(self, viewer: py3Dmol.view, filename: str) -> str:
        """Export visualization as standalone HTML file"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Protein 3D Visualization</title>
            <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
        </head>
        <body>
            <div id="viewer" style="width: 100%; height: 600px;"></div>
            <script>
                {viewer.js()}
            </script>
        </body>
        </html>
        """
        
        with open(filename, 'w') as f:
            f.write(html_content)
        
        return filename
    
    def create_animation_frames(self, sequence: str, mutation_trajectory: List[List[Tuple]],
                              width: int = 800, height: int = 600) -> List[py3Dmol.view]:
        """Create animation frames showing mutation progression"""
        frames = []
        
        for frame_mutations in mutation_trajectory:
            viewer = py3Dmol.view(width=width, height=height)
            pdb_data = self.create_mock_pdb_structure(sequence)
            viewer.addModel(pdb_data, 'pdb')
            
            # Base structure
            viewer.setStyle({'cartoon': {'color': 'lightgray'}})
            
            # Highlight current mutations
            mutation_visualizations = []
            for pos, from_aa, to_aa in frame_mutations:
                mutation_viz = MutationVisualization(
                    position=pos + 1,
                    original_aa=from_aa,
                    mutated_aa=to_aa,
                    color='red',
                    size=0.5,
                    label=f"{from_aa}{pos+1}{to_aa}"
                )
                mutation_visualizations.append(mutation_viz)
            
            self.highlight_mutations(viewer, mutation_visualizations)
            viewer.zoomTo()
            frames.append(viewer)
        
        return frames


# Utility functions for easy integration
def create_simple_protein_view(sequence: str, mutations: List[Tuple] = None,
                             width: int = 800, height: int = 600,
                             structure_type: str = 'helix', color_scheme: str = 'spectrum',
                             visual_style: str = 'cartoon') -> py3Dmol.view:
    """Create a simple protein visualization with optional mutations"""
    visualizer = Protein3DVisualizer()
    viewer = visualizer.create_basic_viewer(width, height)

    pdb_data = visualizer.create_mock_pdb_structure(sequence, structure_type)
    viewer.addModel(pdb_data, 'pdb')

    # Apply color scheme and visual style
    if color_scheme == 'spectrum':
        viewer.setStyle({visual_style: {'colorscheme': 'spectrum'}})
    else:
        base_color = visualizer._get_base_color_for_scheme(color_scheme)
        style_dict = {visual_style: {'color': base_color}}
        viewer.setStyle(style_dict)
    
    if mutations:
        mutation_visualizations = []
        for pos, from_aa, to_aa in mutations:
            mutation_viz = MutationVisualization(
                position=pos + 1,
                original_aa=from_aa,
                mutated_aa=to_aa,
                color='red',
                size=0.5,
                label=f"{from_aa}{pos+1}{to_aa}"
            )
            mutation_visualizations.append(mutation_viz)
        
        viewer = visualizer.highlight_mutations(viewer, mutation_visualizations)
    
    viewer.zoomTo()
    return viewer

def create_mutation_impact_view(sequence: str, mutations_with_impact: List[Dict],
                              width: int = 800, height: int = 600) -> py3Dmol.view:
    """Create visualization showing mutation impacts with color coding"""
    visualizer = Protein3DVisualizer()
    return visualizer.create_interactive_mutation_explorer(
        sequence, mutations_with_impact, width, height
    )


# Example usage
if __name__ == "__main__":
    # Example protein sequence (first 50 residues of spike protein)
    example_sequence = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLH"
    
    # Example mutations
    example_mutations = [
        (10, 'T', 'A'),  # T11A
        (25, 'R', 'K'),  # R26K
        (40, 'S', 'L')   # S41L
    ]
    
    visualizer = Protein3DVisualizer()
    
    # Create basic view
    print("Creating basic protein view...")
    basic_view = create_simple_protein_view(example_sequence, example_mutations)
    
    # Create comparison view
    print("Creating mutation comparison view...")
    mutated_sequence = list(example_sequence)
    for pos, from_aa, to_aa in example_mutations:
        mutated_sequence[pos] = to_aa
    mutated_sequence = ''.join(mutated_sequence)
    
    comparison_view = visualizer.create_mutation_comparison_view(
        example_sequence, mutated_sequence, example_mutations
    )
    
    print("3D visualization examples created successfully!")
    print("Use these viewers in Streamlit with: stmol.showmol(viewer)")