#!/usr/bin/env python3
"""
Main Integration Script - Virus Mutation Simulation AI Framework
Comprehensive entry point for all simulation and analysis capabilities
"""

import os
import sys
import argparse
import json
import time
import psutil
import platform
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# Core imports
from backend.simulator.mutation_engine import MutationEngine
from backend.simulator.pruning_engine import PruningEngine, prune_mutation_tree
from backend.simulator.epidemiological_model import EpidemiologicalModel, ViralStrain
from backend.analyzer.structural_biology import StructuralBiologyAnalyzer
from backend.analyzer.protein_interaction import ProteinInteractionAnalyzer
from backend.analyzer.protein_3d import Protein3DVisualizer, create_simple_protein_view
from backend.analyzer.report_generator import ReportManager, quick_log_simulation
from backend.models.advanced_ai import AdvancedAIFramework
from frontend.multilingual import MultilingualSupport, get_translator

# Dynamic configuration imports
from backend.utils.dynamic_config import get_dynamic_config_manager, DynamicConfigurationManager
from backend.utils.constants import get_dynamic_constants

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/simulation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class VirusMutationSimulationFramework:
    """
    Main framework class that integrates all components
    """
    
    def __init__(self, config: Optional[Dict] = None, sequence: str = "ACDEFGHIKLMNPQRSTVWY"):
        """Initialize the comprehensive framework with dynamic configuration"""
        # Initialize dynamic configuration manager
        self.config_manager = get_dynamic_config_manager()
        self.constants = get_dynamic_constants()

        # Get dynamic configuration based on sequence
        self.sequence = sequence
        self.config = self._load_dynamic_config(config)
        self.setup_directories()

        # Initialize components
        self.mutation_engine = None
        self.pruning_engine = PruningEngine()
        self.epidemiology_model = None
        self.structural_analyzer = StructuralBiologyAnalyzer()
        self.protein_analyzer = ProteinInteractionAnalyzer()
        self.protein_visualizer = Protein3DVisualizer()
        self.report_manager = ReportManager()
        self.ai_framework = AdvancedAIFramework()
        self.translator = get_translator()

        logger.info("🧬 Virus Mutation Simulation AI Framework initialized with dynamic configuration!")
        logger.info(f"📊 Configuration: {len(self.config)} parameters loaded")
        logger.info(f"🖥️ System Resources: {self.config_manager.system_resources.cpu_count} CPUs, "
                   f"{self.config_manager.system_resources.memory_gb:.1f}GB RAM")
        if self.config_manager.system_resources.gpu_available:
            logger.info(f"🎮 GPU Available: {self.config_manager.system_resources.gpu_count} devices")
    
    def _load_dynamic_config(self, user_config: Optional[Dict] = None) -> Dict:
        """Load dynamic configuration based on sequence and system state"""
        # Get dynamic parameters
        sim_params = self.config_manager.get_simulation_parameters(self.sequence, user_config)
        ai_params = self.config_manager.get_ai_model_parameters(self.sequence, user_config)
        viz_params = self.config_manager.get_visualization_parameters(self.sequence, user_config)
        perf_params = self.config_manager.get_performance_parameters(self.sequence, user_config)

        # Combine into unified configuration
        config = {
            # Simulation parameters (dynamic)
            'mutation_rate': sim_params.mutation_rate,
            'max_generations': sim_params.max_generations,
            'branches_per_node': sim_params.branches_per_node,
            'pruning_method': 'adaptive',
            'pruning_threshold': sim_params.pruning_threshold,

            # Epidemiology parameters (dynamic)
            'population_size': sim_params.population_size,
            'initial_infected': sim_params.initial_infected,
            'transmission_rate': sim_params.transmission_rate,
            'recovery_rate': sim_params.recovery_rate,
            'vaccination_rate': sim_params.vaccination_rate,

            # AI model parameters (dynamic)
            'use_advanced_ai': True,
            'gnn_hidden_dim': ai_params.gnn_hidden_dim,
            'gnn_num_layers': ai_params.gnn_num_layers,
            'transformer_d_model': ai_params.transformer_d_model,
            'transformer_nhead': ai_params.transformer_nhead,
            'transformer_num_layers': ai_params.transformer_num_layers,
            'max_sequence_length': ai_params.max_sequence_length,
            'batch_size': ai_params.batch_size,
            'learning_rate': ai_params.learning_rate,
            'dropout_rate': ai_params.dropout_rate,
            'bayesian_optimization': True,

            # Visualization parameters (dynamic)
            'generate_3d_structures': True,
            'create_mutation_animations': False,
            'export_interactive_plots': True,
            'default_width': viz_params.default_width,
            'default_height': viz_params.default_height,
            'max_residues_display': viz_params.max_residues_display,
            'animation_fps': viz_params.animation_fps,
            'marker_size': viz_params.marker_size,
            'line_width': viz_params.line_width,

            # Output parameters
            'generate_pdf_report': True,
            'export_all_formats': True,
            'save_intermediate_results': True,

            # Language and localization
            'language': 'en',
            'date_format': '%Y-%m-%d %H:%M:%S',

            # Performance parameters (dynamic)
            'use_gpu': self.config_manager.system_resources.gpu_available,
            'parallel_processing': True,
            'gpu_acceleration': self.config_manager.system_resources.gpu_available,
            'memory_optimization': True,
            'max_workers': perf_params.max_workers,
            'chunk_size': perf_params.chunk_size,
            'memory_limit_gb': perf_params.memory_limit_gb,
            'gpu_memory_fraction': perf_params.gpu_memory_fraction,
            'cache_size_mb': perf_params.cache_size_mb,
            'timeout_seconds': perf_params.timeout_seconds
        }

        # Apply user overrides
        if user_config:
            config.update(user_config)

        return config
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            'data', 'logs', 'reports', 'exports', 'temp',
            'docs/research_examples', 'docs/tutorials'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        logger.info(f"📁 Created {len(directories)} directories")
    
    def run_comprehensive_simulation(self, reference_sequence: str,
                                   simulation_name: str = "comprehensive_analysis") -> Dict:
        """
        Run a complete simulation with all components
        
        Args:
            reference_sequence: Viral protein sequence
            simulation_name: Name for this simulation run
            
        Returns:
            Dictionary containing all results and file paths
        """
        start_time = time.time()
        logger.info(f"🚀 Starting comprehensive simulation: {simulation_name}")
        
        results = {
            'simulation_name': simulation_name,
            'start_time': datetime.now().isoformat(),
            'reference_sequence': reference_sequence,
            'config': self.config.copy(),
            'results': {},
            'files_generated': [],
            'errors': []
        }
        
        try:
            # Step 1: Mutation Simulation
            logger.info("🧬 Step 1: Running mutation simulation...")
            mutation_results = self._run_mutation_simulation(reference_sequence)
            results['results']['mutation_simulation'] = mutation_results
            
            # Step 2: Structural Biology Analysis
            logger.info("🔬 Step 2: Analyzing protein structure...")
            structural_results = self._run_structural_analysis(
                reference_sequence, mutation_results.get('mutations', [])
            )
            results['results']['structural_analysis'] = structural_results
            
            # Step 3: Protein Interaction Analysis
            logger.info("🤝 Step 3: Analyzing protein interactions...")
            interaction_results = self._run_interaction_analysis(
                reference_sequence, mutation_results.get('mutations', [])
            )
            results['results']['protein_interactions'] = interaction_results
            
            # Step 4: Epidemiological Modeling
            logger.info("📊 Step 4: Running epidemiological simulation...")
            epi_results = self._run_epidemiological_simulation(reference_sequence)
            results['results']['epidemiology'] = epi_results
            
            # Step 5: Advanced AI Analysis
            if self.config['use_advanced_ai']:
                logger.info("🤖 Step 5: Running AI analysis...")
                ai_results = self._run_ai_analysis(
                    reference_sequence, mutation_results.get('mutations', [])
                )
                results['results']['ai_analysis'] = ai_results
            
            # Step 6: 3D Visualization
            if self.config['generate_3d_structures']:
                logger.info("🎨 Step 6: Generating 3D visualizations...")
                viz_results = self._generate_visualizations(
                    reference_sequence, mutation_results.get('mutations', [])
                )
                results['results']['visualizations'] = viz_results
            
            # Step 7: Comprehensive Reporting
            logger.info("📄 Step 7: Generating comprehensive reports...")
            report_results = self._generate_reports(results, simulation_name)
            results['results']['reports'] = report_results
            results['files_generated'].extend(report_results.get('files', []))
            
            # Calculate execution time
            execution_time = time.time() - start_time
            results['execution_time'] = execution_time
            results['end_time'] = datetime.now().isoformat()
            
            logger.info(f"✅ Simulation completed successfully in {execution_time:.2f} seconds")
            logger.info(f"📁 Generated {len(results['files_generated'])} output files")
            
        except Exception as e:
            logger.error(f"❌ Simulation failed: {str(e)}")
            results['errors'].append(str(e))
            results['status'] = 'failed'
            return results
        
        results['status'] = 'completed'
        return results
    
    def _run_mutation_simulation(self, sequence: str) -> Dict:
        """Run mutation simulation with pruning"""
        self.mutation_engine = MutationEngine(sequence, self.config['mutation_rate'])
        
        # Run simulation
        sim_results = self.mutation_engine.run_simulation(
            max_generations=self.config['max_generations'],
            branches_per_node=self.config['branches_per_node'],
            pruning_method=self.config['pruning_method'],
            pruning_threshold=self.config['pruning_threshold']
        )
        
        # Extract mutation data
        mutations = []
        for node_id, node in sim_results['tree'].items():
            if node.mutations:
                mutations.extend([
                    {
                        'node_id': node_id,
                        'position': mut[0],
                        'from_aa': mut[1],
                        'to_aa': mut[2],
                        'fitness': node.fitness,
                        'generation': node.generation
                    }
                    for mut in node.mutations
                ])
        
        return {
            'total_nodes': sim_results['total_nodes'],
            'final_generation_size': len(sim_results['final_generation']),
            'mutations': mutations,
            'tree_structure': {k: {
                'fitness': v.fitness,
                'generation': v.generation,
                'parent_id': v.parent_id
            } for k, v in sim_results['tree'].items()}
        }
    
    def _run_structural_analysis(self, sequence: str, mutations: List[Dict]) -> Dict:
        """Run structural biology analysis"""
        
        # Analyze protein features
        features = self.structural_analyzer.analyze_protein_features(sequence)
        
        # Analyze mutation impacts
        mutation_tuples = [(m['position'], m['from_aa'], m['to_aa']) for m in mutations[:10]]
        impact_analysis = self.structural_analyzer.analyze_mutation_impact(
            sequence, mutation_tuples
        )
        
        return {
            'sequence_length': len(sequence),
            'protein_features': len(features),
            'mutation_impacts': impact_analysis,
            'conservation_scores': [f.conservation_score for f in features[:50]],
            'secondary_structure': [f.secondary_structure for f in features[:50]]
        }
    
    def _run_interaction_analysis(self, sequence: str, mutations: List[Dict]) -> Dict:
        """Run protein interaction analysis"""
        
        # Mock PDB files for demonstration
        mock_pdb1 = self.protein_visualizer.create_mock_pdb_structure(sequence[:100])
        mock_pdb2 = self.protein_visualizer.create_mock_pdb_structure(sequence[50:150])
        
        # Save temporary PDB files
        pdb_file1 = 'temp/protein1.pdb'
        pdb_file2 = 'temp/protein2.pdb'
        
        with open(pdb_file1, 'w') as f:
            f.write(mock_pdb1)
        with open(pdb_file2, 'w') as f:
            f.write(mock_pdb2)
        
        # Perform docking analysis
        docking_result = self.protein_analyzer.perform_docking_analysis(
            pdb_file1, pdb_file2
        )
        
        # Analyze mutation effects on binding
        mutation_tuples = [(m['position'], m['from_aa'], m['to_aa']) for m in mutations[:5]]
        binding_effects = self.protein_analyzer.analyze_mutation_effect_on_binding(
            sequence, mutation_tuples, pdb_file1, pdb_file2
        )
        
        # Cleanup temporary files
        os.remove(pdb_file1)
        os.remove(pdb_file2)
        
        return {
            'binding_energy': docking_result.binding_energy,
            'interaction_sites': len(docking_result.interaction_sites),
            'interface_area': docking_result.interface_area,
            'confidence_score': docking_result.confidence_score,
            'mutation_binding_effects': binding_effects
        }
    
    def _run_epidemiological_simulation(self, sequence: str) -> Dict:
        """Run epidemiological simulation"""
        
        # Initialize model
        self.epidemiology_model = EpidemiologicalModel(self.config['population_size'])
        
        # Create viral strain
        strain = ViralStrain(
            id="original",
            sequence=sequence,
            transmissibility=1.0,
            virulence=0.5,
            immune_escape=0.1,
            generation=0
        )
        
        self.epidemiology_model.add_viral_strain(strain)
        self.epidemiology_model.introduce_infection("original", self.config['initial_infected'])
        
        # Run simulation
        epi_results = self.epidemiology_model.run_simulation(num_steps=365)
        
        # Calculate summary statistics
        final_stats = epi_results[-1] if epi_results else {}
        peak_infected = max([r['infected'] for r in epi_results]) if epi_results else 0
        
        return {
            'simulation_days': len(epi_results),
            'final_susceptible': final_stats.get('susceptible', 0),
            'final_infected': final_stats.get('infected', 0),
            'final_recovered': final_stats.get('recovered', 0),
            'final_vaccinated': final_stats.get('vaccinated', 0),
            'peak_infected': peak_infected,
            'r0_estimate': self.epidemiology_model.calculate_r0("original"),
            'time_series': epi_results[-100:]  # Last 100 days for reporting
        }
    
    def _run_ai_analysis(self, sequence: str, mutations: List[Dict]) -> Dict:
        """Run advanced AI analysis"""
        
        # Initialize AI models
        self.ai_framework.initialize_models()
        
        # Predict mutation effects
        mutation_tuples = [(m['position'], m['from_aa'], m['to_aa']) for m in mutations[:10]]
        predictions = []
        
        for pos, from_aa, to_aa in mutation_tuples:
            try:
                prediction = self.ai_framework.predict_mutation_effect(
                    sequence, [(pos, from_aa, to_aa)]
                )
                predictions.append({
                    'position': pos,
                    'mutation': f"{from_aa}{pos+1}{to_aa}",
                    'gnn_score': prediction['gnn_score'],
                    'transformer_score': prediction['transformer_score'],
                    'ensemble_score': prediction['ensemble_score'],
                    'confidence': prediction['confidence']
                })
            except Exception as e:
                logger.warning(f"AI prediction failed for {from_aa}{pos+1}{to_aa}: {e}")
        
        # Generate attention maps
        try:
            attention_maps = self.ai_framework.generate_attention_maps(sequence[:100])
        except Exception as e:
            logger.warning(f"Attention map generation failed: {e}")
            attention_maps = {}
        
        return {
            'predictions': predictions,
            'attention_maps_generated': len(attention_maps),
            'average_ensemble_score': sum(p['ensemble_score'] for p in predictions) / len(predictions) if predictions else 0,
            'high_impact_mutations': [p for p in predictions if p['ensemble_score'] > 0.7]
        }
    
    def _generate_visualizations(self, sequence: str, mutations: List[Dict]) -> Dict:
        """Generate 3D visualizations"""
        
        viz_files = []
        
        try:
            # Create basic protein view
            mutation_tuples = [(m['position'], m['from_aa'], m['to_aa']) for m in mutations[:5]]
            basic_view = create_simple_protein_view(sequence, mutation_tuples)
            
            # Export as HTML
            html_file = f"exports/protein_3d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            self.protein_visualizer.export_visualization_html(basic_view, html_file)
            viz_files.append(html_file)
            
            # Create mutation comparison view
            if mutations:
                mutated_sequence = list(sequence)
                for mut in mutations[:3]:  # Apply first 3 mutations
                    pos = mut['position']
                    if pos < len(mutated_sequence):
                        mutated_sequence[pos] = mut['to_aa']
                mutated_sequence = ''.join(mutated_sequence)
                
                comparison_view = self.protein_visualizer.create_mutation_comparison_view(
                    sequence, mutated_sequence, mutation_tuples[:3]
                )
                
                comparison_file = f"exports/mutation_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                self.protein_visualizer.export_visualization_html(comparison_view, comparison_file)
                viz_files.append(comparison_file)
            
        except Exception as e:
            logger.warning(f"3D visualization generation failed: {e}")
        
        return {
            'files_generated': viz_files,
            'visualization_count': len(viz_files)
        }
    
    def _generate_reports(self, results: Dict, simulation_name: str) -> Dict:
        """Generate comprehensive reports"""
        
        # Log simulation to database
        run_id = quick_log_simulation(
            simulation_type="comprehensive_analysis",
            parameters=self.config,
            results=results['results'],
            duration=results.get('execution_time', 0)
        )
        
        # Generate comprehensive report
        report_files = self.report_manager.create_full_report(
            run_id, include_exports=self.config['export_all_formats']
        )
        
        # Generate summary JSON
        summary_file = f"exports/simulation_summary_{simulation_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        report_files['summary_json'] = summary_file
        
        return {
            'run_id': run_id,
            'files': report_files,
            'summary_file': summary_file
        }
    
    def run_quick_analysis(self, sequence: str, mutations: List[tuple] = None) -> Dict:
        """Run a quick analysis for testing purposes"""
        logger.info("⚡ Running quick analysis...")
        
        if mutations is None:
            # Generate a few random mutations for testing
            import random
            mutations = []
            for _ in range(3):
                pos = random.randint(0, len(sequence) - 1)
                from_aa = sequence[pos]
                to_aa = random.choice(['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I'])
                mutations.append((pos, from_aa, to_aa))
        
        # Quick structural analysis
        impact_analysis = self.structural_analyzer.analyze_mutation_impact(sequence, mutations)
        
        # Quick AI prediction
        if self.config['use_advanced_ai']:
            try:
                self.ai_framework.initialize_models()
                ai_prediction = self.ai_framework.predict_mutation_effect(sequence, mutations)
            except Exception as e:
                logger.warning(f"AI analysis failed: {e}")
                ai_prediction = {'ensemble_score': 0.5, 'confidence': 0.5}
        else:
            ai_prediction = {'ensemble_score': 0.5, 'confidence': 0.5}
        
        return {
            'sequence_length': len(sequence),
            'mutations_analyzed': len(mutations),
            'structural_impact': impact_analysis,
            'ai_prediction': ai_prediction,
            'analysis_time': time.time()
        }


def load_reference_sequences() -> Dict[str, str]:
    """Load reference sequences from FASTA file"""
    sequences = {}
    try:
        reference_file = os.path.join('data', 'reference_sequences.fasta')
        if os.path.exists(reference_file):
            with open(reference_file, 'r') as f:
                current_name = None
                current_seq = []
                
                for line in f:
                    line = line.strip()
                    if line.startswith('>'):
                        # Save previous sequence
                        if current_name and current_seq:
                            sequences[current_name] = ''.join(current_seq)
                        # Start new sequence
                        current_name = line[1:]  # Remove '>'
                        current_seq = []
                    elif line and not line.startswith('>'):
                        current_seq.append(line)
                
                # Save last sequence
                if current_name and current_seq:
                    sequences[current_name] = ''.join(current_seq)
                    
    except Exception as e:
        logger.warning(f"Could not load reference sequences: {e}")
    
    return sequences

def list_available_sequences():
    """List all available reference sequences"""
    sequences = load_reference_sequences()
    if sequences:
        print("\n🧬 Available Reference Sequences:")
        print("=" * 50)
        for name, seq in sequences.items():
            print(f"📄 {name}")
            print(f"   Length: {len(seq)} amino acids")
            print(f"   Preview: {seq[:50]}{'...' if len(seq) > 50 else ''}")
            print()
    else:
        print("❌ No reference sequences found in data/reference_sequences.fasta")

def main():
    """Main entry point with command line interface"""
    parser = argparse.ArgumentParser(
        description="Virus Mutation Simulation AI Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --sequence "SARS-CoV-2" --quick  # Use reference sequence by name
  python main.py --config config.json --comprehensive
  python main.py --streamlit
  python main.py --list-sequences  # Show available reference sequences
        """
    )
    
    parser.add_argument('--sequence', '-s', type=str,
                       help='Protein sequence for analysis')
    parser.add_argument('--config', '-c', type=str,
                       help='Configuration file (JSON)')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick analysis')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Run comprehensive simulation')
    parser.add_argument('--streamlit', action='store_true',
                       help='Launch Streamlit web interface')
    parser.add_argument('--language', '-l', type=str, default='en',
                       help='Interface language (en, es, fr, de, zh, ja, etc.)')
    parser.add_argument('--output-dir', '-o', type=str, default='.',
                       help='Output directory for results')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    parser.add_argument('--list-sequences', action='store_true',
                       help='List available reference sequences and exit')
    
    args = parser.parse_args()
    
    # Handle --list-sequences
    if args.list_sequences:
        list_available_sequences()
        return
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Change to output directory
    if args.output_dir != '.':
        os.chdir(args.output_dir)
    
    # Load configuration
    config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            logger.info(f"📋 Loaded configuration from {args.config}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return 1
    
    # Set language
    if args.language:
        translator = get_translator()
        if translator.set_language(args.language):
            logger.info(f"🌍 Language set to {args.language}")
        else:
            logger.warning(f"Language {args.language} not supported, using English")
    
    # Launch Streamlit interface
    if args.streamlit:
        logger.info("🚀 Launching Streamlit web interface...")
        os.system("streamlit run frontend/streamlit_app.py")
        return 0
    
    # Initialize framework
    framework = VirusMutationSimulationFramework(config)
    
    # Handle sequence input - can be sequence name or actual sequence
    if not args.sequence:
        # No sequence provided - use first available reference sequence
        reference_sequences = load_reference_sequences()
        if reference_sequences:
            first_name = list(reference_sequences.keys())[0]
            args.sequence = reference_sequences[first_name]
            logger.info(f"Using reference sequence: {first_name}")
        else:
            # Fallback to hardcoded sequence
            args.sequence = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLH"
            logger.info("Using default SARS-CoV-2 spike protein sequence (fallback)")
    else:
        # Check if the provided sequence is a reference name or actual sequence
        if len(args.sequence) < 50 and not any(c in args.sequence for c in 'ACDEFGHIKLMNPQRSTVWY'):
            # Likely a sequence name, try to load from references
            reference_sequences = load_reference_sequences()
            found_sequence = None
            
            # Try exact match first
            if args.sequence in reference_sequences:
                found_sequence = reference_sequences[args.sequence]
            else:
                # Try partial match (case insensitive)
                for name, seq in reference_sequences.items():
                    if args.sequence.lower() in name.lower():
                        found_sequence = seq
                        logger.info(f"Found matching reference sequence: {name}")
                        break
            
            if found_sequence:
                args.sequence = found_sequence
                logger.info(f"Using reference sequence for: {args.sequence}")
            else:
                logger.error(f"Reference sequence '{args.sequence}' not found. Use --list-sequences to see available options.")
                return
        else:
            # Assume it's an actual sequence
            logger.info(f"Using provided sequence ({len(args.sequence)} amino acids)")
    
    # Run analysis
    try:
        if args.quick:
            logger.info("⚡ Running quick analysis...")
            results = framework.run_quick_analysis(args.sequence)
            print(json.dumps(results, indent=2, default=str))
            
        elif args.comprehensive:
            logger.info("🔬 Running comprehensive simulation...")
            results = framework.run_comprehensive_simulation(
                args.sequence, 
                f"comprehensive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            print(f"\n{'='*60}")
            print("🎉 SIMULATION COMPLETED SUCCESSFULLY!")
            print(f"{'='*60}")
            print(f"📊 Execution time: {results.get('execution_time', 0):.2f} seconds")
            print(f"📁 Files generated: {len(results.get('files_generated', []))}")
            print(f"🔬 Components analyzed: {len(results.get('results', {}))}")
            
            if results.get('files_generated'):
                print(f"\n📄 Generated files:")
                for file_path in results['files_generated'][:10]:  # Show first 10
                    print(f"  • {file_path}")
                if len(results['files_generated']) > 10:
                    print(f"  ... and {len(results['files_generated']) - 10} more files")
            
        else:
            print("Please specify --quick, --comprehensive, or --streamlit")
            parser.print_help()
            return 1
            
    except KeyboardInterrupt:
        logger.info("🛑 Analysis interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        return 1
    
    logger.info("✅ Analysis completed successfully")
    return 0


if __name__ == "__main__":
    exit(main())