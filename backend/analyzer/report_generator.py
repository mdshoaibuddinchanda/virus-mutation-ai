"""
Report Generator - PDF export and SQLite logging functionality
Comprehensive reporting system for virus mutation simulation results
"""
import sqlite3
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import json
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.barcharts import VerticalBarChart
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from pathlib import Path

class DatabaseLogger:
    """SQLite database logger for simulation results"""
    
    def __init__(self, db_path: str = "data/simulation_logs.db"):
        self.db_path = db_path
        self.ensure_data_dir()
        self.init_database()
    
    def ensure_data_dir(self):
        """Ensure data directory exists"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Simulation runs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS simulation_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                simulation_type TEXT NOT NULL,
                parameters TEXT NOT NULL,
                results TEXT,
                duration_seconds REAL,
                status TEXT DEFAULT 'completed',
                user_notes TEXT
            )
        ''')
        
        # Mutation data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mutations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                generation INTEGER,
                mutation_id TEXT,
                parent_id TEXT,
                sequence TEXT,
                fitness_score REAL,
                structural_impact REAL,
                conservation_score REAL,
                FOREIGN KEY (run_id) REFERENCES simulation_runs (id)
            )
        ''')
        
        # Protein interactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS protein_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                protein_a TEXT,
                protein_b TEXT,
                binding_energy REAL,
                interaction_type TEXT,
                confidence_score REAL,
                FOREIGN KEY (run_id) REFERENCES simulation_runs (id)
            )
        ''')
        
        # Epidemiological data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS epidemiology_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                time_step INTEGER,
                susceptible INTEGER,
                infected INTEGER,
                recovered INTEGER,
                mutation_prevalence TEXT,
                FOREIGN KEY (run_id) REFERENCES simulation_runs (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_simulation_run(self, simulation_type: str, parameters: Dict[str, Any], 
                          results: Optional[Dict[str, Any]] = None, 
                          duration: Optional[float] = None,
                          user_notes: str = "") -> int:
        """Log a simulation run and return the run ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO simulation_runs 
            (simulation_type, parameters, results, duration_seconds, user_notes)
            VALUES (?, ?, ?, ?, ?)
        ''', (simulation_type, json.dumps(parameters), 
              json.dumps(results) if results else None, duration, user_notes))
        
        run_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return run_id
    
    def log_mutations(self, run_id: int, mutations: List[Dict[str, Any]]):
        """Log mutation data for a simulation run"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for mutation in mutations:
            cursor.execute('''
                INSERT INTO mutations 
                (run_id, generation, mutation_id, parent_id, sequence, 
                 fitness_score, structural_impact, conservation_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (run_id, mutation.get('generation', 0), mutation.get('id'),
                  mutation.get('parent_id'), mutation.get('sequence'),
                  mutation.get('fitness_score'), mutation.get('structural_impact'),
                  mutation.get('conservation_score')))
        
        conn.commit()
        conn.close()
    
    def log_protein_interactions(self, run_id: int, interactions: List[Dict[str, Any]]):
        """Log protein interaction data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for interaction in interactions:
            cursor.execute('''
                INSERT INTO protein_interactions 
                (run_id, protein_a, protein_b, binding_energy, interaction_type, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (run_id, interaction.get('protein_a'), interaction.get('protein_b'),
                  interaction.get('binding_energy'), interaction.get('interaction_type'),
                  interaction.get('confidence_score')))
        
        conn.commit()
        conn.close()
    
    def log_epidemiology_data(self, run_id: int, epi_data: List[Dict[str, Any]]):
        """Log epidemiological simulation data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for data_point in epi_data:
            cursor.execute('''
                INSERT INTO epidemiology_data 
                (run_id, time_step, susceptible, infected, recovered, mutation_prevalence)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (run_id, data_point.get('time_step'), data_point.get('susceptible'),
                  data_point.get('infected'), data_point.get('recovered'),
                  json.dumps(data_point.get('mutation_prevalence', {}))))
        
        conn.commit()
        conn.close()
    
    def get_simulation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent simulation history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, timestamp, simulation_type, parameters, status, user_notes
            FROM simulation_runs 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'timestamp': row[1],
                'simulation_type': row[2],
                'parameters': json.loads(row[3]) if row[3] else {},
                'status': row[4],
                'user_notes': row[5]
            })
        
        conn.close()
        return results

class PDFReportGenerator:
    """Generate comprehensive PDF reports for simulation results"""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        self.ensure_output_dir()
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def ensure_output_dir(self):
        """Ensure output directory exists"""
        os.makedirs(self.output_dir, exist_ok=True)
    
    def setup_custom_styles(self):
        """Setup custom paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            textColor=colors.darkblue,
            alignment=1  # Center alignment
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=12,
            textColor=colors.darkgreen
        ))
        
        self.styles.add(ParagraphStyle(
            name='MethodText',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceBefore=6,
            spaceAfter=6,
            leftIndent=20
        ))
    
    def create_chart_image(self, data: Dict[str, Any], chart_type: str) -> str:
        """Create matplotlib chart and return as base64 image"""
        plt.style.use('seaborn-v0_8')
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if chart_type == 'mutation_fitness':
            generations = data.get('generations', [])
            fitness_scores = data.get('fitness_scores', [])
            ax.plot(generations, fitness_scores, 'b-', linewidth=2, marker='o')
            ax.set_xlabel('Generation')
            ax.set_ylabel('Average Fitness Score')
            ax.set_title('Mutation Fitness Evolution')
            ax.grid(True, alpha=0.3)
            
        elif chart_type == 'sir_curve':
            time_steps = data.get('time_steps', [])
            susceptible = data.get('susceptible', [])
            infected = data.get('infected', [])
            recovered = data.get('recovered', [])
            
            ax.plot(time_steps, susceptible, 'b-', label='Susceptible', linewidth=2)
            ax.plot(time_steps, infected, 'r-', label='Infected', linewidth=2)
            ax.plot(time_steps, recovered, 'g-', label='Recovered', linewidth=2)
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Population')
            ax.set_title('SIR Epidemiological Model')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        elif chart_type == 'binding_energy':
            proteins = data.get('proteins', [])
            energies = data.get('binding_energies', [])
            ax.bar(proteins, energies, color='skyblue', edgecolor='navy')
            ax.set_xlabel('Protein Interactions')
            ax.set_ylabel('Binding Energy (kcal/mol)')
            ax.set_title('Protein-Protein Interaction Energies')
            plt.xticks(rotation=45)
            
        elif chart_type == 'mutation_tree':
            # Simple tree visualization
            nodes = data.get('nodes', [])
            edges = data.get('edges', [])
            
            # Create a simple network plot
            import networkx as nx
            G = nx.DiGraph()
            G.add_nodes_from(nodes)
            G.add_edges_from(edges)
            
            pos = nx.spring_layout(G)
            nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue',
                   node_size=500, font_size=8, arrows=True)
            ax.set_title('Mutation Lineage Tree')
        
        plt.tight_layout()
        
        # Save to bytes
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        
        return img_str
    
    def generate_comprehensive_report(self, run_id: int, db_logger: DatabaseLogger,
                                    additional_data: Optional[Dict[str, Any]] = None) -> str:
        """Generate a comprehensive PDF report for a simulation run"""
        
        # Get simulation data from database
        conn = sqlite3.connect(db_logger.db_path)
        
        # Get run information
        run_info = pd.read_sql_query(
            "SELECT * FROM simulation_runs WHERE id = ?", conn, params=(run_id,)
        ).iloc[0]
        
        # Get mutations data
        mutations_df = pd.read_sql_query(
            "SELECT * FROM mutations WHERE run_id = ?", conn, params=(run_id,)
        )
        
        # Get protein interactions
        interactions_df = pd.read_sql_query(
            "SELECT * FROM protein_interactions WHERE run_id = ?", conn, params=(run_id,)
        )
        
        # Get epidemiology data
        epi_df = pd.read_sql_query(
            "SELECT * FROM epidemiology_data WHERE run_id = ?", conn, params=(run_id,)
        )
        
        conn.close()
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simulation_report_{run_id}_{timestamp}.pdf"
        filepath = os.path.join(self.output_dir, filename)
        
        # Create PDF document
        doc = SimpleDocTemplate(filepath, pagesize=A4)
        story = []
        
        # Title page
        story.append(Paragraph("Virus Mutation Simulation Report", self.styles['CustomTitle']))
        story.append(Spacer(1, 20))
        
        # Executive summary
        story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        parameters = json.loads(run_info['parameters']) if run_info['parameters'] else {}
        summary_text = f"""
        <b>Simulation Type:</b> {run_info['simulation_type']}<br/>
        <b>Run ID:</b> {run_id}<br/>
        <b>Timestamp:</b> {run_info['timestamp']}<br/>
        <b>Duration:</b> {run_info['duration_seconds']:.2f} seconds<br/>
        <b>Status:</b> {run_info['status']}<br/>
        <b>Total Mutations Generated:</b> {len(mutations_df)}<br/>
        <b>Protein Interactions Analyzed:</b> {len(interactions_df)}<br/>
        """
        
        story.append(Paragraph(summary_text, self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Parameters section
        story.append(Paragraph("Simulation Parameters", self.styles['SectionHeader']))
        
        param_data = []
        for key, value in parameters.items():
            param_data.append([key.replace('_', ' ').title(), str(value)])
        
        if param_data:
            param_table = Table(param_data, colWidths=[2*inch, 3*inch])
            param_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(param_table)
        
        story.append(PageBreak())
        
        # Mutation Analysis Section
        if not mutations_df.empty:
            story.append(Paragraph("Mutation Analysis", self.styles['SectionHeader']))
            
            # Mutation statistics
            avg_fitness = mutations_df['fitness_score'].mean()
            max_fitness = mutations_df['fitness_score'].max()
            min_fitness = mutations_df['fitness_score'].min()
            
            mutation_stats = f"""
            <b>Average Fitness Score:</b> {avg_fitness:.4f}<br/>
            <b>Maximum Fitness Score:</b> {max_fitness:.4f}<br/>
            <b>Minimum Fitness Score:</b> {min_fitness:.4f}<br/>
            <b>Fitness Standard Deviation:</b> {mutations_df['fitness_score'].std():.4f}<br/>
            """
            
            story.append(Paragraph(mutation_stats, self.styles['Normal']))
            story.append(Spacer(1, 15))
            
            # Create fitness evolution chart
            if len(mutations_df) > 1:
                chart_data = {
                    'generations': mutations_df['generation'].tolist(),
                    'fitness_scores': mutations_df['fitness_score'].tolist()
                }
                
                img_str = self.create_chart_image(chart_data, 'mutation_fitness')
                
                # Save temporary image
                temp_img_path = os.path.join(self.output_dir, f"temp_chart_{run_id}.png")
                with open(temp_img_path, 'wb') as f:
                    f.write(base64.b64decode(img_str))
                
                story.append(Image(temp_img_path, width=6*inch, height=4*inch))
                story.append(Spacer(1, 15))
                
                # Clean up temp file
                os.remove(temp_img_path)
        
        # Protein Interaction Analysis
        if not interactions_df.empty:
            story.append(PageBreak())
            story.append(Paragraph("Protein Interaction Analysis", self.styles['SectionHeader']))
            
            avg_binding = interactions_df['binding_energy'].mean()
            interaction_stats = f"""
            <b>Total Interactions Analyzed:</b> {len(interactions_df)}<br/>
            <b>Average Binding Energy:</b> {avg_binding:.4f} kcal/mol<br/>
            <b>Strongest Interaction:</b> {interactions_df['binding_energy'].min():.4f} kcal/mol<br/>
            <b>Weakest Interaction:</b> {interactions_df['binding_energy'].max():.4f} kcal/mol<br/>
            """
            
            story.append(Paragraph(interaction_stats, self.styles['Normal']))
            story.append(Spacer(1, 15))
            
            # Top interactions table
            top_interactions = interactions_df.nsmallest(10, 'binding_energy')
            interaction_data = [['Protein A', 'Protein B', 'Binding Energy', 'Type', 'Confidence']]
            
            for _, row in top_interactions.iterrows():
                interaction_data.append([
                    row['protein_a'], row['protein_b'], 
                    f"{row['binding_energy']:.3f}", 
                    row['interaction_type'], 
                    f"{row['confidence_score']:.3f}"
                ])
            
            interaction_table = Table(interaction_data, colWidths=[1.2*inch, 1.2*inch, 1*inch, 1*inch, 1*inch])
            interaction_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(interaction_table)
        
        # Epidemiological Analysis
        if not epi_df.empty:
            story.append(PageBreak())
            story.append(Paragraph("Epidemiological Analysis", self.styles['SectionHeader']))
            
            final_infected = epi_df['infected'].iloc[-1] if len(epi_df) > 0 else 0
            peak_infected = epi_df['infected'].max()
            
            epi_stats = f"""
            <b>Simulation Duration:</b> {len(epi_df)} time steps<br/>
            <b>Peak Infected Population:</b> {peak_infected}<br/>
            <b>Final Infected Population:</b> {final_infected}<br/>
            <b>Total Recovered:</b> {epi_df['recovered'].iloc[-1] if len(epi_df) > 0 else 0}<br/>
            """
            
            story.append(Paragraph(epi_stats, self.styles['Normal']))
            story.append(Spacer(1, 15))
            
            # Create SIR curve
            if len(epi_df) > 1:
                sir_data = {
                    'time_steps': epi_df['time_step'].tolist(),
                    'susceptible': epi_df['susceptible'].tolist(),
                    'infected': epi_df['infected'].tolist(),
                    'recovered': epi_df['recovered'].tolist()
                }
                
                img_str = self.create_chart_image(sir_data, 'sir_curve')
                
                temp_img_path = os.path.join(self.output_dir, f"temp_sir_{run_id}.png")
                with open(temp_img_path, 'wb') as f:
                    f.write(base64.b64decode(img_str))
                
                story.append(Image(temp_img_path, width=6*inch, height=4*inch))
                os.remove(temp_img_path)
        
        # Methodology section
        story.append(PageBreak())
        story.append(Paragraph("Methodology", self.styles['SectionHeader']))
        
        methodology_text = """
        This simulation employed advanced computational methods including:
        
        • <b>Structural Biology Analysis:</b> AlphaFold-based protein structure prediction with solvent accessibility and conservation scoring
        
        • <b>Graph Neural Networks:</b> Mutation-aware protein interaction modeling using GNN architectures
        
        • <b>Transformer Embeddings:</b> ESM/ProtBERT-style sequence representations for enhanced mutation impact prediction
        
        • <b>Epidemiological Modeling:</b> Individual-based simulation with SIR dynamics and mutation tracking
        
        • <b>Bayesian Optimization:</b> Gaussian Process-based hyperparameter tuning for optimal simulation parameters
        
        • <b>3D Molecular Docking:</b> Energy-based protein-protein interaction prediction with binding affinity calculation
        """
        
        story.append(Paragraph(methodology_text, self.styles['MethodText']))
        
        # Footer with generation info
        story.append(Spacer(1, 30))
        footer_text = f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by Virus Mutation Simulation AI Framework"
        story.append(Paragraph(footer_text, self.styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        return filepath


class MultiFormatExporter:
    """Export simulation data in multiple formats (JSON, CSV, FASTA, PDB)"""
    
    def __init__(self, output_dir: str = "exports"):
        self.output_dir = output_dir
        self.ensure_output_dir()
    
    def ensure_output_dir(self):
        """Ensure output directory exists"""
        os.makedirs(self.output_dir, exist_ok=True)
    
    def export_to_json(self, data: Dict[str, Any], filename: str) -> str:
        """Export data to JSON format"""
        filepath = os.path.join(self.output_dir, f"{filename}.json")
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return filepath
    
    def export_to_csv(self, data: pd.DataFrame, filename: str) -> str:
        """Export DataFrame to CSV format"""
        filepath = os.path.join(self.output_dir, f"{filename}.csv")
        data.to_csv(filepath, index=False)
        return filepath
    
    def export_mutations_to_fasta(self, mutations_data: List[Dict[str, Any]], filename: str) -> str:
        """Export mutation sequences to FASTA format"""
        filepath = os.path.join(self.output_dir, f"{filename}.fasta")
        
        with open(filepath, 'w') as f:
            for i, mutation in enumerate(mutations_data):
                mutation_id = mutation.get('id', f'mutation_{i}')
                sequence = mutation.get('sequence', '')
                fitness = mutation.get('fitness_score', 0.0)
                generation = mutation.get('generation', 0)
                
                header = f">{mutation_id} | Generation:{generation} | Fitness:{fitness:.4f}"
                f.write(f"{header}\n")
                
                # Write sequence in 80-character lines
                for j in range(0, len(sequence), 80):
                    f.write(f"{sequence[j:j+80]}\n")
        
        return filepath
    
    def export_protein_structures_to_pdb(self, structures_data: List[Dict[str, Any]], filename: str) -> str:
        """Export protein structure data to PDB format (simplified)"""
        filepath = os.path.join(self.output_dir, f"{filename}.pdb")
        
        with open(filepath, 'w') as f:
            f.write("HEADER    VIRUS MUTATION SIMULATION STRUCTURES\n")
            f.write(f"TITLE     EXPORTED ON {datetime.now().strftime('%Y-%m-%d')}\n")
            
            for structure in structures_data:
                protein_name = structure.get('name', 'UNKNOWN')
                f.write(f"MODEL     {protein_name}\n")
                
                # Write simplified atom records (this would need real PDB data in practice)
                atoms = structure.get('atoms', [])
                for atom in atoms:
                    f.write(f"ATOM  {atom.get('serial', 1):5d}  {atom.get('name', 'CA'):4s} "
                           f"{atom.get('residue', 'ALA')} A{atom.get('residue_num', 1):4d}    "
                           f"{atom.get('x', 0.0):8.3f}{atom.get('y', 0.0):8.3f}{atom.get('z', 0.0):8.3f}"
                           f"  1.00 20.00           {atom.get('element', 'C'):>2s}\n")
                
                f.write("ENDMDL\n")
        
        return filepath
    
    def export_simulation_summary(self, run_id: int, db_logger: DatabaseLogger) -> Dict[str, str]:
        """Export complete simulation data in all formats"""
        
        # Get data from database
        conn = sqlite3.connect(db_logger.db_path)
        
        run_info = pd.read_sql_query(
            "SELECT * FROM simulation_runs WHERE id = ?", conn, params=(run_id,)
        )
        
        mutations_df = pd.read_sql_query(
            "SELECT * FROM mutations WHERE run_id = ?", conn, params=(run_id,)
        )
        
        interactions_df = pd.read_sql_query(
            "SELECT * FROM protein_interactions WHERE run_id = ?", conn, params=(run_id,)
        )
        
        epi_df = pd.read_sql_query(
            "SELECT * FROM epidemiology_data WHERE run_id = ?", conn, params=(run_id,)
        )
        
        conn.close()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"simulation_{run_id}_{timestamp}"
        
        exported_files = {}
        
        # Export run information as JSON
        if not run_info.empty:
            run_data = run_info.iloc[0].to_dict()
            exported_files['run_info'] = self.export_to_json(run_data, f"{base_filename}_run_info")
        
        # Export mutations as CSV and FASTA
        if not mutations_df.empty:
            exported_files['mutations_csv'] = self.export_to_csv(mutations_df, f"{base_filename}_mutations")
            
            mutations_list = mutations_df.to_dict('records')
            exported_files['mutations_fasta'] = self.export_mutations_to_fasta(
                mutations_list, f"{base_filename}_sequences"
            )
        
        # Export interactions as CSV
        if not interactions_df.empty:
            exported_files['interactions'] = self.export_to_csv(interactions_df, f"{base_filename}_interactions")
        
        # Export epidemiology data as CSV
        if not epi_df.empty:
            exported_files['epidemiology'] = self.export_to_csv(epi_df, f"{base_filename}_epidemiology")
        
        return exported_files


class ReportManager:
    """Main interface for report generation and data export"""
    
    def __init__(self, db_path: str = "data/simulation_logs.db", 
                 reports_dir: str = "reports", exports_dir: str = "exports"):
        self.db_logger = DatabaseLogger(db_path)
        self.pdf_generator = PDFReportGenerator(reports_dir)
        self.exporter = MultiFormatExporter(exports_dir)
    
    def create_full_report(self, run_id: int, include_exports: bool = True) -> Dict[str, str]:
        """Create comprehensive report with PDF and exports"""
        
        results = {}
        
        # Generate PDF report
        try:
            pdf_path = self.pdf_generator.generate_comprehensive_report(run_id, self.db_logger)
            results['pdf_report'] = pdf_path
            print(f"✅ PDF report generated: {pdf_path}")
        except Exception as e:
            print(f"❌ Error generating PDF report: {str(e)}")
            results['pdf_error'] = str(e)
        
        # Generate exports
        if include_exports:
            try:
                export_files = self.exporter.export_simulation_summary(run_id, self.db_logger)
                results.update(export_files)
                print(f"✅ Data exports completed: {len(export_files)} files")
            except Exception as e:
                print(f"❌ Error generating exports: {str(e)}")
                results['export_error'] = str(e)
        
        return results
    
    def get_simulation_statistics(self) -> Dict[str, Any]:
        """Get overall simulation statistics"""
        conn = sqlite3.connect(self.db_logger.db_path)
        
        stats = {}
        
        # Total simulations
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM simulation_runs")
        stats['total_simulations'] = cursor.fetchone()[0]
        
        # Simulation types breakdown
        cursor.execute("SELECT simulation_type, COUNT(*) FROM simulation_runs GROUP BY simulation_type")
        stats['simulation_types'] = dict(cursor.fetchall())
        
        # Average duration
        cursor.execute("SELECT AVG(duration_seconds) FROM simulation_runs WHERE duration_seconds IS NOT NULL")
        avg_duration = cursor.fetchone()[0]
        stats['average_duration'] = avg_duration if avg_duration else 0
        
        # Total mutations generated
        cursor.execute("SELECT COUNT(*) FROM mutations")
        stats['total_mutations'] = cursor.fetchone()[0]
        
        # Total interactions analyzed
        cursor.execute("SELECT COUNT(*) FROM protein_interactions")
        stats['total_interactions'] = cursor.fetchone()[0]
        
        conn.close()
        return stats
    
    def cleanup_old_reports(self, days_old: int = 30):
        """Clean up old report files"""
        import time
        
        current_time = time.time()
        cutoff_time = current_time - (days_old * 24 * 60 * 60)
        
        cleaned_files = []
        
        # Clean PDF reports
        for filename in os.listdir(self.pdf_generator.output_dir):
            filepath = os.path.join(self.pdf_generator.output_dir, filename)
            if os.path.isfile(filepath) and os.path.getmtime(filepath) < cutoff_time:
                os.remove(filepath)
                cleaned_files.append(filepath)
        
        # Clean exports
        for filename in os.listdir(self.exporter.output_dir):
            filepath = os.path.join(self.exporter.output_dir, filename)
            if os.path.isfile(filepath) and os.path.getmtime(filepath) < cutoff_time:
                os.remove(filepath)
                cleaned_files.append(filepath)
        
        return cleaned_files


# Utility functions for integration with other modules
def quick_log_simulation(simulation_type: str, parameters: Dict[str, Any], 
                        results: Optional[Dict[str, Any]] = None,
                        duration: Optional[float] = None) -> int:
    """Quick function to log a simulation run"""
    logger = DatabaseLogger()
    return logger.log_simulation_run(simulation_type, parameters, results, duration)

def generate_quick_report(run_id: int) -> str:
    """Quick function to generate a PDF report"""
    manager = ReportManager()
    results = manager.create_full_report(run_id)
    return results.get('pdf_report', 'Report generation failed')

def export_simulation_data(run_id: int) -> Dict[str, str]:
    """Quick function to export simulation data"""
    manager = ReportManager()
    return manager.exporter.export_simulation_summary(run_id, manager.db_logger)

# Example usage and testing functions
if __name__ == "__main__":
    # Example usage
    manager = ReportManager()
    
    # Example simulation data
    example_params = {
        'mutation_rate': 0.01,
        'population_size': 1000,
        'generations': 100,
        'selection_pressure': 0.8
    }
    
    example_results = {
        'final_fitness': 0.95,
        'mutations_generated': 150,
        'convergence_generation': 85
    }
    
    # Log example simulation
    run_id = manager.db_logger.log_simulation_run(
        'mutation_evolution', example_params, example_results, 45.2
    )
    
    print(f"Logged simulation with ID: {run_id}")
    
    # Generate report
    report_files = manager.create_full_report(run_id)
    print("Generated files:", report_files)
    
    # Show statistics
    stats = manager.get_simulation_statistics()
    print("Simulation statistics:", stats)