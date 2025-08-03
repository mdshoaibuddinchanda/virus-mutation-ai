"""
Multilingual Support - Translation system with fallback mechanisms
"""
import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Try to import transformers for AI translation, fallback to static translations
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. Using static translations only.")

@dataclass
class LanguageConfig:
    """Configuration for a supported language"""
    code: str
    name: str
    native_name: str
    rtl: bool = False  # Right-to-left text
    available_models: List[str] = None

class MultilingualSupport:
    """Comprehensive multilingual support system"""
    
    def __init__(self, default_language: str = 'en'):
        self.default_language = default_language
        self.current_language = default_language
        self.static_translations = {}
        self.ai_translator = None
        
        # Supported languages
        self.supported_languages = {
            'en': LanguageConfig('en', 'English', 'English'),
            'es': LanguageConfig('es', 'Spanish', 'Español'),
            'fr': LanguageConfig('fr', 'French', 'Français'),
            'de': LanguageConfig('de', 'German', 'Deutsch'),
            'it': LanguageConfig('it', 'Italian', 'Italiano'),
            'pt': LanguageConfig('pt', 'Portuguese', 'Português'),
            'ru': LanguageConfig('ru', 'Russian', 'Русский'),
            'zh': LanguageConfig('zh', 'Chinese', '中文'),
            'ja': LanguageConfig('ja', 'Japanese', '日本語'),
            'ko': LanguageConfig('ko', 'Korean', '한국어'),
            'ar': LanguageConfig('ar', 'Arabic', 'العربية', rtl=True),
            'hi': LanguageConfig('hi', 'Hindi', 'हिन्दी'),
            'bn': LanguageConfig('bn', 'Bengali', 'বাংলা'),
            'tr': LanguageConfig('tr', 'Turkish', 'Türkçe'),
            'pl': LanguageConfig('pl', 'Polish', 'Polski'),
            'nl': LanguageConfig('nl', 'Dutch', 'Nederlands'),
            'sv': LanguageConfig('sv', 'Swedish', 'Svenska'),
            'da': LanguageConfig('da', 'Danish', 'Dansk'),
            'no': LanguageConfig('no', 'Norwegian', 'Norsk'),
            'fi': LanguageConfig('fi', 'Finnish', 'Suomi')
        }
        
        self._load_static_translations()
        self._initialize_ai_translator()
    
    def _load_static_translations(self):
        """Load static translation dictionaries"""
        
        # Core UI translations
        self.static_translations = {
            'en': {
                # Main interface
                'app_title': 'Virus Mutation Simulation AI',
                'app_subtitle': 'A Comprehensive AI-Based Framework for Simulating, Visualizing & Analyzing Viral Mutations',
                
                # Navigation
                'mutation_tree': 'Mutation Tree',
                'epidemiology': 'Epidemiology',
                'structural_analysis': 'Structural Analysis',
                'ai_insights': 'AI Insights',
                'reports': 'Reports',
                
                # Configuration
                'configuration': 'Configuration',
                'mutation_simulation': 'Mutation Simulation',
                'reference_sequence': 'Reference Sequence',
                'mutation_rate': 'Mutation Rate',
                'max_generations': 'Max Generations',
                'branches_per_node': 'Branches per Node',
                'pruning_strategy': 'Pruning Strategy',
                'epidemiological_model': 'Epidemiological Model',
                'population_size': 'Population Size',
                'initial_infected': 'Initial Infected',
                
                # Actions
                'run_simulation': 'Run Simulation',
                'export_pdf': 'Export PDF Report',
                'export_csv': 'Export Data (CSV)',
                'export_fasta': 'Export Sequences (FASTA)',
                
                # Results
                'total_nodes': 'Total Nodes',
                'final_generation_size': 'Final Generation Size',
                'average_fitness': 'Average Fitness',
                'max_fitness': 'Max Fitness',
                'simulation_completed': 'Simulation completed successfully!',
                'simulation_failed': 'Simulation failed',
                
                # Analysis
                'mutation_analysis': 'Mutation Analysis',
                'fitness_evolution': 'Fitness Evolution Across Generations',
                'nodes_per_generation': 'Nodes per Generation',
                'top_mutations': 'Top Performing Mutations',
                'binding_energy': 'Binding Energy',
                'protein_interactions': 'Protein Interactions',
                'conservation_score': 'Conservation Score',
                'structural_impact': 'Structural Impact',
                
                # Epidemiology
                'epidemic_curves': 'Epidemic Curves',
                'peak_analysis': 'Peak Analysis',
                'peak_infections': 'Peak Infections',
                'peak_day': 'Peak Day',
                'susceptible': 'Susceptible',
                'infected': 'Infected',
                'recovered': 'Recovered',
                'vaccinated': 'Vaccinated',
                
                # AI Models
                'gnn_score': 'GNN Score',
                'transformer_score': 'Transformer Score',
                'ensemble_score': 'Ensemble Score',
                'prediction_confidence': 'Prediction Confidence',
                'enable_advanced_ai': 'Enable Advanced AI Models',
                
                # Status messages
                'running_simulation': 'Running mutation simulation...',
                'running_epidemiology': 'Running epidemiological simulation...',
                'generating_report': 'Generating report...',
                'no_data_available': 'No data available. Run a simulation first.',
                'error_occurred': 'An error occurred',
                
                # Scientific terms
                'amino_acid': 'Amino Acid',
                'protein': 'Protein',
                'mutation': 'Mutation',
                'fitness': 'Fitness',
                'generation': 'Generation',
                'sequence': 'Sequence',
                'structure': 'Structure',
                'binding': 'Binding',
                'interaction': 'Interaction',
                'conservation': 'Conservation',
                'stability': 'Stability',
                'transmissibility': 'Transmissibility',
                'virulence': 'Virulence',
                'immune_escape': 'Immune Escape',
                'alphafold': 'AlphaFold',
                'docking': 'Molecular Docking',
                'gnn_analysis': 'Graph Neural Network Analysis',
                'transformer_embedding': 'Transformer Embedding',
                'bayesian_optimization': 'Bayesian Optimization',
                'solvent_accessibility': 'Solvent Accessibility',
                'secondary_structure': 'Secondary Structure',
                'tertiary_structure': 'Tertiary Structure',
                'binding_affinity': 'Binding Affinity',
                'conformational_change': 'Conformational Change',
                'allosteric_effect': 'Allosteric Effect',
                'epitope': 'Epitope',
                'paratope': 'Paratope',
                'phylogenetic': 'Phylogenetic',
                'clade': 'Clade',
                'lineage': 'Lineage',
                'variant_of_concern': 'Variant of Concern',
                'variant_of_interest': 'Variant of Interest',
                'spike_protein': 'Spike Protein',
                'receptor_binding_domain': 'Receptor Binding Domain',
                'neutralizing_antibody': 'Neutralizing Antibody',
                'cross_reactivity': 'Cross Reactivity',
                'antigenic_drift': 'Antigenic Drift',
                'antigenic_shift': 'Antigenic Shift'
            },
            
            'es': {
                'app_title': 'IA de Simulación de Mutaciones Virales',
                'app_subtitle': 'Un Marco Integral Basado en IA para Simular, Visualizar y Analizar Mutaciones Virales',
                'mutation_tree': 'Árbol de Mutaciones',
                'epidemiology': 'Epidemiología',
                'structural_analysis': 'Análisis Estructural',
                'ai_insights': 'Perspectivas de IA',
                'reports': 'Informes',
                'configuration': 'Configuración',
                'mutation_simulation': 'Simulación de Mutaciones',
                'reference_sequence': 'Secuencia de Referencia',
                'mutation_rate': 'Tasa de Mutación',
                'max_generations': 'Generaciones Máximas',
                'branches_per_node': 'Ramas por Nodo',
                'pruning_strategy': 'Estrategia de Poda',
                'epidemiological_model': 'Modelo Epidemiológico',
                'population_size': 'Tamaño de Población',
                'initial_infected': 'Infectados Iniciales',
                'run_simulation': 'Ejecutar Simulación',
                'export_pdf': 'Exportar Informe PDF',
                'export_csv': 'Exportar Datos (CSV)',
                'export_fasta': 'Exportar Secuencias (FASTA)',
                'total_nodes': 'Nodos Totales',
                'final_generation_size': 'Tamaño de Generación Final',
                'average_fitness': 'Aptitud Promedio',
                'max_fitness': 'Aptitud Máxima',
                'simulation_completed': '¡Simulación completada exitosamente!',
                'simulation_failed': 'Simulación fallida',
                'mutation_analysis': 'Análisis de Mutaciones',
                'fitness_evolution': 'Evolución de Aptitud a Través de Generaciones',
                'nodes_per_generation': 'Nodos por Generación',
                'top_mutations': 'Mejores Mutaciones',
                'binding_energy': 'Energía de Enlace',
                'protein_interactions': 'Interacciones Proteicas',
                'conservation_score': 'Puntuación de Conservación',
                'structural_impact': 'Impacto Estructural',
                'epidemic_curves': 'Curvas Epidémicas',
                'peak_analysis': 'Análisis de Pico',
                'peak_infections': 'Infecciones Pico',
                'peak_day': 'Día Pico',
                'susceptible': 'Susceptible',
                'infected': 'Infectado',
                'recovered': 'Recuperado',
                'vaccinated': 'Vacunado',
                'gnn_score': 'Puntuación GNN',
                'transformer_score': 'Puntuación Transformer',
                'ensemble_score': 'Puntuación Conjunto',
                'prediction_confidence': 'Confianza de Predicción',
                'enable_advanced_ai': 'Habilitar Modelos IA Avanzados',
                'running_simulation': 'Ejecutando simulación de mutaciones...',
                'running_epidemiology': 'Ejecutando simulación epidemiológica...',
                'generating_report': 'Generando informe...',
                'no_data_available': 'No hay datos disponibles. Ejecute una simulación primero.',
                'error_occurred': 'Ocurrió un error',
                'amino_acid': 'Aminoácido',
                'protein': 'Proteína',
                'mutation': 'Mutación',
                'fitness': 'Aptitud',
                'generation': 'Generación',
                'sequence': 'Secuencia',
                'structure': 'Estructura',
                'binding': 'Enlace',
                'interaction': 'Interacción',
                'conservation': 'Conservación',
                'stability': 'Estabilidad',
                'transmissibility': 'Transmisibilidad',
                'virulence': 'Virulencia',
                'immune_escape': 'Escape Inmune',
                'alphafold': 'AlphaFold',
                'docking': 'Acoplamiento Molecular',
                'gnn_analysis': 'Análisis de Red Neuronal Gráfica',
                'transformer_embedding': 'Incrustación Transformer',
                'bayesian_optimization': 'Optimización Bayesiana',
                'solvent_accessibility': 'Accesibilidad al Solvente',
                'secondary_structure': 'Estructura Secundaria',
                'tertiary_structure': 'Estructura Terciaria',
                'binding_affinity': 'Afinidad de Enlace',
                'conformational_change': 'Cambio Conformacional',
                'allosteric_effect': 'Efecto Alostérico',
                'epitope': 'Epítopo',
                'paratope': 'Parátopo',
                'phylogenetic': 'Filogenético',
                'clade': 'Clado',
                'lineage': 'Linaje',
                'variant_of_concern': 'Variante de Preocupación',
                'variant_of_interest': 'Variante de Interés',
                'spike_protein': 'Proteína Espiga',
                'receptor_binding_domain': 'Dominio de Unión al Receptor',
                'neutralizing_antibody': 'Anticuerpo Neutralizante',
                'cross_reactivity': 'Reactividad Cruzada',
                'antigenic_drift': 'Deriva Antigénica',
                'antigenic_shift': 'Cambio Antigénico'
            },
            
            'fr': {
                'app_title': 'IA de Simulation de Mutations Virales',
                'app_subtitle': 'Un Cadre Complet Basé sur l\'IA pour Simuler, Visualiser et Analyser les Mutations Virales',
                'mutation_tree': 'Arbre de Mutations',
                'epidemiology': 'Épidémiologie',
                'structural_analysis': 'Analyse Structurelle',
                'ai_insights': 'Perspectives IA',
                'reports': 'Rapports',
                'configuration': 'Configuration',
                'mutation_simulation': 'Simulation de Mutations',
                'reference_sequence': 'Séquence de Référence',
                'mutation_rate': 'Taux de Mutation',
                'max_generations': 'Générations Maximales',
                'branches_per_node': 'Branches par Nœud',
                'pruning_strategy': 'Stratégie d\'Élagage',
                'epidemiological_model': 'Modèle Épidémiologique',
                'population_size': 'Taille de Population',
                'initial_infected': 'Infectés Initiaux',
                'run_simulation': 'Exécuter la Simulation',
                'export_pdf': 'Exporter Rapport PDF',
                'export_csv': 'Exporter Données (CSV)',
                'export_fasta': 'Exporter Séquences (FASTA)',
                'total_nodes': 'Nœuds Totaux',
                'final_generation_size': 'Taille de Génération Finale',
                'average_fitness': 'Aptitude Moyenne',
                'max_fitness': 'Aptitude Maximale',
                'simulation_completed': 'Simulation terminée avec succès!',
                'simulation_failed': 'Échec de la simulation',
                'mutation_analysis': 'Analyse des Mutations',
                'fitness_evolution': 'Évolution de l\'Aptitude à Travers les Générations',
                'nodes_per_generation': 'Nœuds par Génération',
                'top_mutations': 'Meilleures Mutations',
                'binding_energy': 'Énergie de Liaison',
                'protein_interactions': 'Interactions Protéiques',
                'conservation_score': 'Score de Conservation',
                'structural_impact': 'Impact Structurel',
                'epidemic_curves': 'Courbes Épidémiques',
                'peak_analysis': 'Analyse de Pic',
                'peak_infections': 'Infections de Pic',
                'peak_day': 'Jour de Pic',
                'susceptible': 'Susceptible',
                'infected': 'Infecté',
                'recovered': 'Guéri',
                'vaccinated': 'Vacciné',
                'gnn_score': 'Score GNN',
                'transformer_score': 'Score Transformer',
                'ensemble_score': 'Score d\'Ensemble',
                'prediction_confidence': 'Confiance de Prédiction',
                'enable_advanced_ai': 'Activer les Modèles IA Avancés',
                'running_simulation': 'Exécution de la simulation de mutations...',
                'running_epidemiology': 'Exécution de la simulation épidémiologique...',
                'generating_report': 'Génération du rapport...',
                'no_data_available': 'Aucune donnée disponible. Exécutez d\'abord une simulation.',
                'error_occurred': 'Une erreur s\'est produite',
                'amino_acid': 'Acide Aminé',
                'protein': 'Protéine',
                'mutation': 'Mutation',
                'fitness': 'Aptitude',
                'generation': 'Génération',
                'sequence': 'Séquence',
                'structure': 'Structure',
                'binding': 'Liaison',
                'interaction': 'Interaction',
                'conservation': 'Conservation',
                'stability': 'Stabilité',
                'transmissibility': 'Transmissibilité',
                'virulence': 'Virulence',
                'immune_escape': 'Échappement Immunitaire',
                'alphafold': 'AlphaFold',
                'docking': 'Amarrage Moléculaire',
                'gnn_analysis': 'Analyse de Réseau de Neurones Graphiques',
                'transformer_embedding': 'Plongement Transformer',
                'bayesian_optimization': 'Optimisation Bayésienne',
                'solvent_accessibility': 'Accessibilité au Solvant',
                'secondary_structure': 'Structure Secondaire',
                'tertiary_structure': 'Structure Tertiaire',
                'binding_affinity': 'Affinité de Liaison',
                'conformational_change': 'Changement Conformationnel',
                'allosteric_effect': 'Effet Allostérique',
                'epitope': 'Épitope',
                'paratope': 'Paratope',
                'phylogenetic': 'Phylogénétique',
                'clade': 'Clade',
                'lineage': 'Lignée',
                'variant_of_concern': 'Variant Préoccupant',
                'variant_of_interest': 'Variant d\'Intérêt',
                'spike_protein': 'Protéine Spike',
                'receptor_binding_domain': 'Domaine de Liaison au Récepteur',
                'neutralizing_antibody': 'Anticorps Neutralisant',
                'cross_reactivity': 'Réactivité Croisée',
                'antigenic_drift': 'Dérive Antigénique',
                'antigenic_shift': 'Saut Antigénique'
            },
            
            'de': {
                'app_title': 'KI für Virusmutations-Simulation',
                'app_subtitle': 'Ein Umfassendes KI-Basiertes Framework zur Simulation, Visualisierung und Analyse von Virusmutationen',
                'mutation_tree': 'Mutationsbaum',
                'epidemiology': 'Epidemiologie',
                'structural_analysis': 'Strukturanalyse',
                'ai_insights': 'KI-Erkenntnisse',
                'reports': 'Berichte',
                'configuration': 'Konfiguration',
                'mutation_simulation': 'Mutationssimulation',
                'reference_sequence': 'Referenzsequenz',
                'mutation_rate': 'Mutationsrate',
                'max_generations': 'Maximale Generationen',
                'run_simulation': 'Simulation Ausführen',
                'total_nodes': 'Gesamte Knoten',
                'average_fitness': 'Durchschnittliche Fitness',
                'simulation_completed': 'Simulation erfolgreich abgeschlossen!',
                'mutation_analysis': 'Mutationsanalyse',
                'susceptible': 'Anfällig',
                'infected': 'Infiziert',
                'recovered': 'Genesen',
                'vaccinated': 'Geimpft'
            },
            
            'zh': {
                'app_title': '病毒突变模拟人工智能',
                'app_subtitle': '基于人工智能的病毒突变模拟、可视化和分析综合框架',
                'mutation_tree': '突变树',
                'epidemiology': '流行病学',
                'structural_analysis': '结构分析',
                'ai_insights': 'AI洞察',
                'reports': '报告',
                'configuration': '配置',
                'mutation_simulation': '突变模拟',
                'reference_sequence': '参考序列',
                'mutation_rate': '突变率',
                'max_generations': '最大代数',
                'run_simulation': '运行模拟',
                'total_nodes': '总节点数',
                'average_fitness': '平均适应度',
                'simulation_completed': '模拟成功完成！',
                'mutation_analysis': '突变分析',
                'susceptible': '易感',
                'infected': '感染',
                'recovered': '康复',
                'vaccinated': '接种疫苗'
            },
            
            'ja': {
                'app_title': 'ウイルス変異シミュレーションAI',
                'app_subtitle': 'ウイルス変異のシミュレーション、可視化、分析のための包括的AIベースフレームワーク',
                'mutation_tree': '変異ツリー',
                'epidemiology': '疫学',
                'structural_analysis': '構造解析',
                'ai_insights': 'AI洞察',
                'reports': 'レポート',
                'configuration': '設定',
                'mutation_simulation': '変異シミュレーション',
                'reference_sequence': '参照配列',
                'mutation_rate': '変異率',
                'max_generations': '最大世代数',
                'run_simulation': 'シミュレーション実行',
                'total_nodes': '総ノード数',
                'average_fitness': '平均適応度',
                'simulation_completed': 'シミュレーションが正常に完了しました！',
                'mutation_analysis': '変異解析',
                'susceptible': '感受性',
                'infected': '感染',
                'recovered': '回復',
                'vaccinated': 'ワクチン接種済み'
            }
        }
    
    def _initialize_ai_translator(self):
        """Initialize AI-based translation if available"""
        if not TRANSFORMERS_AVAILABLE:
            return
        
        try:
            # Try to use a more advanced multilingual model
            self.ai_translator = pipeline(
                "translation",
                model="facebook/mbart-large-50-many-to-many-mmt",  # Better multilingual model
                device=-1  # Use CPU
            )
            print("Advanced AI translator (mBART) initialized successfully")
        except Exception:
            try:
                # Fallback to Helsinki model
                self.ai_translator = pipeline(
                    "translation",
                    model="Helsinki-NLP/opus-mt-en-mul",
                    device=-1
                )
                print("AI translator (Helsinki) initialized successfully")
            except Exception as e:
                print(f"Failed to initialize AI translator: {e}")
                self.ai_translator = None
    
    def set_language(self, language_code: str) -> bool:
        """Set the current language"""
        if language_code in self.supported_languages:
            self.current_language = language_code
            return True
        return False
    
    def get_supported_languages(self) -> Dict[str, LanguageConfig]:
        """Get all supported languages"""
        return self.supported_languages
    
    def translate(self, key: str, language: Optional[str] = None, 
                 fallback_to_ai: bool = True) -> str:
        """
        Translate a text key to the specified language
        
        Args:
            key: Translation key
            language: Target language code (uses current if None)
            fallback_to_ai: Whether to use AI translation if static not available
        
        Returns:
            Translated text or original key if translation not found
        """
        target_lang = language or self.current_language
        
        # Try static translation first
        if (target_lang in self.static_translations and 
            key in self.static_translations[target_lang]):
            return self.static_translations[target_lang][key]
        
        # Try English fallback
        if (target_lang != 'en' and 
            'en' in self.static_translations and 
            key in self.static_translations['en']):
            
            english_text = self.static_translations['en'][key]
            
            # Try AI translation if available and requested
            if fallback_to_ai and self.ai_translator and target_lang != 'en':
                try:
                    # Create language pair for translation
                    translation_task = f"translate English to {self.supported_languages[target_lang].name}"
                    result = self.ai_translator(english_text, 
                                              src_lang='en', 
                                              tgt_lang=target_lang)
                    
                    if result and len(result) > 0:
                        translated_text = result[0]['translation_text']
                        
                        # Cache the translation
                        if target_lang not in self.static_translations:
                            self.static_translations[target_lang] = {}
                        self.static_translations[target_lang][key] = translated_text
                        
                        return translated_text
                        
                except Exception as e:
                    print(f"AI translation failed for {key} to {target_lang}: {e}")
            
            # Return English text if AI translation fails
            return english_text
        
        # Return the key itself if no translation found
        return key
    
    def translate_dict(self, data: Dict[str, Any], language: Optional[str] = None) -> Dict[str, Any]:
        """Translate all string values in a dictionary"""
        target_lang = language or self.current_language
        
        if target_lang == 'en':
            return data  # No translation needed
        
        translated = {}
        for key, value in data.items():
            if isinstance(value, str):
                translated[key] = self.translate(value, target_lang)
            elif isinstance(value, dict):
                translated[key] = self.translate_dict(value, target_lang)
            elif isinstance(value, list):
                translated[key] = [
                    self.translate(item, target_lang) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                translated[key] = value
        
        return translated
    
    def get_language_direction(self, language: Optional[str] = None) -> str:
        """Get text direction for the language (ltr or rtl)"""
        target_lang = language or self.current_language
        
        if target_lang in self.supported_languages:
            return 'rtl' if self.supported_languages[target_lang].rtl else 'ltr'
        
        return 'ltr'
    
    def format_number(self, number: float, language: Optional[str] = None) -> str:
        """Format number according to language conventions"""
        target_lang = language or self.current_language
        
        # Simple formatting - in a real implementation, use locale-specific formatting
        if target_lang in ['de', 'fr', 'es', 'it']:
            # European formatting (comma as decimal separator)
            return f"{number:.3f}".replace('.', ',')
        else:
            # Default formatting
            return f"{number:.3f}"
    
    def get_date_format(self, language: Optional[str] = None) -> str:
        """Get date format string for the language"""
        target_lang = language or self.current_language
        
        date_formats = {
            'en': '%Y-%m-%d',
            'es': '%d/%m/%Y',
            'fr': '%d/%m/%Y',
            'de': '%d.%m.%Y',
            'it': '%d/%m/%Y',
            'pt': '%d/%m/%Y',
            'ru': '%d.%m.%Y',
            'zh': '%Y年%m月%d日',
            'ja': '%Y年%m月%d日',
            'ko': '%Y년 %m월 %d일',
            'ar': '%d/%m/%Y',
            'hi': '%d/%m/%Y'
        }
        
        return date_formats.get(target_lang, '%Y-%m-%d')
    
    def add_custom_translation(self, language: str, key: str, translation: str):
        """Add a custom translation"""
        if language not in self.static_translations:
            self.static_translations[language] = {}
        
        self.static_translations[language][key] = translation
    
    def export_translations(self, language: str, filename: str):
        """Export translations to JSON file"""
        if language in self.static_translations:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.static_translations[language], f, 
                         ensure_ascii=False, indent=2)
    
    def import_translations(self, language: str, filename: str):
        """Import translations from JSON file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                translations = json.load(f)
                
            if language not in self.static_translations:
                self.static_translations[language] = {}
            
            self.static_translations[language].update(translations)
            return True
            
        except Exception as e:
            print(f"Failed to import translations: {e}")
            return False
    
    def get_translation_coverage(self, language: str) -> float:
        """Get translation coverage percentage for a language"""
        if 'en' not in self.static_translations or language not in self.static_translations:
            return 0.0
        
        english_keys = set(self.static_translations['en'].keys())
        target_keys = set(self.static_translations[language].keys())
        
        if not english_keys:
            return 100.0
        
        coverage = len(english_keys.intersection(target_keys)) / len(english_keys)
        return coverage * 100
    
    def get_missing_translations(self, language: str) -> List[str]:
        """Get list of missing translation keys for a language"""
        if 'en' not in self.static_translations:
            return []
        
        english_keys = set(self.static_translations['en'].keys())
        
        if language not in self.static_translations:
            return list(english_keys)
        
        target_keys = set(self.static_translations[language].keys())
        missing = english_keys - target_keys
        
        return list(missing)
    
    def translate_scientific_text(self, text: str, language: Optional[str] = None, 
                                context: str = "virology") -> str:
        """
        Translate scientific text with context awareness
        
        Args:
            text: Text to translate
            language: Target language
            context: Scientific context (virology, biochemistry, epidemiology)
        
        Returns:
            Translated text with scientific accuracy
        """
        target_lang = language or self.current_language
        
        if target_lang == 'en':
            return text
        
        # Scientific term mapping for better accuracy
        scientific_terms = {
            'virology': {
                'SARS-CoV-2': 'SARS-CoV-2',  # Keep as is
                'COVID-19': 'COVID-19',      # Keep as is
                'spike protein': self.translate('spike_protein', target_lang),
                'RBD': self.translate('receptor_binding_domain', target_lang),
                'ACE2': 'ACE2',  # Keep as is
                'neutralizing antibodies': self.translate('neutralizing_antibody', target_lang),
                'variant of concern': self.translate('variant_of_concern', target_lang),
                'immune escape': self.translate('immune_escape', target_lang)
            },
            'biochemistry': {
                'protein folding': 'protein folding',
                'molecular docking': self.translate('docking', target_lang),
                'binding affinity': self.translate('binding_affinity', target_lang),
                'conformational change': self.translate('conformational_change', target_lang),
                'allosteric effect': self.translate('allosteric_effect', target_lang)
            },
            'epidemiology': {
                'basic reproduction number': 'R₀',
                'effective reproduction number': 'Rₑ',
                'herd immunity': 'herd immunity',
                'case fatality rate': 'CFR',
                'infection fatality rate': 'IFR'
            }
        }
        
        # Replace scientific terms before translation
        processed_text = text
        if context in scientific_terms:
            for term, translation in scientific_terms[context].items():
                processed_text = processed_text.replace(term, f"[{translation}]")
        
        # Try AI translation
        if self.ai_translator:
            try:
                # Use appropriate language codes for mBART
                lang_codes = {
                    'es': 'es_XX', 'fr': 'fr_XX', 'de': 'de_DE', 'it': 'it_IT',
                    'pt': 'pt_XX', 'ru': 'ru_RU', 'zh': 'zh_CN', 'ja': 'ja_XX',
                    'ko': 'ko_KR', 'ar': 'ar_AR', 'hi': 'hi_IN', 'tr': 'tr_TR'
                }
                
                src_lang = 'en_XX'
                tgt_lang = lang_codes.get(target_lang, target_lang)
                
                result = self.ai_translator(processed_text, 
                                          src_lang=src_lang, 
                                          tgt_lang=tgt_lang)
                
                if result and len(result) > 0:
                    translated = result[0]['translation_text']
                    
                    # Restore scientific terms
                    translated = translated.replace('[', '').replace(']', '')
                    
                    return translated
                    
            except Exception as e:
                print(f"Scientific AI translation failed: {e}")
        
        # Fallback to regular translation
        return self.translate(text, target_lang)
    
    def batch_translate_interface(self, interface_dict: Dict[str, str], 
                                language: Optional[str] = None) -> Dict[str, str]:
        """
        Batch translate interface elements efficiently
        
        Args:
            interface_dict: Dictionary of key-value pairs to translate
            language: Target language
        
        Returns:
            Dictionary with translated values
        """
        target_lang = language or self.current_language
        
        if target_lang == 'en':
            return interface_dict
        
        translated = {}
        
        # Group translations by availability
        static_keys = []
        ai_keys = []
        
        for key, value in interface_dict.items():
            if (target_lang in self.static_translations and 
                key in self.static_translations[target_lang]):
                translated[key] = self.static_translations[target_lang][key]
            else:
                static_keys.append(key)
                ai_keys.append(value)
        
        # Batch AI translation for missing keys
        if ai_keys and self.ai_translator:
            try:
                # Translate in batches for efficiency
                batch_size = 10
                for i in range(0, len(ai_keys), batch_size):
                    batch_texts = ai_keys[i:i+batch_size]
                    batch_keys = static_keys[i:i+batch_size]
                    
                    for j, text in enumerate(batch_texts):
                        try:
                            result = self.ai_translator(text, 
                                                      src_lang='en', 
                                                      tgt_lang=target_lang)
                            if result and len(result) > 0:
                                translated[batch_keys[j]] = result[0]['translation_text']
                            else:
                                translated[batch_keys[j]] = text
                        except:
                            translated[batch_keys[j]] = text
                            
            except Exception as e:
                print(f"Batch translation failed: {e}")
                # Fallback to original values
                for key, value in zip(static_keys, ai_keys):
                    translated[key] = value
        else:
            # No AI available, use original values
            for key, value in zip(static_keys, ai_keys):
                translated[key] = value
        
        return translated
    
    def get_language_specific_formatting(self, language: Optional[str] = None) -> Dict[str, Any]:
        """Get language-specific formatting preferences"""
        target_lang = language or self.current_language
        
        formatting = {
            'decimal_separator': '.',
            'thousands_separator': ',',
            'currency_symbol': '$',
            'date_format': self.get_date_format(target_lang),
            'time_format': '%H:%M:%S',
            'text_direction': self.get_language_direction(target_lang),
            'font_family': 'Arial, sans-serif'
        }
        
        # Language-specific overrides
        if target_lang in ['de', 'fr', 'es', 'it', 'pt']:
            formatting['decimal_separator'] = ','
            formatting['thousands_separator'] = '.'
        
        if target_lang in ['ar', 'he']:
            formatting['text_direction'] = 'rtl'
            formatting['font_family'] = 'Arial, "Times New Roman", serif'
        
        if target_lang in ['zh', 'ja', 'ko']:
            formatting['font_family'] = '"Microsoft YaHei", "Hiragino Sans GB", sans-serif'
        
        if target_lang == 'ru':
            formatting['font_family'] = '"Times New Roman", serif'
        
        return formatting


# Global instance for easy access
_global_translator = None

def get_translator() -> MultilingualSupport:
    """Get the global translator instance"""
    global _global_translator
    if _global_translator is None:
        _global_translator = MultilingualSupport()
    return _global_translator

def t(key: str, language: Optional[str] = None) -> str:
    """Shorthand function for translation"""
    return get_translator().translate(key, language)

def set_language(language_code: str) -> bool:
    """Set the global language"""
    return get_translator().set_language(language_code)

def get_current_language() -> str:
    """Get the current language code"""
    return get_translator().current_language

def get_supported_languages() -> Dict[str, LanguageConfig]:
    """Get all supported languages"""
    return get_translator().get_supported_languages()


# Example usage and testing
if __name__ == "__main__":
    # Initialize translator
    translator = MultilingualSupport()
    
    # Test translations
    test_keys = ['app_title', 'mutation_tree', 'run_simulation', 'total_nodes']
    test_languages = ['en', 'es', 'fr', 'de', 'zh', 'ja']
    
    print("Translation Test Results:")
    print("=" * 50)
    
    for lang in test_languages:
        print(f"\n{translator.supported_languages[lang].native_name} ({lang}):")
        print("-" * 30)
        
        for key in test_keys:
            translation = translator.translate(key, lang)
            print(f"{key}: {translation}")
        
        # Show coverage
        coverage = translator.get_translation_coverage(lang)
        print(f"Coverage: {coverage:.1f}%")
    
    # Test AI translation (if available)
    if translator.ai_translator:
        print("\n" + "=" * 50)
        print("AI Translation Test:")
        print("=" * 50)
        
        test_text = "Advanced protein structure analysis"
        for lang in ['es', 'fr', 'de']:
            try:
                result = translator.ai_translator(test_text, src_lang='en', tgt_lang=lang)
                if result:
                    print(f"{lang}: {result[0]['translation_text']}")
            except Exception as e:
                print(f"{lang}: Translation failed - {e}")
    
    print("\nMultilingual support system initialized successfully!")


# Streamlit integration utilities
def create_language_selector(translator: MultilingualSupport, key: str = "language_selector"):
    """Create a Streamlit language selector widget"""
    try:
        import streamlit as st
        
        languages = translator.get_supported_languages()
        language_options = {
            f"{config.native_name} ({config.name})": code 
            for code, config in languages.items()
        }
        
        current_display = None
        for display, code in language_options.items():
            if code == translator.current_language:
                current_display = display
                break
        
        selected_display = st.selectbox(
            translator.translate('select_language', 'en'),
            options=list(language_options.keys()),
            index=list(language_options.keys()).index(current_display) if current_display else 0,
            key=key
        )
        
        selected_code = language_options[selected_display]
        
        if selected_code != translator.current_language:
            translator.set_language(selected_code)
            st.rerun()
        
        return selected_code
        
    except ImportError:
        print("Streamlit not available for language selector")
        return translator.current_language

def apply_language_styling(translator: MultilingualSupport):
    """Apply language-specific CSS styling to Streamlit"""
    try:
        import streamlit as st
        
        formatting = translator.get_language_specific_formatting()
        
        css = f"""
        <style>
        .main {{
            direction: {formatting['text_direction']};
            font-family: {formatting['font_family']};
        }}
        
        .stSelectbox label, .stSlider label, .stTextInput label {{
            direction: {formatting['text_direction']};
            text-align: {'right' if formatting['text_direction'] == 'rtl' else 'left'};
        }}
        
        .metric-container {{
            direction: {formatting['text_direction']};
        }}
        
        .stDataFrame {{
            direction: ltr;  /* Keep data tables LTR for readability */
        }}
        </style>
        """
        
        st.markdown(css, unsafe_allow_html=True)
        
    except ImportError:
        pass

def translate_streamlit_metrics(metrics_dict: Dict[str, Any], translator: MultilingualSupport) -> Dict[str, Any]:
    """Translate Streamlit metrics dictionary"""
    translated = {}
    
    for key, value in metrics_dict.items():
        if isinstance(value, dict) and 'label' in value:
            translated[key] = value.copy()
            translated[key]['label'] = translator.translate(value['label'])
        else:
            translated_key = translator.translate(key)
            translated[translated_key] = value
    
    return translated

def format_scientific_number(number: float, translator: MultilingualSupport, 
                           precision: int = 3, scientific_notation: bool = False) -> str:
    """Format numbers according to language conventions with scientific notation support"""
    
    formatting = translator.get_language_specific_formatting()
    
    if scientific_notation and (abs(number) >= 1e6 or abs(number) <= 1e-3):
        # Use scientific notation for very large or small numbers
        formatted = f"{number:.{precision}e}"
    else:
        # Regular formatting
        formatted = f"{number:.{precision}f}"
    
    # Apply language-specific decimal separator
    if formatting['decimal_separator'] != '.':
        formatted = formatted.replace('.', formatting['decimal_separator'])
    
    return formatted

# Advanced translation caching system
class TranslationCache:
    """Cache system for translations to improve performance"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
    
    def get(self, key: str, language: str) -> Optional[str]:
        """Get cached translation"""
        cache_key = f"{language}:{key}"
        
        if cache_key in self.cache:
            self.access_count[cache_key] = self.access_count.get(cache_key, 0) + 1
            return self.cache[cache_key]
        
        return None
    
    def set(self, key: str, language: str, translation: str):
        """Cache a translation"""
        cache_key = f"{language}:{key}"
        
        # Remove least used items if cache is full
        if len(self.cache) >= self.max_size:
            least_used = min(self.access_count.items(), key=lambda x: x[1])
            del self.cache[least_used[0]]
            del self.access_count[least_used[0]]
        
        self.cache[cache_key] = translation
        self.access_count[cache_key] = 1
    
    def clear(self):
        """Clear the cache"""
        self.cache.clear()
        self.access_count.clear()

# Enhanced global translator with caching
class CachedMultilingualSupport(MultilingualSupport):
    """Enhanced multilingual support with caching"""
    
    def __init__(self, default_language: str = 'en', cache_size: int = 1000):
        super().__init__(default_language)
        self.cache = TranslationCache(cache_size)
    
    def translate(self, key: str, language: Optional[str] = None, 
                 fallback_to_ai: bool = True) -> str:
        """Enhanced translate method with caching"""
        target_lang = language or self.current_language
        
        # Check cache first
        cached = self.cache.get(key, target_lang)
        if cached:
            return cached
        
        # Get translation using parent method
        translation = super().translate(key, target_lang, fallback_to_ai)
        
        # Cache the result
        self.cache.set(key, target_lang, translation)
        
        return translation

# Create enhanced global instance
_global_cached_translator = None

def get_cached_translator() -> CachedMultilingualSupport:
    """Get the global cached translator instance"""
    global _global_cached_translator
    if _global_cached_translator is None:
        _global_cached_translator = CachedMultilingualSupport()
    return _global_cached_translator

# Update convenience functions to use cached translator
def t_cached(key: str, language: Optional[str] = None) -> str:
    """Cached shorthand function for translation"""
    return get_cached_translator().translate(key, language)

def translate_scientific(text: str, language: Optional[str] = None, 
                        context: str = "virology") -> str:
    """Translate scientific text with context"""
    return get_cached_translator().translate_scientific_text(text, language, context)

# Batch processing utilities
def translate_dataframe_columns(df, translator: MultilingualSupport, 
                              column_mapping: Optional[Dict[str, str]] = None):
    """Translate DataFrame column names"""
    try:
        import pandas as pd
        
        if column_mapping:
            # Use provided mapping
            translated_columns = {}
            for col in df.columns:
                if col in column_mapping:
                    translated_columns[col] = translator.translate(column_mapping[col])
                else:
                    translated_columns[col] = col
        else:
            # Auto-translate column names
            translated_columns = {
                col: translator.translate(col.lower().replace(' ', '_').replace('-', '_'))
                for col in df.columns
            }
        
        return df.rename(columns=translated_columns)
        
    except ImportError:
        print("Pandas not available for DataFrame translation")
        return df

def create_multilingual_help_text(translator: MultilingualSupport) -> Dict[str, str]:
    """Create comprehensive help text in current language"""
    
    help_keys = [
        'mutation_rate_help',
        'max_generations_help', 
        'pruning_strategy_help',
        'population_size_help',
        'ai_models_help',
        'export_help'
    ]
    
    # Add help text to translations if not present
    help_translations = {
        'en': {
            'mutation_rate_help': 'Probability of mutation per generation (0.0-1.0). Higher values create more diverse mutation trees.',
            'max_generations_help': 'Maximum number of generations to simulate. More generations allow for more complex evolutionary patterns.',
            'pruning_strategy_help': 'Method to limit tree size: Top-K keeps best mutations, Threshold removes low-fitness mutations.',
            'population_size_help': 'Initial population size for epidemiological simulation. Larger populations show more realistic spread patterns.',
            'ai_models_help': 'Enable advanced AI models (GNN, Transformer) for enhanced mutation impact prediction.',
            'export_help': 'Export simulation results in various formats: PDF reports, CSV data, FASTA sequences.'
        }
    }
    
    # Add to translator if not present
    current_lang = translator.current_language
    if current_lang not in translator.static_translations:
        translator.static_translations[current_lang] = {}
    
    if 'en' in help_translations:
        for key, text in help_translations['en'].items():
            if key not in translator.static_translations.get(current_lang, {}):
                translator.add_custom_translation(current_lang, key, 
                                                translator.translate_scientific_text(text, current_lang))
    
    # Return translated help text
    return {key: translator.translate(key) for key in help_keys}

# Testing and validation utilities
def validate_translation_completeness(translator: MultilingualSupport) -> Dict[str, float]:
    """Validate translation completeness for all languages"""
    results = {}
    
    for lang_code in translator.supported_languages.keys():
        coverage = translator.get_translation_coverage(lang_code)
        results[lang_code] = coverage
    
    return results

def generate_translation_report(translator: MultilingualSupport) -> str:
    """Generate a comprehensive translation status report"""
    
    report = ["=== Translation Status Report ===\n"]
    
    completeness = validate_translation_completeness(translator)
    
    for lang_code, coverage in completeness.items():
        lang_config = translator.supported_languages[lang_code]
        missing_count = len(translator.get_missing_translations(lang_code))
        
        status = "✅ Complete" if coverage == 100 else f"⚠️  {coverage:.1f}% ({missing_count} missing)"
        
        report.append(f"{lang_config.native_name} ({lang_code}): {status}")
    
    report.append(f"\nAI Translation Available: {'✅ Yes' if translator.ai_translator else '❌ No'}")
    report.append(f"Total Supported Languages: {len(translator.supported_languages)}")
    
    return "\n".join(report)

if __name__ == "__main__":
    # Enhanced testing
    print("🌍 Enhanced Multilingual Support System Test")
    print("=" * 50)
    
    # Test cached translator
    translator = get_cached_translator()
    
    # Test scientific translation
    scientific_text = "The spike protein shows high binding affinity to ACE2 receptor"
    
    print("\n🧬 Scientific Translation Test:")
    for lang in ['es', 'fr', 'de']:
        translated = translate_scientific(scientific_text, lang, 'virology')
        print(f"{lang}: {translated}")
    
    # Test batch translation
    interface_elements = {
        'run_simulation': 'Run Simulation',
        'mutation_analysis': 'Mutation Analysis', 
        'export_pdf': 'Export PDF Report'
    }
    
    print("\n📊 Batch Translation Test:")
    for lang in ['es', 'fr']:
        batch_result = translator.batch_translate_interface(interface_elements, lang)
        print(f"{lang}: {batch_result}")
    
    # Generate translation report
    print("\n📋 Translation Report:")
    print(generate_translation_report(translator))
    
    print("\n✅ Enhanced multilingual system ready for tier-S++ deployment!")