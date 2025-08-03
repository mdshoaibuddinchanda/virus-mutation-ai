"""
Data Serialization Utilities - Lightweight serialization formats for optimized I/O
"""
import numpy as np
import pickle
import gzip
import json
import os
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

class LightweightSerializer:
    """Lightweight serialization utilities for optimized data I/O"""
    
    def __init__(self, compression_level: int = 6, disable_logging: bool = False):
        self.compression_level = compression_level
        
        # Disable logging during peak computation if requested
        if disable_logging:
            logging.getLogger().setLevel(logging.ERROR)
    
    def save_mutation_tree_compressed(self, tree_data: Dict, filename: str) -> str:
        """Save mutation tree using compressed numpy format"""
        try:
            # Extract key data for compression
            sequences = []
            fitness_scores = []
            generations = []
            mutations_list = []
            
            for node_id, node in tree_data.items():
                if hasattr(node, 'sequence'):
                    sequences.append(node.sequence)
                    fitness_scores.append(node.fitness)
                    generations.append(node.generation)
                    mutations_list.append(node.mutations)
                else:
                    # Handle dict format
                    sequences.append(node.get('sequence', ''))
                    fitness_scores.append(node.get('fitness', 0.0))
                    generations.append(node.get('generation', 0))
                    mutations_list.append(node.get('mutations', []))
            
            # Convert to numpy arrays for compression
            sequences_array = np.array(sequences, dtype=object)
            fitness_array = np.array(fitness_scores, dtype=np.float32)
            generations_array = np.array(generations, dtype=np.int32)
            
            # Save with compression
            np.savez_compressed(
                filename,
                sequences=sequences_array,
                fitness_scores=fitness_array,
                generations=generations_array,
                mutations=mutations_list,
                node_ids=list(tree_data.keys()),
                metadata={
                    'total_nodes': len(tree_data),
                    'timestamp': datetime.now().isoformat(),
                    'compression_level': self.compression_level
                }
            )
            
            return filename
            
        except Exception as e:
            print(f"Error saving mutation tree: {e}")
            return None
    
    def load_mutation_tree_compressed(self, filename: str) -> Dict:
        """Load mutation tree from compressed numpy format"""
        try:
            data = np.load(filename, allow_pickle=True)
            
            # Reconstruct tree
            tree_data = {}
            node_ids = data['node_ids']
            sequences = data['sequences']
            fitness_scores = data['fitness_scores']
            generations = data['generations']
            mutations = data['mutations']
            
            for i, node_id in enumerate(node_ids):
                tree_data[node_id] = {
                    'sequence': sequences[i],
                    'fitness': float(fitness_scores[i]),
                    'generation': int(generations[i]),
                    'mutations': mutations[i]
                }
            
            return tree_data
            
        except Exception as e:
            print(f"Error loading mutation tree: {e}")
            return {}
    
    def save_sequences_compressed(self, sequences: List[str], filename: str) -> str:
        """Save sequences using compressed pickle + gzip"""
        try:
            with gzip.open(filename, 'wb', compresslevel=self.compression_level) as f:
                pickle.dump(sequences, f)
            return filename
        except Exception as e:
            print(f"Error saving sequences: {e}")
            return None
    
    def load_sequences_compressed(self, filename: str) -> List[str]:
        """Load sequences from compressed pickle + gzip"""
        try:
            with gzip.open(filename, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading sequences: {e}")
            return []
    
    def save_dataframe_optimized(self, df: 'pd.DataFrame', filename: str) -> str:
        """Save DataFrame using pyarrow for optimal performance"""
        if not PANDAS_AVAILABLE or not PYARROW_AVAILABLE:
            # Fallback to pickle
            return self.save_dataframe_pickle(df, filename)
        
        try:
            # Convert to pyarrow table and save as parquet
            table = pa.Table.from_pandas(df)
            pq.write_table(table, filename, compression='gzip')
            return filename
        except Exception as e:
            print(f"Error saving DataFrame with pyarrow: {e}")
            return self.save_dataframe_pickle(df, filename)
    
    def load_dataframe_optimized(self, filename: str) -> 'pd.DataFrame':
        """Load DataFrame using pyarrow for optimal performance"""
        if not PANDAS_AVAILABLE or not PYARROW_AVAILABLE:
            # Fallback to pickle
            return self.load_dataframe_pickle(filename)
        
        try:
            # Load from parquet
            table = pq.read_table(filename)
            return table.to_pandas()
        except Exception as e:
            print(f"Error loading DataFrame with pyarrow: {e}")
            return self.load_dataframe_pickle(filename)
    
    def save_dataframe_pickle(self, df: 'pd.DataFrame', filename: str) -> str:
        """Save DataFrame using pickle + gzip (fallback)"""
        try:
            with gzip.open(filename, 'wb', compresslevel=self.compression_level) as f:
                pickle.dump(df, f)
            return filename
        except Exception as e:
            print(f"Error saving DataFrame: {e}")
            return None
    
    def load_dataframe_pickle(self, filename: str) -> 'pd.DataFrame':
        """Load DataFrame using pickle + gzip (fallback)"""
        try:
            with gzip.open(filename, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading DataFrame: {e}")
            return pd.DataFrame()
    
    def save_simulation_report_compressed(self, report_data: Dict, filename: str) -> str:
        """Save simulation report using compressed JSON"""
        try:
            # Compress JSON with gzip
            json_str = json.dumps(report_data, default=str)
            with gzip.open(filename, 'wt', encoding='utf-8', compresslevel=self.compression_level) as f:
                f.write(json_str)
            return filename
        except Exception as e:
            print(f"Error saving simulation report: {e}")
            return None
    
    def load_simulation_report_compressed(self, filename: str) -> Dict:
        """Load simulation report from compressed JSON"""
        try:
            with gzip.open(filename, 'rt', encoding='utf-8') as f:
                json_str = f.read()
            return json.loads(json_str)
        except Exception as e:
            print(f"Error loading simulation report: {e}")
            return {}
    
    def batch_save_mutations(self, mutations_list: List[List], filename: str) -> str:
        """Save batch of mutations using compressed numpy"""
        try:
            # Convert to numpy arrays
            positions = []
            from_aas = []
            to_aas = []
            
            for mutations in mutations_list:
                for pos, from_aa, to_aa in mutations:
                    positions.append(pos)
                    from_aas.append(from_aa)
                    to_aas.append(to_aa)
            
            np.savez_compressed(
                filename,
                positions=np.array(positions, dtype=np.int32),
                from_aas=np.array(from_aas, dtype=object),
                to_aas=np.array(to_aas, dtype=object),
                mutation_counts=np.array([len(m) for m in mutations_list], dtype=np.int32)
            )
            
            return filename
        except Exception as e:
            print(f"Error saving mutations batch: {e}")
            return None
    
    def batch_load_mutations(self, filename: str) -> List[List]:
        """Load batch of mutations from compressed numpy"""
        try:
            data = np.load(filename, allow_pickle=True)
            
            positions = data['positions']
            from_aas = data['from_aas']
            to_aas = data['to_aas']
            mutation_counts = data['mutation_counts']
            
            # Reconstruct mutations
            mutations_list = []
            idx = 0
            for count in mutation_counts:
                mutations = []
                for i in range(count):
                    mutations.append((int(positions[idx]), from_aas[idx], to_aas[idx]))
                    idx += 1
                mutations_list.append(mutations)
            
            return mutations_list
        except Exception as e:
            print(f"Error loading mutations batch: {e}")
            return []
    
    def get_file_size_mb(self, filename: str) -> float:
        """Get file size in MB"""
        try:
            return os.path.getsize(filename) / (1024 * 1024)
        except:
            return 0.0
    
    def cleanup_old_files(self, directory: str, max_age_days: int = 7):
        """Clean up old serialized files"""
        try:
            current_time = datetime.now()
            for filename in os.listdir(directory):
                filepath = os.path.join(directory, filename)
                if os.path.isfile(filepath):
                    file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                    if (current_time - file_time).days > max_age_days:
                        os.remove(filepath)
                        print(f"🗑️ Cleaned up old file: {filename}")
        except Exception as e:
            print(f"Error cleaning up files: {e}")

# Global serializer instance
serializer = LightweightSerializer()

def save_data_optimized(data: Any, filename: str, data_type: str = "auto") -> str:
    """Save data using optimal serialization method"""
    if data_type == "auto":
        if isinstance(data, dict) and "tree" in str(data):
            return serializer.save_mutation_tree_compressed(data, filename)
        elif isinstance(data, list) and all(isinstance(x, str) for x in data):
            return serializer.save_sequences_compressed(data, filename)
        elif PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            return serializer.save_dataframe_optimized(data, filename)
        else:
            return serializer.save_simulation_report_compressed(data, filename)
    elif data_type == "tree":
        return serializer.save_mutation_tree_compressed(data, filename)
    elif data_type == "sequences":
        return serializer.save_sequences_compressed(data, filename)
    elif data_type == "dataframe":
        return serializer.save_dataframe_optimized(data, filename)
    elif data_type == "report":
        return serializer.save_simulation_report_compressed(data, filename)
    else:
        return serializer.save_simulation_report_compressed(data, filename)

def load_data_optimized(filename: str, data_type: str = "auto") -> Any:
    """Load data using optimal deserialization method"""
    if data_type == "auto":
        if "tree" in filename or filename.endswith(".npz"):
            return serializer.load_mutation_tree_compressed(filename)
        elif "sequences" in filename or filename.endswith(".pkl.gz"):
            return serializer.load_sequences_compressed(filename)
        elif filename.endswith(".parquet"):
            return serializer.load_dataframe_optimized(filename)
        else:
            return serializer.load_simulation_report_compressed(filename)
    elif data_type == "tree":
        return serializer.load_mutation_tree_compressed(filename)
    elif data_type == "sequences":
        return serializer.load_sequences_compressed(filename)
    elif data_type == "dataframe":
        return serializer.load_dataframe_optimized(filename)
    elif data_type == "report":
        return serializer.load_simulation_report_compressed(filename)
    else:
        return serializer.load_simulation_report_compressed(filename) 