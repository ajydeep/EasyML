from typing import List, Dict, Any, Union, Optional, Callable
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import re
import os
from scipy import sparse
import logging
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import joblib
from document_parser import DocumentParser, ParsedDocument  # Import parser
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ClusteringConfig:
    """Configuration for text clustering pipeline."""
    max_features: int = 5000           # Maximum number of features for TF-IDF
    n_components: int = 100            # Number of components for dimensionality reduction
    random_state: int = 42             # Random seed for reproducibility
    n_clusters_range: tuple = (2, 20)  # Range of clusters to try
    min_cluster_size: int = 5          # Minimum cluster size
    n_jobs: int = -1                   # Number of parallel jobs (-1 for all cores)
    progress_callback: Optional[Callable[[int], None]] = None  # Add this for UI progress updates
    
    def validate(self):
        """Validate configuration parameters."""
        if self.max_features < 1:
            raise ValueError("max_features must be positive")
        if self.n_components >= self.max_features:
            raise ValueError("n_components must be less than max_features")
        if self.n_clusters_range[0] < 2:
            raise ValueError("minimum number of clusters must be at least 2")
        if self.n_clusters_range[0] >= self.n_clusters_range[1]:
            raise ValueError("invalid cluster range")

class TextPreprocessor:
    """Text preprocessing pipeline with parallel processing capability."""
    
    def __init__(self, segment_size: int = 5):
        """Initialize the text preprocessor with proper NLTK data download."""
        self.segment_size = segment_size
        self.lemmatizer = None
        self.stop_words = None
        self.parser = None
        self._initialize_nltk()
        
    def _initialize_nltk(self):
        """Initialize NLTK resources with proper error handling."""
        try:
            # Create nltk_data directory if it doesn't exist
            nltk_data_dir = os.path.expanduser('~/nltk_data')
            os.makedirs(nltk_data_dir, exist_ok=True)
            
            # Download required NLTK data with retry mechanism
            required_packages = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
            max_retries = 3
            
            for package in required_packages:
                for attempt in range(max_retries):
                    try:
                        nltk.download(package, quiet=True, raise_on_error=True)
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise Exception(f"Failed to download NLTK package {package}: {str(e)}")
                        time.sleep(1)  # Wait before retry
            
            # Initialize components after successful download
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            self.parser = DocumentParser()
            
            logger.info("TextPreprocessor initialized successfully with NLTK resources")
            
        except Exception as e:
            logger.error(f"Failed to initialize NLTK resources: {str(e)}")
            raise

    def _segment_text(self, text: str) -> List[str]:
        """Split text into segments with error handling."""
        try:
            # Fallback to period-based splitting if NLTK fails
            try:
                sentences = sent_tokenize(text)
            except Exception as e:
                logger.warning(f"NLTK sentence tokenization failed: {str(e)}")
                # Simple period-based splitting as fallback
                sentences = [s.strip() for s in text.split('.') if s.strip()]
            
            # Remove very short sentences
            sentences = [s for s in sentences if len(s.split()) >= 3]
            
            if not sentences:
                return [text]  # Return original text if no valid sentences
            
            # Create segments
            segments = []
            current_segment = []
            
            for sentence in sentences:
                current_segment.append(sentence)
                if len(current_segment) >= self.segment_size:
                    segments.append(' '.join(current_segment))
                    current_segment = []
            
            # Add remaining sentences
            if current_segment:
                segments.append(' '.join(current_segment))
            
            return segments if segments else [text]
            
        except Exception as e:
            logger.warning(f"Text segmentation failed: {str(e)}")
            return [text]
        
    def _clean_text(self, text: str) -> str:
        """Clean text with robust error handling."""
        if not text or not isinstance(text, str):
            return ""
            
        try:
            # Basic cleaning (lowercase and special characters)
            text = text.lower()
            text = re.sub(r'[^\w\s]', ' ', text)
            
            # Tokenization with error handling
            try:
                tokens = word_tokenize(text)
            except Exception as e:
                logger.warning(f"Tokenization failed, falling back to simple split: {str(e)}")
                tokens = text.split()
            
            # Lemmatization and stopword removal with error handling
            cleaned_tokens = []
            for token in tokens:
                if token not in self.stop_words and len(token) > 2:
                    try:
                        lemma = self.lemmatizer.lemmatize(token)
                        cleaned_tokens.append(lemma)
                    except Exception as e:
                        logger.warning(f"Lemmatization failed for token '{token}': {str(e)}")
                        cleaned_tokens.append(token)
            
            return ' '.join(cleaned_tokens)
            
        except Exception as e:
            logger.warning(f"Error in text cleaning: {str(e)}")
            # Return original text if cleaning fails
            return text.lower().strip()
    
    def process_files(self, file_paths: List[str]) -> List[str]:
        """Process files with enhanced error handling."""
        try:
            # Parse documents
            parsed_docs = self.parser.parse_files(file_paths)
            
            # Extract and segment text content
            all_segments = []
            for doc in parsed_docs:
                if doc.success and doc.content:
                    segments = self._segment_text(doc.content)
                    all_segments.extend(segments)
            
            logger.info(f"Created {len(all_segments)} text segments")
            
            # Process segments
            valid_segments = []
            for segment in all_segments:
                cleaned = self._clean_text(segment)
                if cleaned and len(cleaned.split()) >= 10:
                    valid_segments.append(cleaned)
            
            # Handle insufficient segments
            if len(valid_segments) < 2:
                if self.segment_size > 2:
                    self.segment_size = max(2, self.segment_size // 2)
                    logger.info(f"Retrying with smaller segment size: {self.segment_size}")
                    return self.process_files(file_paths)
                else:
                    # If we can't create enough segments, duplicate the text with slight variations
                    if len(valid_segments) == 1:
                        text = valid_segments[0]
                        valid_segments.append(text + " [duplicate]")
            
            return valid_segments
            
        except Exception as e:
            logger.error(f"Error processing files: {str(e)}")
            if not file_paths:
                return []
            # Emergency fallback: try to process as raw text
            return [self._clean_text(open(fp, 'r').read()) for fp in file_paths if os.path.exists(fp)]

    def process_texts(self, texts: List[str], n_jobs: int = -1) -> List[str]:
        """Process multiple texts with parallel processing."""
        if not texts:
            return []
        
        try:
            with ThreadPoolExecutor(
                max_workers=n_jobs if n_jobs > 0 else None
            ) as executor:
                processed_texts = list(executor.map(self._clean_text, texts))
            
            # Filter and validate
            valid_processed = [
                text for text in processed_texts 
                if text and len(text.split()) >= 5
            ]
            
            # Handle insufficient texts
            if len(valid_processed) < 2:
                if len(valid_processed) == 1:
                    # Duplicate text with slight variation if only one valid text
                    valid_processed.append(valid_processed[0] + " [duplicate]")
                else:
                    raise ValueError("No valid texts after processing")
            
            return valid_processed
            
        except Exception as e:
            logger.error(f"Error in text processing: {str(e)}")
            # Emergency fallback: return cleaned original texts
            return [self._clean_text(text) for text in texts if text]

class TextVectorizer:
    """Text vectorization with dimensionality reduction."""
    
    def __init__(self, config: ClusteringConfig):
        """Initialize the text vectorizer."""
        self.config = config
        self.vectorizer = TfidfVectorizer(
            max_features=config.max_features,
            strip_accents='unicode',
            lowercase=True,
            norm='l2'
        )
        self.svd = TruncatedSVD(
            n_components=config.n_components,
            random_state=config.random_state
        )
        self.scaler = StandardScaler()
        logger.info("TextVectorizer initialized with provided configuration")
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to normalized, reduced-dimension vectors."""
        try:
            if not texts:
                raise ValueError("Empty text list provided for vectorization")
            
            logger.info("Performing TF-IDF vectorization...")
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            logger.info("Applying dimensionality reduction...")
            reduced_matrix = self.svd.fit_transform(tfidf_matrix)
            
            normalized_matrix = self.scaler.fit_transform(reduced_matrix)
            
            logger.info(f"Successfully vectorized {len(texts)} texts to "
                      f"{normalized_matrix.shape[1]} features")
            return normalized_matrix
            
        except Exception as e:
            logger.error(f"Error in text vectorization: {str(e)}")
            raise
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform new texts using fitted vectorizer."""
        try:
            tfidf_matrix = self.vectorizer.transform(texts)
            reduced_matrix = self.svd.transform(tfidf_matrix)
            return self.scaler.transform(reduced_matrix)
        except Exception as e:
            logger.error(f"Error transforming new texts: {str(e)}")
            raise

class ClusteringOptimizer:
    """Optimize clustering parameters and select best algorithm."""
    
    def __init__(self, config: ClusteringConfig):
        """Initialize clustering optimizer."""
        self.config = config
        self.algorithms = {
            'kmeans': {
                'model': KMeans,
                'params': {'random_state': config.random_state}
            },
            'dbscan': {
                'model': DBSCAN,
                'params': {'min_samples': config.min_cluster_size}
            },
            'hierarchical': {
                'model': AgglomerativeClustering,
                'params': {}
            }
        }
    
    def _evaluate_clustering(self, labels: np.ndarray, features: np.ndarray) -> Dict[str, float]:
        """Calculate clustering quality scores."""
        try:
            # Skip evaluation if all points are in one cluster
            unique_labels = set(labels)
            if len(unique_labels) < 2:
                return {'combined_score': -float('inf')}
            
            silhouette = silhouette_score(features, labels)
            calinski = calinski_harabasz_score(features, labels)
            
            # Combine scores with weights
            combined_score = 0.6 * silhouette + 0.4 * (calinski / 100)
            
            return {
                'silhouette': silhouette,
                'calinski_harabasz': calinski,
                'combined_score': combined_score
            }
        except Exception as e:
            logger.warning(f"Error in clustering evaluation: {str(e)}")
            return {'combined_score': -float('inf')}
    
    def find_best_clustering(self, features: np.ndarray) -> Dict[str, Any]:
        """Find best clustering algorithm and parameters."""
        best_result = {
            'algorithm': None,
            'params': None,
            'labels': None,
            'scores': {'combined_score': -float('inf')}
        }
        
        try:
            n_samples = features.shape[0]
            
            # Adjust n_clusters_range based on number of samples
            min_clusters = self.config.n_clusters_range[0]
            max_clusters = min(self.config.n_clusters_range[1], n_samples)
            
            # If we have too few samples, adjust minimum clusters
            if n_samples < min_clusters:
                min_clusters = max(2, n_samples // 2)
            
            # Ensure valid range
            if min_clusters >= max_clusters:
                max_clusters = min_clusters + 1
                
            logger.info(f"Adjusted clustering range: {min_clusters} to {max_clusters} for {n_samples} samples")
            
            for algo_name, algo_info in self.algorithms.items():
                logger.info(f"Trying {algo_name} clustering...")
                
                if algo_name in ['kmeans', 'hierarchical']:
                    # Try different numbers of clusters within adjusted range
                    for n_clusters in range(min_clusters, max_clusters):
                        params = {
                            'n_clusters': n_clusters,
                            **algo_info['params']
                        }
                        clusterer = algo_info['model'](**params)
                        labels = clusterer.fit_predict(features)
                        scores = self._evaluate_clustering(labels, features)
                        
                        if scores['combined_score'] > best_result['scores']['combined_score']:
                            best_result.update({
                                'algorithm': algo_name,
                                'params': params,
                                'labels': labels,
                                'scores': scores
                            })
                
                elif algo_name == 'dbscan':
                    # Adjust DBSCAN parameters for small datasets
                    if n_samples < self.config.min_cluster_size:
                        algo_info['params']['min_samples'] = max(2, n_samples // 3)
                    
                    # Try different eps values
                    distances = np.linalg.norm(features, axis=1)
                    eps_range = np.percentile(distances, [10, 90])
                    for eps in np.linspace(eps_range[0], eps_range[1], 10):
                        params = {'eps': eps, **algo_info['params']}
                        clusterer = algo_info['model'](**params)
                        labels = clusterer.fit_predict(features)
                        scores = self._evaluate_clustering(labels, features)
                        
                        if scores['combined_score'] > best_result['scores']['combined_score']:
                            best_result.update({
                                'algorithm': algo_name,
                                'params': params,
                                'labels': labels,
                                'scores': scores
                            })
            
            # If no good clustering found, fall back to minimum clusters
            if best_result['algorithm'] is None:
                logger.warning("No suitable clustering found, falling back to minimal clustering")
                clusterer = KMeans(n_clusters=min_clusters, random_state=self.config.random_state)
                labels = clusterer.fit_predict(features)
                best_result.update({
                    'algorithm': 'kmeans',
                    'params': {'n_clusters': min_clusters},
                    'labels': labels,
                    'scores': self._evaluate_clustering(labels, features)
                })
            
            logger.info(f"Best clustering algorithm: {best_result['algorithm']}")
            logger.info(f"Best parameters: {best_result['params']}")
            logger.info(f"Best score: {best_result['scores']['combined_score']:.3f}")
            
            return best_result
            
        except Exception as e:
            logger.error(f"Error in clustering optimization: {str(e)}")
            raise
    
class TextClusteringPipeline:
    """Main pipeline for text clustering."""
    
    def __init__(self, config: Optional[ClusteringConfig] = None, progress_callback=None):
        self.config = config or ClusteringConfig()
        self.config.validate()
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TextVectorizer(self.config)
        self.optimizer = ClusteringOptimizer(self.config)
        self.is_fitted = False
        self.best_model = None
        self.clustering_results = None
        self.progress_callback = progress_callback
        logger.info("Text clustering pipeline initialized")

    def _update_progress(self, percentage: int, message: str = ""):
        """Update progress in UI"""
        if self.progress_callback:
            self.progress_callback(percentage)
            logger.info(f"Progress: {percentage}% - {message}")

    def fit_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """Fit pipeline with enhanced document validation and sparse matrix handling"""
        try:
            self._update_progress(0, "Starting clustering pipeline...")
            
            # Validate input
            if not file_paths:
                return {
                    'success': False,
                    'error': "No input files provided",
                    'processed_files': []
                }
            
            # Process files
            self._update_progress(20, "Parsing documents...")
            try:
                processed_texts = self.preprocessor.process_files(file_paths)
            except ValueError as e:
                return {
                    'success': False,
                    'error': str(e),
                    'processed_files': []
                }
            
            # Log document statistics
            logger.info(f"""
            Document Statistics:
            - Input files: {len(file_paths)}
            - Processed documents: {len(processed_texts)}
            - Unique documents: {len(set(processed_texts))}
            """)
            
            # Vectorize texts
            self._update_progress(40, "Vectorizing texts...")
            feature_matrix = self.vectorizer.fit_transform(processed_texts)
            
            # Handle sparse matrix statistics properly
            n_nonzero = feature_matrix.nnz if sparse.issparse(feature_matrix) else np.count_nonzero(feature_matrix)
            
            logger.info(f"""
            Feature Matrix Statistics:
            - Shape: {feature_matrix.shape}
            - Non-zero elements: {n_nonzero}
            - Density: {n_nonzero / (feature_matrix.shape[0] * feature_matrix.shape[1]):.4f}
            """)
            
            # Convert to dense if necessary for clustering
            if sparse.issparse(feature_matrix):
                feature_matrix = feature_matrix.toarray()
            
            # Find optimal clustering
            self._update_progress(60, "Finding optimal clustering...")
            self.clustering_results = self.optimizer.find_best_clustering(feature_matrix)
            
            # Save the best model
            if self.clustering_results['algorithm'] == 'kmeans':
                self.best_model = KMeans(**self.clustering_results['params']).fit(feature_matrix)
            elif self.clustering_results['algorithm'] == 'dbscan':
                self.best_model = DBSCAN(**self.clustering_results['params']).fit(feature_matrix)
            else:  # hierarchical
                self.best_model = AgglomerativeClustering(**self.clustering_results['params']).fit(feature_matrix)
            
            self.is_fitted = True
            
            # Create final results
            results = {
                'success': True,
                'error': None,
                'labels': self.clustering_results['labels'].tolist(),
                'algorithm': self.clustering_results['algorithm'],
                'parameters': self.clustering_results['params'],
                'scores': self.clustering_results['scores'],
                'n_clusters': len(set(self.clustering_results['labels'][
                    self.clustering_results['labels'] >= 0
                ])),
                'n_processed_files': len(processed_texts),
                'processed_files': [os.path.basename(fp) for fp in file_paths],
                'file_paths': file_paths,
                'document_statistics': {
                    'input_files': len(file_paths),
                    'processed_documents': len(processed_texts),
                    'unique_documents': len(set(processed_texts)),
                    'feature_matrix_shape': feature_matrix.shape,
                    'feature_matrix_nonzero': n_nonzero
                },
                'cluster_distribution': {
                    f'cluster_{i}': sum(self.clustering_results['labels'] == i)
                    for i in range(max(self.clustering_results['labels']) + 1)
                }
            }
            
            # Add document clusters mapping
            results['document_clusters'] = {
                os.path.basename(fp): label
                for fp, label in zip(file_paths, self.clustering_results['labels'])
            }
            
            self._update_progress(100, "Clustering complete!")
            return results
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in clustering: {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'processed_files': []
            }

    def predict_files(self, file_paths: List[str]) -> np.ndarray:
        """Predict clusters for new files."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before prediction")
        
        try:
            # Process new files
            processed_texts = self.preprocessor.process_files(file_paths)
            
            # Handle empty results
            if not processed_texts:
                return np.array([])
            
            # Predict using existing model
            return self.predict(processed_texts)
            
        except Exception as e:
            logger.error(f"Error in file prediction: {str(e)}")
            raise

    def predict(self, texts: List[str]) -> np.ndarray:
        """Predict clusters for new texts using fitted pipeline."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before prediction")
        
        try:
            # Process texts
            processed_texts = self.preprocessor.process_texts(texts, self.config.n_jobs)
            
            # Handle empty results
            if not processed_texts:
                return np.array([])
            
            # Transform texts to feature matrix
            feature_matrix = self.vectorizer.transform(processed_texts)
            
            # Convert to dense if necessary
            if sparse.issparse(feature_matrix):
                feature_matrix = feature_matrix.toarray()
            
            # Predict using best model
            return self.best_model.predict(feature_matrix)
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise

    def save(self, path: str) -> Dict[str, Any]:
        """Enhanced save method with error handling"""
        if not self.is_fitted:
            return {
                'success': False,
                'error': "Model not fitted yet",
                'path': None
            }
        
        try:
            joblib.dump(self, path)
            return {
                'success': True,
                'error': None,
                'path': path
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'path': None
            }
    
    @classmethod
    def load(cls, path: str) -> 'TextClusteringPipeline':
        """Load fitted pipeline from disk."""
        try:
            pipeline = joblib.load(path)
            logger.info(f"Pipeline loaded successfully from {path}")
            return pipeline
        except Exception as e:
            logger.error(f"Error loading pipeline: {str(e)}")
            raise
