from typing import List, Dict, Any, Optional, Union
import numpy as np
from dataclasses import dataclass
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import logging
import joblib
import subprocess
from document_parser import DocumentParser, ParsedDocument
#nltk.download('all')
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
#subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
@dataclass
class NLPConfig:
    """Enhanced configuration for NLP pipeline."""
    model_name: str = 'en_core_web_sm'    # spaCy model
    max_features: int = 5000              # Max features for TF-IDF
    min_word_length: int = 3              # Minimum word length to keep
    parallel_processing: bool = True       # Enable parallel processing
    num_workers: int = None               # Number of workers for parallel processing

class TextPreprocessor:
    """Enhanced text preprocessing component."""
    
    def __init__(self):
        """Initialize text preprocessor."""
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            logger.info("Text preprocessor initialized")
        except Exception as e:
            logger.error(f"Error initializing preprocessor: {str(e)}")
            raise
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        try:
            if not text or not isinstance(text, str):
                return ""
            
            # Convert to lowercase and remove special characters
            text = re.sub(r'[^\w\s]', '', text.lower())
            
            # Tokenize and remove short words
            tokens = word_tokenize(text)
            tokens = [
                self.lemmatizer.lemmatize(token)
                for token in tokens
                if token not in self.stop_words and len(token) >= 3
            ]
            
            return ' '.join(tokens)
        except Exception as e:
            logger.warning(f"Error cleaning text: {str(e)}")
            return ""
    
    def process_texts(self, texts: List[str]) -> List[str]:
        """Process multiple texts."""
        return [self.clean_text(text) for text in texts]

class TextAnalyzer:
    """Enhanced text analysis component."""
    
    def __init__(self, config: NLPConfig):
        """Initialize text analyzer."""
        self.config = config
        self.nlp = spacy.load(config.model_name)
        self.vectorizer = TfidfVectorizer(
            max_features=config.max_features,
            stop_words='english'
        )
    
    def extract_features(self, texts: List[str]) -> Dict[str, Any]:
        """Extract basic features from texts."""
        try:
            # Get TF-IDF features
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Process with spaCy for entities and key phrases
            features = []
            for doc in self.nlp.pipe(texts):
                doc_features = {
                    'entities': [(ent.text, ent.label_) for ent in doc.ents],
                    'noun_phrases': [chunk.text for chunk in doc.noun_chunks]
                }
                features.append(doc_features)
            
            return {
                'tfidf_matrix': tfidf_matrix,
                'feature_names': self.vectorizer.get_feature_names_out(),
                'doc_features': features
            }
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            raise

class NLPPipeline:
    """Enhanced NLP pipeline with document parsing capabilities."""
    
    def __init__(self, config: Optional[NLPConfig] = None):
        """Initialize NLP pipeline."""
        self.config = config or NLPConfig()
        self.preprocessor = TextPreprocessor()
        self.analyzer = TextAnalyzer(self.config)
        self.parser = DocumentParser(
            num_workers=self.config.num_workers,
            use_process_pool=self.config.parallel_processing
        )
        logger.info("NLP pipeline initialized")
    
    def process_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """Process documents from files through the pipeline."""
        try:
            # Parse documents
            logger.info("Parsing documents...")
            parsed_docs = self.parser.parse_files(file_paths)
            
            # Extract texts from successful parses
            texts = []
            metadata = []
            failed_files = []
            
            for doc in parsed_docs:
                if doc.success:
                    texts.append(doc.content)
                    metadata.append({
                        'file_path': doc.file_path,
                        'metadata': doc.metadata
                    })
                else:
                    failed_files.append({
                        'file_path': doc.file_path,
                        'error': doc.error_message
                    })
            
            # Process texts through NLP pipeline
            if texts:
                nlp_results = self.process(texts)
                
                # Combine results
                results = {
                    **nlp_results,
                    'document_metadata': metadata,
                    'failed_files': failed_files
                }
            else:
                results = {
                    'error': 'No valid documents to process',
                    'failed_files': failed_files
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in file processing: {str(e)}")
            raise
    
    def process(self, texts: List[str]) -> Dict[str, Any]:
        """Process texts through the pipeline."""
        try:
            # Preprocess texts
            logger.info("Preprocessing texts...")
            processed_texts = self.preprocessor.process_texts(texts)
            
            # Extract features
            logger.info("Extracting features...")
            features = self.analyzer.extract_features(processed_texts)
            
            results = {
                'processed_texts': processed_texts,
                'tfidf_matrix': features['tfidf_matrix'],
                'feature_names': features['feature_names'],
                'document_features': features['doc_features']
            }
            
            logger.info("Text processing complete")
            return results
        
        except Exception as e:
            logger.error(f"Error in pipeline processing: {str(e)}")
            raise
    
    def save(self, path: str):
        """Save pipeline to disk."""
        try:
            joblib.dump(self, path)
            logger.info(f"Pipeline saved to {path}")
        except Exception as e:
            logger.error(f"Error saving pipeline: {str(e)}")
            raise
    
    @classmethod
    def load(cls, path: str) -> 'NLPPipeline':
        """Load pipeline from disk."""
        try:
            pipeline = joblib.load(path)
            logger.info(f"Pipeline loaded from {path}")
            return pipeline
        except Exception as e:
            logger.error(f"Error loading pipeline: {str(e)}")
            raise
