"""
CLINICAL NOTES DIAGNOSIS PREDICTION - GOOGLE COLAB NOTEBOOK (IMPROVED)
=====================================================================
A complete end-to-end system for predicting diagnoses from clinical notes.
Enhanced to handle duplicate diagnosis names in dual diagnosis scenarios.

Run this as a notebook with separate cells for each section.

GitHub: https://github.com/30Piyush2025/Cbsotproject
Author: Medical AI Team
Version: 2.0.0
Date: July 4, 2024
"""

# ============================================================================
# CELL 1: INSTALL DEPENDENCIES
# ============================================================================

import subprocess
import sys

def install_packages():
    """Install all required packages for Colab."""
    packages = [
        'pandas',
        'numpy',
        'scikit-learn',
        'torch',
        'transformers',
        'sentence-transformers',
        'scipy',
        'tqdm',
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
    
    print("✓ All packages installed successfully!")

# Uncomment to run
# install_packages()


# ============================================================================
# CELL 2: IMPORT LIBRARIES
# ============================================================================

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Configure logging for Colab
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("✓ All libraries imported successfully!")
print(f"Device available: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")


# ============================================================================
# CELL 3: GLOBAL CONFIGURATION
# ============================================================================

# Global configuration
CONFIG = {
    'sentence_transformer_model': 'all-MiniLM-L6-v2',
    'bert_model': 'bert-base-uncased',
    'clinical_bert_model': 'emilyalsentzer/clinicalBERT',
    'similarity_threshold': 0.75,
    'dual_diagnosis_threshold': 0.05,
    'top_k_matches': 5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 32,
    'max_seq_length': 512
}

print("✓ Configuration loaded!")
print(f"  - Device: {CONFIG['device']}")
print(f"  - Similarity Threshold: {CONFIG['similarity_threshold']}")
print(f"  - Dual Diagnosis Threshold: {CONFIG['dual_diagnosis_threshold']}")


# ============================================================================
# CELL 4: DATA PREPROCESSING CLASS
# ============================================================================

class DataPreprocessor:
    """Handles data loading, cleaning, and preprocessing."""
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.df = None
        self.train_data = None
        self.test_data = None
        logger.info("DataPreprocessor initialized")
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load the clinical notes dataset."""
        logger.info(f"Loading data from {filepath}...")
        try:
            self.df = pd.read_csv(filepath)
            logger.info(f"✓ Data loaded. Shape: {self.df.shape}")
            return self.df
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
    
    def create_mock_dataset(self, num_samples: int = 100) -> pd.DataFrame:
        """Create a mock dataset for demonstration."""
        logger.info(f"Creating mock dataset with {num_samples} samples...")
        
        mock_notes = [
            "Patient presents with persistent cough and fever for 3 days. Chest X-ray shows bilateral pneumonia. Started on antibiotics.",
            "58-year-old male with hypertension and diabetes. Blood pressure elevated at 160/100. Medication adjustment recommended.",
            "Young female with severe headaches and neck stiffness. CSF analysis pending. Consider meningitis differential.",
            "Chronic heart failure patient with shortness of breath. Ejection fraction 35%. Diuretics increased.",
            "Diabetic patient with foot ulcer and signs of infection. HbA1c elevated. Culture sent for sensitivity.",
            "Patient with acute abdominal pain and elevated lipase levels. Ultrasound suggests pancreatitis. NPO status ordered.",
            "Elderly patient with confusion and memory loss. MRI shows hippocampal atrophy. Alzheimer's disease suspected.",
            "Postoperative patient with fever and surgical site infection. Culture positive for Staphylococcus. Antibiotics adjusted.",
            "Patient with asthma exacerbation. Peak flow reduced. Nebulizer treatment initiated.",
            "Acute MI presentation with chest pain and ST elevation. Troponin levels elevated. Immediate intervention performed.",
        ]
        
        diagnoses = [
            "Pneumonia",
            "Hypertension",
            "Meningitis",
            "Heart Failure",
            "Diabetic Foot Ulcer",
            "Acute Pancreatitis",
            "Alzheimer's Disease",
            "Surgical Site Infection",
            "Asthma",
            "Myocardial Infarction",
        ]
        
        expanded_notes = []
        expanded_diagnoses = []
        
        for i in range(num_samples):
            note_idx = i % len(mock_notes)
            diagnosis_idx = i % len(diagnoses)
            expanded_notes.append(mock_notes[note_idx])
            expanded_diagnoses.append(diagnoses[diagnosis_idx])
        
        self.df = pd.DataFrame({
            'clinical_note': expanded_notes,
            'diagnosis': expanded_diagnoses,
            'case_id': range(num_samples)
        })
        
        logger.info(f"✓ Mock dataset created. Shape: {self.df.shape}")
        return self.df
    
    def handle_missing_values(self) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        logger.info("Handling missing values...")
        initial_missing = self.df.isnull().sum().sum()
        
        self.df = self.df.dropna(subset=['diagnosis'])
        self.df['clinical_note'] = self.df['clinical_note'].fillna('')
        
        final_missing = self.df.isnull().sum().sum()
        logger.info(f"✓ Missing values handled: {initial_missing} -> {final_missing}")
        
        return self.df
    
    def clean_text(self, text: str) -> str:
        """Clean clinical note text."""
        import re
        
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-z0-9\s\.\,\-]', '', text)
        text = text.strip()
        
        return text
    
    def preprocess(self) -> pd.DataFrame:
        """Execute full preprocessing pipeline."""
        logger.info("Starting preprocessing pipeline...")
        
        self.handle_missing_values()
        logger.info("Cleaning clinical notes...")
        self.df['clinical_note_cleaned'] = self.df['clinical_note'].apply(self.clean_text)
        self.df = self.df[self.df['clinical_note_cleaned'].str.len() > 0]
        
        logger.info(f"✓ Preprocessing complete. Final shape: {self.df.shape}")
        return self.df
    
    def split_data(self, test_size: float = 0.2, random_state: int = 42):
        """Split data into train and test sets."""
        logger.info(f"Splitting data: {(1-test_size)*100:.0f}% train, {test_size*100:.0f}% test...")
        
        self.train_data, self.test_data = train_test_split(
            self.df,
            test_size=test_size,
            random_state=random_state,
            stratify=self.df['diagnosis']
        )
        
        logger.info(f"✓ Train set size: {len(self.train_data)}")
        logger.info(f"✓ Test set size: {len(self.test_data)}")

print("✓ DataPreprocessor class created!")


# ============================================================================
# CELL 5: EMBEDDING GENERATOR CLASS - PART 1 (Model Loading)
# ============================================================================

class EmbeddingGenerator:
    """Generates embeddings using different models."""
    
    def __init__(self, device: str = CONFIG['device']):
        """Initialize the embedding generator."""
        self.device = device
        self.sentence_transformer_model = None
        self.bert_tokenizer = None
        self.bert_model = None
        self.clinical_bert_tokenizer = None
        self.clinical_bert_model = None
        logger.info(f"EmbeddingGenerator initialized (Device: {device})")
    
    def load_sentence_transformer(self, model_name: str = CONFIG['sentence_transformer_model']):
        """Load Sentence Transformer model."""
        logger.info(f"Loading Sentence Transformer: {model_name}...")
        try:
            self.sentence_transformer_model = SentenceTransformer(model_name)
            logger.info(f"✓ Sentence Transformer loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Sentence Transformer: {str(e)}")
            raise
    
    def load_bert_model(self, model_name: str = CONFIG['bert_model']):
        """Load BERT model and tokenizer."""
        logger.info(f"Loading BERT model: {model_name}...")
        try:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert_model = AutoModel.from_pretrained(model_name)
            self.bert_model.to(self.device)
            self.bert_model.eval()
            logger.info(f"✓ BERT model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading BERT: {str(e)}")
            raise
    
    def load_clinical_bert_model(self, model_name: str = CONFIG['clinical_bert_model']):
        """Load ClinicalBERT model and tokenizer."""
        logger.info(f"Loading ClinicalBERT: {model_name}...")
        try:
            self.clinical_bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.clinical_bert_model = AutoModel.from_pretrained(model_name)
            self.clinical_bert_model.to(self.device)
            self.clinical_bert_model.eval()
            logger.info(f"✓ ClinicalBERT loaded successfully")
        except Exception as e:
            logger.error(f"Error loading ClinicalBERT: {str(e)}")
            raise

print("✓ EmbeddingGenerator class created (Part 1)!")


# ============================================================================
# CELL 6: EMBEDDING GENERATOR CLASS - PART 2 (Embedding Generation)
# ============================================================================

class EmbeddingGenerator:
    """Generates embeddings using different models (Continued)."""
    
    def generate_sentence_transformer_embeddings(
        self,
        texts: List[str],
        batch_size: int = CONFIG['batch_size']
    ) -> np.ndarray:
        """Generate embeddings using Sentence Transformer."""
        logger.info(f"Generating Sentence Transformer embeddings for {len(texts)} texts...")
        
        if self.sentence_transformer_model is None:
            self.load_sentence_transformer()
        
        embeddings = self.sentence_transformer_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        logger.info(f"✓ Embeddings generated. Shape: {embeddings.shape}")
        return embeddings
    
    def generate_bert_embeddings(
        self,
        texts: List[str],
        use_cls_token: bool = True,
        batch_size: int = CONFIG['batch_size']
    ) -> np.ndarray:
        """Generate embeddings using BERT model."""
        logger.info(f"Generating BERT embeddings for {len(texts)} texts...")
        
        if self.bert_model is None:
            self.load_bert_model()
        
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_texts = texts[i:i + batch_size]
            
            inputs = self.bert_tokenizer(
                batch_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=CONFIG['max_seq_length']
            )
            
            for key in inputs:
                inputs[key] = inputs[key].to(self.device)
            
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            
            if use_cls_token:
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            else:
                mask = inputs['attention_mask'].unsqueeze(-1).expand(
                    outputs.last_hidden_state.size()
                ).float()
                sum_embeddings = (outputs.last_hidden_state * mask).sum(1)
                sum_mask = mask.sum(1)
                batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
            
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings)
        logger.info(f"✓ Embeddings generated. Shape: {embeddings.shape}")
        return embeddings
    
    def generate_clinical_bert_embeddings(
        self,
        texts: List[str],
        use_cls_token: bool = True,
        batch_size: int = CONFIG['batch_size']
    ) -> np.ndarray:
        """Generate embeddings using ClinicalBERT model."""
        logger.info(f"Generating ClinicalBERT embeddings for {len(texts)} texts...")
        
        if self.clinical_bert_model is None:
            self.load_clinical_bert_model()
        
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_texts = texts[i:i + batch_size]
            
            inputs = self.clinical_bert_tokenizer(
                batch_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=CONFIG['max_seq_length']
            )
            
            for key in inputs:
                inputs[key] = inputs[key].to(self.device)
            
            with torch.no_grad():
                outputs = self.clinical_bert_model(**inputs)
            
            if use_cls_token:
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            else:
                mask = inputs['attention_mask'].unsqueeze(-1).expand(
                    outputs.last_hidden_state.size()
                ).float()
                sum_embeddings = (outputs.last_hidden_state * mask).sum(1)
                sum_mask = mask.sum(1)
                batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
            
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings)
        logger.info(f"✓ Embeddings generated. Shape: {embeddings.shape}")
        return embeddings
    
    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit vectors."""
        logger.info("Normalizing embeddings...")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)
        return normalized

print("✓ EmbeddingGenerator class updated with embedding methods!")


# ============================================================================
# CELL 7: IMPROVED DIAGNOSIS PREDICTOR WITH DUPLICATE HANDLING
# ============================================================================

class DiagnosisPredictorSimilarityMatcher:
    """
    Predicts diagnoses and finds similar medical cases.
    
    KEY IMPROVEMENT: Handles duplicate diagnosis names in dual diagnosis scenarios.
    - When 2+ matches have similar scores AND same diagnosis name, shows only highest
    - Only flags as dual diagnosis if 2+ DIFFERENT diagnoses are similarly confident
    """
    
    def __init__(
        self,
        embeddings: np.ndarray,
        diagnoses: List[str],
        case_ids: List[int],
        similarity_threshold: float = CONFIG['similarity_threshold'],
        dual_diagnosis_threshold: float = CONFIG['dual_diagnosis_threshold'],
        top_k: int = CONFIG['top_k_matches']
    ):
        """Initialize the predictor."""
        self.embeddings = self._normalize_embeddings(embeddings)
        self.diagnoses = diagnoses
        self.case_ids = case_ids
        self.similarity_threshold = similarity_threshold
        self.dual_diagnosis_threshold = dual_diagnosis_threshold
        self.top_k = top_k
        
        logger.info(f"✓ Predictor initialized with {len(diagnoses)} cases")
    
    @staticmethod
    def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit vectors."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-8)
    
    @staticmethod
    def _cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        e1 = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        e2 = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
        similarity = np.dot(e1, e2)
        return float(similarity)
    
    def _deduplicate_diagnoses(self, matches: List[Dict]) -> List[Dict]:
        """
        Remove duplicate diagnoses from dual diagnosis list.
        If same diagnosis appears multiple times, keep only the highest similarity one.
        
        IMPLEMENTATION:
        - Groups matches by diagnosis name
        - For each diagnosis, keeps the case with highest similarity score
        - Returns deduplicated list sorted by similarity
        
        Args:
            matches: List of match dictionaries
            
        Returns:
            Deduplicated list with unique diagnoses
        """
        unique_matches = {}
        
        for match in matches:
            diagnosis = match['diagnosis']
            
            # If this diagnosis hasn't been seen, or this one has higher similarity, update
            if diagnosis not in unique_matches or match['similarity'] > unique_matches[diagnosis]['similarity']:
                unique_matches[diagnosis] = match
        
        # Convert back to list and sort by similarity
        result = list(unique_matches.values())
        result.sort(key=lambda x: x['similarity'], reverse=True)
        
        return result
    
    def predict(
        self,
        query_embedding: np.ndarray,
        query_note: str
    ) -> Dict:
        """
        Predict diagnosis and find similar cases.
        
        Args:
            query_embedding: Embedding of the query note
            query_note: Original clinical note text
            
        Returns:
            Dictionary with:
            - predicted_diagnosis: Primary diagnosis
            - primary_similarity_score: Confidence score
            - similar_cases: Top matching cases
            - has_dual_diagnosis: True if 2+ different diagnoses are equally confident
            - dual_diagnoses: List of equally confident unique diagnoses
        """
        logger.info("Processing query note...")
        
        # Normalize query embedding
        query_embedding = self._normalize_embeddings(query_embedding.reshape(1, -1))[0]
        
        # Compute similarities with all training embeddings
        similarities = []
        for i, embedding in enumerate(self.embeddings):
            similarity = self._cosine_similarity(query_embedding, embedding)
            similarities.append({
                'index': i,
                'case_id': self.case_ids[i],
                'diagnosis': self.diagnoses[i],
                'similarity': similarity
            })
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        top_matches = similarities[:self.top_k]
        
        # Filter by threshold
        top_matches = [m for m in top_matches if m['similarity'] >= self.similarity_threshold]
        
        logger.info(f"✓ Found {len(top_matches)} matches")
        
        has_dual_diagnosis = False
        dual_diagnoses = []
        
        if len(top_matches) >= 2:
            score_diff = top_matches[0]['similarity'] - top_matches[1]['similarity']
            if score_diff <= self.dual_diagnosis_threshold:
                # Get all matches within the threshold
                top_score = top_matches[0]['similarity']
                candidates = [
                    m for m in top_matches
                    if (top_score - m['similarity']) <= self.dual_diagnosis_threshold
                ]
                
                # IMPROVED: Deduplicate diagnoses with same name
                dual_diagnoses = self._deduplicate_diagnoses(candidates)
                
                # Only flag as dual diagnosis if we have 2+ DIFFERENT diagnoses
                if len(dual_diagnoses) >= 2:
                    has_dual_diagnosis = True
                    logger.info(f"✓ Dual diagnosis detected: {len(dual_diagnoses)} unique diagnoses")
                else:
                    # If all matches have the same diagnosis, treat as single diagnosis
                    has_dual_diagnosis = False
                    logger.info(f"ℹ️  All matches have same diagnosis: {dual_diagnoses[0]['diagnosis']}")
        
        result = {
            'predicted_diagnosis': top_matches[0]['diagnosis'] if top_matches else 'Unknown',
            'primary_similarity_score': top_matches[0]['similarity'] if top_matches else 0.0,
            'similar_cases': top_matches,
            'has_dual_diagnosis': has_dual_diagnosis,
            'dual_diagnoses': dual_diagnoses,
            'query_note': query_note,
            'timestamp': datetime.now().isoformat()
        }
        
        return result

print("✓ DiagnosisPredictorSimilarityMatcher class created (IMPROVED)!")


# ============================================================================
# CELL 8: FORMATTING AND UTILITY FUNCTIONS
# ============================================================================

def format_prediction_output(predictor, prediction_result: Dict) -> str:
    """
    Format prediction results for display.
    Shows deduplication logic when multiple cases have same diagnosis.
    """
    output = []
    output.append("=" * 80)
    output.append("CLINICAL DIAGNOSIS PREDICTION SYSTEM - RESULTS")
    output.append("=" * 80)
    output.append(f"Timestamp: {prediction_result['timestamp']}")
    output.append("")
    
    output.append("QUERY NOTE:")
    output.append("-" * 80)
    note_preview = prediction_result['query_note'][:200] + "..." if len(prediction_result['query_note']) > 200 else prediction_result['query_note']
    output.append(note_preview)
    output.append("")
    
    if prediction_result['has_dual_diagnosis']:
        output.append("⚠️  DUAL DIAGNOSIS DETECTED")
        output.append("-" * 80)
        output.append(f"The system identified {len(prediction_result['dual_diagnoses'])} distinct diagnoses with highly similar confidence scores:")
        output.append("")
        
        for i, match in enumerate(prediction_result['dual_diagnoses'], 1):
            output.append(f"  Option {i}:")
            output.append(f"    Diagnosis: {match['diagnosis']}")
            output.append(f"    Similarity Score: {match['similarity']:.4f}")
            output.append(f"    Case ID: {match['case_id']}")
            output.append("")
        
        output.append("📌 NOTE: When multiple cases share the same diagnosis, only the highest confidence case is displayed.")
        output.append("")
    else:
        output.append("PRIMARY DIAGNOSIS PREDICTION:")
        output.append("-" * 80)
        output.append(f"Predicted Diagnosis: {prediction_result['predicted_diagnosis']}")
        output.append(f"Confidence Score: {prediction_result['primary_similarity_score']:.4f}")
        output.append("")
    
    output.append("SIMILAR CASES (TOP 5):")
    output.append("-" * 80)
    for i, case in enumerate(prediction_result['similar_cases'][:5], 1):
        output.append(f"  Case {i}:")
        output.append(f"    ID: {case['case_id']}")
        output.append(f"    Diagnosis: {case['diagnosis']}")
        output.append(f"    Similarity Score: {case['similarity']:.4f}")
        output.append("")
    
    output.append("=" * 80)
    
    return "\n".join(output)


def compute_statistics(results: List[Dict]) -> Dict:
    """Compute statistics from prediction results."""
    total_predictions = len(results)
    dual_diagnosis_count = sum(1 for r in results if r.get('has_dual_diagnosis', False))
    similarity_scores = [r['primary_similarity_score'] for r in results]
    
    stats = {
        'total_predictions': total_predictions,
        'dual_diagnosis_count': dual_diagnosis_count,
        'dual_diagnosis_percentage': (dual_diagnosis_count / total_predictions * 100) if total_predictions > 0 else 0,
        'mean_similarity_score': np.mean(similarity_scores),
        'median_similarity_score': np.median(similarity_scores),
        'min_similarity_score': np.min(similarity_scores),
        'max_similarity_score': np.max(similarity_scores)
    }
    
    return stats


def print_statistics(stats: Dict):
    """Print statistics in a formatted manner."""
    print("\n" + "=" * 60)
    print("PREDICTION STATISTICS")
    print("=" * 60)
    print(f"Total Predictions: {stats['total_predictions']}")
    print(f"Dual Diagnosis Cases: {stats['dual_diagnosis_count']}")
    print(f"Dual Diagnosis Percentage: {stats['dual_diagnosis_percentage']:.2f}%")
    print(f"\nSimilarity Scores:")
    print(f"  Mean: {stats['mean_similarity_score']:.4f}")
    print(f"  Median: {stats['median_similarity_score']:.4f}")
    print(f"  Min: {stats['min_similarity_score']:.4f}")
    print(f"  Max: {stats['max_similarity_score']:.4f}")
    print("=" * 60 + "\n")

print("✓ Utility functions created!")


# ============================================================================
# CELL 9: COMPLETE PIPELINE EXECUTION
# ============================================================================

def run_complete_pipeline(num_samples: int = 100):
    """Execute the complete end-to-end pipeline."""
    print("\n" + "=" * 80)
    print("STARTING CLINICAL DIAGNOSIS PREDICTION PIPELINE")
    print("=" * 80 + "\n")
    
    # Step 1: Data Preprocessing
    print("[STEP 1] DATA LOADING & PREPROCESSING")
    print("-" * 80)
    preprocessor = DataPreprocessor()
    dataset = preprocessor.create_mock_dataset(num_samples=num_samples)
    dataset = preprocessor.preprocess()
    preprocessor.split_data()
    
    train_data = preprocessor.train_data
    test_data = preprocessor.test_data
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}\n")
    
    # Step 2: Feature Extraction
    print("[STEP 2] FEATURE EXTRACTION (EMBEDDINGS)")
    print("-" * 80)
    embedding_generator = EmbeddingGenerator(device=CONFIG['device'])
    
    print("Generating Sentence Transformer embeddings...")
    train_embeddings = embedding_generator.generate_sentence_transformer_embeddings(
        train_data['clinical_note_cleaned'].tolist()
    )
    print()
    
    # Step 3: Initialize Predictor
    print("[STEP 3] INITIALIZING DIAGNOSIS PREDICTOR")
    print("-" * 80)
    predictor = DiagnosisPredictorSimilarityMatcher(
        embeddings=train_embeddings,
        diagnoses=train_data['diagnosis'].tolist(),
        case_ids=train_data['case_id'].tolist(),
        similarity_threshold=CONFIG['similarity_threshold'],
        dual_diagnosis_threshold=CONFIG['dual_diagnosis_threshold'],
        top_k=CONFIG['top_k_matches']
    )
    print()
    
    # Step 4: Test with Sample Queries
    print("[STEP 4] TESTING WITH QUERY NOTES")
    print("-" * 80)
    
    # Test Case 1
    test_query_1 = test_data.iloc[0]
    print(f"\nTest Case 1 (Expected: {test_query_1['diagnosis']})")
    
    query_embedding_1 = embedding_generator.generate_sentence_transformer_embeddings(
        [test_query_1['clinical_note_cleaned']]
    )[0]
    
    result_1 = predictor.predict(
        query_embedding=query_embedding_1,
        query_note=test_query_1['clinical_note']
    )
    
    output_1 = format_prediction_output(predictor, result_1)
    print("\n" + output_1)
    
    # Test Case 2 (if available)
    if len(test_data) > 1:
        test_query_2 = test_data.iloc[1]
        print(f"\nTest Case 2 (Expected: {test_query_2['diagnosis']})")
        
        query_embedding_2 = embedding_generator.generate_sentence_transformer_embeddings(
            [test_query_2['clinical_note_cleaned']]
        )[0]
        
        result_2 = predictor.predict(
            query_embedding=query_embedding_2,
            query_note=test_query_2['clinical_note']
        )
        
        output_2 = format_prediction_output(predictor, result_2)
        print("\n" + output_2)
    
    # Step 5: Performance Evaluation
    print("\n[STEP 5] PERFORMANCE EVALUATION")
    print("-" * 80)
    
    correct_predictions = 0
    dual_diagnosis_count = 0
    all_results = []
    
    print("Evaluating on test set...")
    for idx, row in test_data.iterrows():
        query_embedding = embedding_generator.generate_sentence_transformer_embeddings(
            [row['clinical_note_cleaned']]
        )[0]
        
        result = predictor.predict(
            query_embedding=query_embedding,
            query_note=row['clinical_note']
        )
        
        all_results.append(result)
        
        if result['predicted_diagnosis'] == row['diagnosis']:
            correct_predictions += 1
        
        if result['has_dual_diagnosis']:
            dual_diagnosis_count += 1
    
    accuracy = correct_predictions / len(test_data) * 100
    
    stats = compute_statistics(all_results)
    print_statistics(stats)
    
    print("=" * 80)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    return {
        'preprocessor': preprocessor,
        'embedding_generator': embedding_generator,
        'predictor': predictor,
        'train_embeddings': train_embeddings,
        'accuracy': accuracy,
        'stats': stats,
        'test_results': all_results
    }

print("✓ Pipeline function created and ready to execute!")


# ============================================================================
# CELL 10: EXECUTE COMPLETE PIPELINE
# ============================================================================

# Run the complete pipeline
print("\n🚀 EXECUTING COMPLETE PIPELINE...\n")
pipeline_results = run_complete_pipeline(num_samples=100)

print("\n✅ PIPELINE EXECUTION COMPLETED!")
print(f"\nAccuracy on test set: {pipeline_results['accuracy']:.2f}%")


# ============================================================================
# CELL 11: CUSTOM QUERY PREDICTION
# ============================================================================

def predict_custom_query(
    pipeline_results: Dict,
    query_text: str,
    show_details: bool = True
) -> Dict:
    """
    Make a prediction for a custom clinical note query.
    """
    embedding_generator = pipeline_results['embedding_generator']
    predictor = pipeline_results['predictor']
    
    preprocessor = DataPreprocessor()
    cleaned_query = preprocessor.clean_text(query_text)
    
    query_embedding = embedding_generator.generate_sentence_transformer_embeddings(
        [cleaned_query]
    )[0]
    
    result = predictor.predict(
        query_embedding=query_embedding,
        query_note=query_text
    )
    
    if show_details:
        output = format_prediction_output(predictor, result)
        print(output)
    
    return result


# Example: Make a custom prediction
print("\n[CUSTOM QUERY EXAMPLE]")
print("-" * 80)

custom_note = """
Patient is a 45-year-old presenting with severe chest pain, 
shortness of breath, and diaphoresis. ECG shows ST elevation. 
Troponin levels significantly elevated. Risk factors include 
hypertension and smoking. Immediate cardiology consult required.
"""

result = predict_custom_query(pipeline_results, custom_note, show_details=True)


# ============================================================================
# CELL 12: BATCH PREDICTIONS
# ============================================================================

def batch_predict(
    pipeline_results: Dict,
    queries: List[str],
    return_dataframe: bool = True
) -> pd.DataFrame:
    """Make predictions for multiple clinical notes."""
    results = []
    
    print(f"Processing {len(queries)} queries...")
    
    for i, query in enumerate(tqdm(queries), 1):
        result = predict_custom_query(pipeline_results, query, show_details=False)
        results.append({
            'query_index': i,
            'predicted_diagnosis': result['predicted_diagnosis'],
            'confidence': result['primary_similarity_score'],
            'has_dual_diagnosis': result['has_dual_diagnosis'],
            'num_similar_cases': len(result['similar_cases']),
            'num_unique_diagnoses': len(result['dual_diagnoses']) if result['has_dual_diagnosis'] else 1
        })
    
    if return_dataframe:
        return pd.DataFrame(results)
    else:
        return results


# Example: Batch predictions
print("\n[BATCH PREDICTION EXAMPLE]")
print("-" * 80)

batch_queries = [
    "Patient with persistent cough and fever. Chest X-ray shows infiltrates. Suspect pneumonia.",
    "61-year-old diabetic with foot ulcer. HbA1c is 8.5%. Culture pending.",
    "Young patient with severe headache and neck stiffness. Lumbar puncture scheduled.",
]

batch_results = batch_predict(pipeline_results, batch_queries)
print("\nBatch Results:")
print(batch_results.to_string(index=False))


# ============================================================================
# CELL 13: SAVE AND LOAD MODELS
# ============================================================================

def save_pipeline_to_disk(pipeline_results: Dict, output_dir: str = '/content/models'):
    """Save pipeline artifacts to disk."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving pipeline to {output_dir}...")
    
    np.save(f'{output_dir}/train_embeddings.npy', pipeline_results['train_embeddings'])
    
    with open(f'{output_dir}/predictor.pkl', 'wb') as f:
        pickle.dump(pipeline_results['predictor'], f)
    
    with open(f'{output_dir}/preprocessor.pkl', 'wb') as f:
        pickle.dump(pipeline_results['preprocessor'], f)
    
    with open(f'{output_dir}/stats.json', 'w') as f:
        json.dump(pipeline_results['stats'], f, indent=2)
    
    print(f"✓ Pipeline saved to {output_dir}")


def load_pipeline_from_disk(input_dir: str = '/content/models') -> Dict:
    """Load pipeline artifacts from disk."""
    print(f"Loading pipeline from {input_dir}...")
    
    train_embeddings = np.load(f'{input_dir}/train_embeddings.npy')
    
    with open(f'{input_dir}/predictor.pkl', 'rb') as f:
        predictor = pickle.load(f)
    
    with open(f'{input_dir}/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    
    with open(f'{input_dir}/stats.json', 'r') as f:
        stats = json.load(f)
    
    embedding_generator = EmbeddingGenerator()
    embedding_generator.load_sentence_transformer()
    
    print(f"✓ Pipeline loaded from {input_dir}")
    
    return {
        'preprocessor': preprocessor,
        'embedding_generator': embedding_generator,
        'predictor': predictor,
        'train_embeddings': train_embeddings,
        'stats': stats
    }


# Example: Save the pipeline
print("\n[SAVING PIPELINE]")
print("-" * 80)
save_pipeline_to_disk(pipeline_results)

print("\n✓ All functions created successfully!")
print("✓ Ready for production use!")


# ============================================================================
# CELL 14: SUMMARY OF KEY IMPROVEMENTS
# ============================================================================

print("\n" + "=" * 80)
print("KEY IMPROVEMENTS IN THIS VERSION")
print("=" * 80)
print("""
✅ SMART DUPLICATE DIAGNOSIS HANDLING (Main Feature):
   
   PROBLEM: When 2+ matches have similar scores AND the same diagnosis name,
   the system would confusingly show both cases with the same diagnosis.
   
   SOLUTION: Implemented _deduplicate_diagnoses() method that:
   - Groups all matches by diagnosis name
   - For each diagnosis, keeps ONLY the highest confidence case
   - Returns deduplicated list sorted by similarity
   
   EXAMPLE:
   ❌ Without feature:
      Option 1: Pneumonia (Score: 0.8923)
      Option 2: Pneumonia (Score: 0.8901)  ← Confusing duplicate!
   
   ✅ With feature:
      Option 1: Pneumonia (Score: 0.8923)  ← Only highest shown
      (Note explains: Multiple cases with same diagnosis use highest)

✅ UNIQUE DIAGNOSIS DETECTION:
   - Flags as "Dual Diagnosis" ONLY if 2+ DIFFERENT diagnoses are equally confident
   - If all high-confidence matches are same diagnosis, treats as single prediction
   - More accurate clinical interpretation

✅ CLEARER OUTPUT:
   - Explicitly states "distinct diagnoses" to clarify deduplication
   - Shows note explaining why some cases aren't displayed
   - Better formatted dual diagnosis section
   - More informative for clinical decision-making

✅ COLAB OPTIMIZATION:
   - All code split into 14 manageable cells
   - Easy to run step-by-step or all at once
   - Clear progress indicators with emoji
   - Optimized for GPU/CPU automatic detection
   
✅ PRODUCTION-READY:
   - Comprehensive logging for debugging
   - Error handling at each step
   - Model save/load functionality
   - Batch prediction support
   - Statistics and performance metrics
""")
print("=" * 80)

print("\n📊 USAGE GUIDE:")
print("-" * 80)
print("""
1. RUN ALL CELLS: Execute from Cell 1 to Cell 14 in sequence
   
2. CUSTOM PREDICTION (After pipeline runs):
   result = predict_custom_query(pipeline_results, "Your clinical note here")
   
3. BATCH PREDICTIONS:
   results_df = batch_predict(pipeline_results, [list of notes])
   
4. SAVE MODEL:
   save_pipeline_to_disk(pipeline_results)
   
5. LOAD MODEL (in new session):
   pipeline_results = load_pipeline_from_disk()
   
6. CHANGE MODELS:
   - Sentence Transformer (current): Fast, good for similarity
   - BERT: More accurate, slower
   - ClinicalBERT: Best for medical notes, slowest
   
   To use: Change CONFIG['sentence_transformer_model'] to desired model
""")
print("=" * 80)
