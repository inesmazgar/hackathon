import pandas as pd
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from langdetect import detect
from nltk.corpus import wordnet
import re
import torch
print(torch.__version__)
import numpy as np
!pip install transformers
import nltk
!pip install sentence-transformers
from transformers import AutoTokenizer, AutoModel                                                                                
from sentence_transformers import SentenceTransformer
# Télécharger les ressources NLTK
nltk.download('punkt')
nltk.download('stopwords')

class AdvancedIHECChatbot:
    def __init__(self, json_path='dataset.json'):
        # Download NLTK resources
        nltk.download('wordnet', quiet=True)
        
        # Advanced multilingual model selection
        self.model_name = 'xlm-roberta-base'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.embedding_model = AutoModel.from_pretrained(self.model_name)
        
        # Load data
        try:
            with open(json_path, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
            self.data = pd.DataFrame(json_data['dataset'])
        except FileNotFoundError:
            self.data = pd.DataFrame(columns=['question', 'answer'])
        
        # Advanced synonym expansion and embedding
        self.enhanced_synonyms_embeddings()
    
    def advanced_synonym_expansion(self, text):
        """
        Comprehensive synonym and semantic expansion
        """
        def get_wordnet_synonyms(word):
            synonyms = set()
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name().replace('_', ' '))
            return synonyms
        
        # Preprocessing
        text = text.lower()
        words = re.findall(r'\w+', text)
        
        # Expansion strategies
        expanded_tokens = []
        for word in words:
            # Original word
            expanded_tokens.append(word)
            
            # Wordnet synonyms
            synonyms = get_wordnet_synonyms(word)
            expanded_tokens.extend(synonyms)
            
            # Morphological variations
            expanded_tokens.append(word + 's')  # Pluriel
            expanded_tokens.append(word + 'ing')  # Forme continue
            expanded_tokens.append(word + 'ed')  # Passé
        
        # Additional contextual expansions
     context_mappings = {
    'inscription': ['registering', 'enrollment', 'admission'],
    'frais': ['costs', 'fees', 'expenses'],
    'ihec': ['institute', 'school', 'university'],
    # Add English variations
    'registration': ['inscription', 'enrollment'],
    'fees': ['frais', 'expenses'],
    'school': ['école', 'ihec']
}
        
        for base, expansions in context_mappings.items():
            if base in text:
                expanded_tokens.extend(expansions)
        
        return ' '.join(set(expanded_tokens))
    
    def enhanced_synonyms_embeddings(self):
        """
        Generate embeddings with advanced synonym expansion
        """
        # Apply advanced synonym expansion
        self.data['expanded_questions'] = self.data['question'].apply(self.advanced_synonym_expansion)
        
        # Generate embeddings
        self.question_embeddings = self.generate_embeddings(self.data['expanded_questions'])
    
    def generate_embeddings(self, texts):
        """
        Generate contextual embeddings using XLM-RoBERTa
        """
        embeddings = []
        for text in texts:
            # Tokenize
            inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
        
        return np.array(embeddings)
    
    def find_best_match(self, query, top_k=3):
        """
        Advanced semantic matching with multiple strategies
        """

 # Detect language
    try:
        detected_lang = detect(query)
    except:
        detected_lang = 'fr'  # Default to French if detection fails
    
    # Optional: Translate query if not in French
    if detected_lang != 'fr':
        blob = TextBlob(query)
        try:
            # Translate to French if not already French
            query = str(blob.translate(to='fr'))
        except:
            pass

        # Expand query with synonyms
        expanded_query = self.advanced_synonym_expansion(query)
        
        # Generate query embedding
        query_embedding = self.generate_embeddings([expanded_query])[0]
        
        # Compute cosine similarity
        similarities = np.dot(self.question_embeddings, query_embedding) / (
            np.linalg.norm(self.question_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Top-k matching
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = [
            {
                "question": self.data.iloc[idx]['question'],
                "answer": self.data.iloc[idx]['answer'],
                "score": similarities[idx]
            } 
            for idx in top_indices
        ]
        
        return results
