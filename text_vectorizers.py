# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 20:10:26 2025

@author: Gavin
"""

import numpy as np


class BagOfWordsVectorizer:
    def __init__(self, max_features=5000):
        self.max_features = max_features
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocabulary_size = 0
    
    def fit(self, texts):
        """Build vocabulary from training texts"""
        word_counts = {}
        
        for text in texts:
            for word in text.split():
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency and take top max_features
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        vocab_words = [word for word, count in sorted_words[:self.max_features]]
        
        # Build mapping
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab_words)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocabulary_size = len(self.word_to_idx)
        
        return self
    
    def transform(self, texts):
        """Convert texts to BoW vectors"""
        vectors = []
        
        for text in texts:
            vector = [0] * self.vocabulary_size
            for word in text.split():
                if word in self.word_to_idx:
                    vector[self.word_to_idx[word]] += 1
            vectors.append(vector)
        
        return np.array(vectors)
    
    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)


class TFIDFVectorizer:
    def __init__(self, max_features=5000):
        self.max_features = max_features
        self.word_to_idx = {}
        self.idf_values = {}
        self.vocabulary_size = 0
    
    def fit(self, texts):
        """Build vocabulary and compute IDF values"""
        # Build vocabulary (same as BoW)
        word_counts = {}
        doc_frequencies = {}
        
        for text in texts:
            words_in_doc = set(text.split())
            for word in text.split():
                word_counts[word] = word_counts.get(word, 0) + 1
            for word in words_in_doc:
                doc_frequencies[word] = doc_frequencies.get(word, 0) + 1
        
        # Select top words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        vocab_words = [word for word, count in sorted_words[:self.max_features]]
        
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab_words)}
        self.vocabulary_size = len(self.word_to_idx)
        
        # Compute IDF values
        total_docs = len(texts)
        for word in vocab_words:
            self.idf_values[word] = np.log(total_docs / doc_frequencies[word])
        
        return self
    
    def transform(self, texts):
        """Convert texts to TF-IDF vectors"""
        vectors = []
        
        for text in texts:
            words = text.split()
            total_words = len(words)
            
            # Compute TF
            tf_dict = {}
            for word in words:
                tf_dict[word] = tf_dict.get(word, 0) + 1
            
            # Normalize TF
            for word in tf_dict:
                tf_dict[word] = tf_dict[word] / total_words
            
            # Compute TF-IDF vector
            vector = [0.0] * self.vocabulary_size
            for word, tf in tf_dict.items():
                if word in self.word_to_idx:
                    idx = self.word_to_idx[word]
                    vector[idx] = tf * self.idf_values[word]
            
            vectors.append(vector)
        
        return np.array(vectors)
    
    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)
