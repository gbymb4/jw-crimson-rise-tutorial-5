# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 20:10:01 2025

@author: Gavin
"""

import re
import os
import pickle
import numpy as np


def clean_text(text):
    """
    Clean text using regex patterns
    """
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove non-alphabetic characters (keeping spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    return text.lower()


def load_text_data(file_path, encoding='utf-8'):
    """
    Robust file loading with error handling
    """
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except UnicodeDecodeError:
        print(f"Error: Cannot decode {file_path} with {encoding}")
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
            return None
    except Exception as e:
        print(f"Unexpected error reading {file_path}: {e}")
        return None


def load_dataset_from_directory(data_dir):
    """
    Load text classification dataset from directory structure
    """
    data = []
    labels = []
    
    for label_dir in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_dir)
        if not os.path.isdir(label_path):
            continue
            
        for filename in os.listdir(label_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(label_path, filename)
                text = load_text_data(file_path)
                if text is not None:
                    data.append(text)
                    labels.append(label_dir)
    
    return data, labels


def preprocess_text(text, remove_stopwords=True, min_length=2):
    """
    Complete text preprocessing pipeline
    """
    if not isinstance(text, str):
        return ""
    
    # Basic cleaning
    text = clean_text(text)
    
    # Tokenization
    tokens = text.split()
    
    # Remove very short tokens
    tokens = [token for token in tokens if len(token) >= min_length]
    
    # Remove stopwords (optional)
    if remove_stopwords:
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                    'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 
                    'into', 'through', 'during', 'before', 'after', 'above', 
                    'below', 'between', 'among', 'this', 'that', 'these', 
                    'those', 'is', 'am', 'are', 'was', 'were', 'be', 'been', 
                    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 
                    'would', 'should', 'could', 'can', 'may', 'might', 'must'}
        tokens = [token for token in tokens if token not in stopwords]
    
    return ' '.join(tokens)


def prepare_dataset(data_dir, output_file='processed_data.pkl'):
    """
    Load, preprocess, and save dataset
    """
    print("Loading dataset...")
    texts, labels = load_dataset_from_directory(data_dir)
    
    print("Preprocessing texts...")
    processed_texts = [preprocess_text(text) for text in texts]
    
    # Convert labels to binary
    label_map = {'positive': 1, 'negative': 0}
    binary_labels = [label_map[label] for label in labels]
    
    print(f"Dataset size: {len(processed_texts)} reviews")
    print(f"Positive reviews: {sum(binary_labels)}")
    print(f"Negative reviews: {len(binary_labels) - sum(binary_labels)}")
    
    # Save processed data
    with open(output_file, 'wb') as f:
        pickle.dump({
            'texts': processed_texts,
            'labels': binary_labels,
            'original_texts': texts
        }, f)
    
    print(f"Processed data saved to {output_file}")
    return processed_texts, binary_labels


if __name__ == "__main__":
    # Prepare dataset
    data_dir = 'movie_reviews'  # Adjust path as needed
    prepare_dataset(data_dir)