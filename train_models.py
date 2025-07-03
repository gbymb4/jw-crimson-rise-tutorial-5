# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 20:11:02 2025

@author: Gavin
"""

import torch
import torch.nn as nn
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from text_vectorizers import BagOfWordsVectorizer, TFIDFVectorizer, SequentialIndexer
from text_models import TextMLP, TextCNN


def train_mlp_classifier(train_texts, train_labels, val_texts, val_labels, 
                        vectorizer_type='bow', epochs=50):
    """
    Train MLP classifier with specified vectorizer
    """
    # Vectorize text
    if vectorizer_type == 'bow':
        vectorizer = BagOfWordsVectorizer(max_features=5000)
    elif vectorizer_type == 'tfidf':
        vectorizer = TFIDFVectorizer(max_features=5000)
    
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    y_train = torch.LongTensor(train_labels)
    y_val = torch.LongTensor(val_labels)
    
    # Initialize model
    model = TextMLP(vocab_size=vectorizer.vocabulary_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_predictions = torch.argmax(val_outputs, dim=1)
            val_accuracy = (val_predictions == y_val).float().mean()
        
        train_losses.append(loss.item())
        val_accuracies.append(val_accuracy.item())
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}, Val Acc: {val_accuracy.item():.4f}')
    
    return model, vectorizer, train_losses, val_accuracies


def train_cnn_classifier(train_texts, train_labels, val_texts, val_labels,  
                         vectorizer_type='bow', epochs=50):  
    """  
    Train CNN classifier with Bag of Words or TF-IDF representation.  
      
    Args:  
        train_texts (list): Training texts.  
        train_labels (list): Training labels.  
        val_texts (list): Validation texts.  
        val_labels (list): Validation labels.  
        vectorizer_type (str): 'bow' for Bag of Words, 'tfidf' for TF-IDF.  
        epochs (int): Number of training epochs.  
      
    Returns:  
        model (TextCNN): Trained CNN model.  
        vectorizer (Vectorizer): Vectorizer used for preprocessing.  
        train_losses (list): List of training losses.  
        val_accuracies (list): List of validation accuracies.  
    """  
    # Select vectorizer  
    if vectorizer_type == 'bow':  
        from text_vectorizers import BagOfWordsVectorizer  
        vectorizer = BagOfWordsVectorizer(max_features=5000)  
    elif vectorizer_type == 'tfidf':  
        from text_vectorizers import TFIDFVectorizer  
        vectorizer = TFIDFVectorizer(max_features=5000)  
    else:  
        raise ValueError("Invalid vectorizer_type. Choose 'bow' or 'tfidf'.")  
  
    # Vectorize the data  
    X_train = vectorizer.fit_transform(train_texts)  # Shape: (num_samples, vocab_size)  
    X_val = vectorizer.transform(val_texts)  # Shape: (num_samples, vocab_size)  
  
    # Convert to PyTorch tensors  
    X_train = torch.FloatTensor(X_train)  # Remove `.toarray()` since the data is already dense  
    X_val = torch.FloatTensor(X_val)  # Remove `.toarray()`  
    y_train = torch.LongTensor(train_labels)  
    y_val = torch.LongTensor(val_labels)  
  
    # Initialize the CNN model  
    model = TextCNN(input_dim=X_train.shape[1])  # Input dimension = vocab_size  
    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  
  
    # Training loop  
    train_losses = []  
    val_accuracies = []  
  
    for epoch in range(epochs):  
        # Training phase  
        model.train()  
        optimizer.zero_grad()  
        outputs = model(X_train)  # Forward pass  
        loss = criterion(outputs, y_train)  # Compute loss  
        loss.backward()  # Backpropagation  
        optimizer.step()  # Update weights  
  
        # Validation phase  
        model.eval()  
        with torch.no_grad():  
            val_outputs = model(X_val)  # Forward pass  
            val_predictions = torch.argmax(val_outputs, dim=1)  # Get predicted labels  
            val_accuracy = (val_predictions == y_val).float().mean()  # Compute accuracy  
  
        # Record metrics  
        train_losses.append(loss.item())  
        val_accuracies.append(val_accuracy.item())  
  
        # Print progress every 10 epochs  
        if epoch % 10 == 0:  
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}, Val Acc: {val_accuracy.item():.4f}')  
  
    return model, vectorizer, train_losses, val_accuracies  
 


def comprehensive_evaluation(train_texts, train_labels, test_texts, test_labels):  
    """  
    Compare all methods (MLP and CNN) with BoW and TF-IDF representations.  
    """  
    results = {}  
  
    # 1. MLP with Bag of Words  
    print("\n=== Training MLP with Bag of Words ===")  
    mlp_bow, bow_vectorizer, bow_losses, bow_acc = train_mlp_classifier(  
        train_texts, train_labels, test_texts, test_labels,  
        vectorizer_type='bow'  
    )  
    results['MLP_BoW'] = {  
        'model': mlp_bow,  
        'vectorizer': bow_vectorizer,  
        'final_accuracy': bow_acc[-1],  
        'training_curve': bow_acc  
    }  
  
    # 2. MLP with TF-IDF  
    print("\n=== Training MLP with TF-IDF ===")  
    mlp_tfidf, tfidf_vectorizer, tfidf_losses, tfidf_acc = train_mlp_classifier(  
        train_texts, train_labels, test_texts, test_labels,  
        vectorizer_type='tfidf'  
    )  
    results['MLP_TFIDF'] = {  
        'model': mlp_tfidf,  
        'vectorizer': tfidf_vectorizer,  
        'final_accuracy': tfidf_acc[-1],  
        'training_curve': tfidf_acc  
    }  
  
    # 3. CNN with Bag of Words  
    print("\n=== Training CNN with Bag of Words ===")  
    cnn_bow, bow_vectorizer, cnn_bow_losses, cnn_bow_acc = train_cnn_classifier(  
        train_texts, train_labels, test_texts, test_labels,  
        vectorizer_type='bow'  
    )  
    results['CNN_BoW'] = {  
        'model': cnn_bow,  
        'vectorizer': bow_vectorizer,  
        'final_accuracy': cnn_bow_acc[-1],  
        'training_curve': cnn_bow_acc  
    }  
  
    # 4. CNN with TF-IDF  
    print("\n=== Training CNN with TF-IDF ===")  
    cnn_tfidf, tfidf_vectorizer, cnn_tfidf_losses, cnn_tfidf_acc = train_cnn_classifier(  
        train_texts, train_labels, test_texts, test_labels,  
        vectorizer_type='tfidf'  
    )  
    results['CNN_TFIDF'] = {  
        'model': cnn_tfidf,  
        'vectorizer': tfidf_vectorizer,  
        'final_accuracy': cnn_tfidf_acc[-1],  
        'training_curve': cnn_tfidf_acc  
    }  
  
    return results  


def main():
    # Load processed data
    with open('processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    texts = data['texts']
    labels = data['labels']
    
    # Split data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training set size: {len(train_texts)}")
    print(f"Test set size: {len(test_texts)}")
    
    # Run comprehensive evaluation
    results = comprehensive_evaluation(train_texts, train_labels, test_texts, test_labels)
    
    # Print final results
    print("\n=== Final Results ===")
    for method, result in results.items():
        print(f"{method}: {result['final_accuracy']:.4f}")
    
    # Save results
    with open('model_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\nResults saved to model_results.pkl")


if __name__ == "__main__":
    main()