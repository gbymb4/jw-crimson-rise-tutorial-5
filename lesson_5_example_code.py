# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 20:13:33 2025

@author: Gavin
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# Import our modules
from text_preprocessing import prepare_dataset, preprocess_text
from text_vectorizers import BagOfWordsVectorizer, TFIDFVectorizer, SequentialIndexer
from train_models import comprehensive_evaluation


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def analyze_semantic_similarity_failures(model, vectorizer, test_cases):
    """
    Test how models handle semantically similar but lexically different texts
    """
    print_section("Semantic Similarity Analysis")
    
    test_cases = [
        ("The movie was fantastic", "The film was excellent"),
        ("I love this product", "I adore this item"),
        ("The service was terrible", "The service was awful"),
        ("Very good quality", "Outstanding quality")
    ]
    
    for text1, text2 in test_cases:
        if hasattr(vectorizer, 'transform'):
            # For BoW/TF-IDF
            processed1 = preprocess_text(text1)
            processed2 = preprocess_text(text2)
            
            vec1 = vectorizer.transform([processed1])
            vec2 = vectorizer.transform([processed2])
            
            # Cosine similarity
            similarity = cosine_similarity(vec1, vec2)[0][0]
            
            print(f"Text 1: {text1}")
            print(f"Text 2: {text2}")
            print(f"Vector similarity: {similarity:.4f}")
            print("---")


def analyze_word_order_sensitivity(vectorizer, test_cases):
    """
    Test how models handle word order changes
    """
    print_section("Word Order Sensitivity Analysis")
    
    test_cases = [
        ("The dog chased the cat", "The cat chased the dog"),
        ("I really don't like this movie", "I don't really like this movie"),
        ("The restaurant has good food", "Good food has the restaurant")
    ]
    
    for original, reordered in test_cases:
        processed_orig = preprocess_text(original)
        processed_reord = preprocess_text(reordered)
        
        if isinstance(vectorizer, BagOfWordsVectorizer):
            # BoW should be identical
            vec1 = vectorizer.transform([processed_orig])
            vec2 = vectorizer.transform([processed_reord])
            identical = np.array_equal(vec1, vec2)
            print(f"Original: {original}")
            print(f"Reordered: {reordered}")
            print(f"BoW vectors identical: {identical}")
            print("---")
        
        elif isinstance(vectorizer, SequentialIndexer):
            # Sequential should be different
            seq1 = vectorizer.transform([processed_orig])
            seq2 = vectorizer.transform([processed_reord])
            identical = np.array_equal(seq1, seq2)
            print(f"Original: {original}")
            print(f"Reordered: {reordered}")
            print(f"Sequential vectors identical: {identical}")
            print("---")


def analyze_morphological_variations(vectorizer, word_groups):
    """
    Test how vectorizers handle morphological variations
    """
    print_section("Morphological Variation Analysis")
    
    word_groups = [
        ["run", "running", "ran", "runner"],
        ["good", "better", "best"],
        ["write", "writing", "wrote", "written"]
    ]
    
    for group in word_groups:
        print(f"Word group: {group}")
        
        # Check if words are in vocabulary
        if hasattr(vectorizer, 'word_to_idx'):
            in_vocab = [word in vectorizer.word_to_idx for word in group]
            print(f"In vocabulary: {in_vocab}")
            
            # Check indices (should be very different)
            indices = [vectorizer.word_to_idx.get(word, -1) for word in group]
            print(f"Assigned indices: {indices}")
        
        print("---")


def demonstrate_failure_cases():
    """
    Demonstrate specific NLP challenges
    """
    print_section("NLP Challenge Examples")
    
    # 1. Negation Handling
    print("1. Negation Handling:")
    negation_examples = [
        "The movie was not bad",  # Positive (double negative)
        "The movie was bad",      # Negative
        "The movie was not good", # Negative
        "The movie was good"      # Positive
    ]
    
    for example in negation_examples:
        processed = preprocess_text(example)
        print(f"Original: {example}")
        print(f"Processed: {processed}")
        print()
    
    # 2. Sarcasm and Irony
    print("2. Sarcasm and Irony:")
    sarcastic_examples = [
        "Oh great, another terrible movie",
        "Just what I needed, a broken product",
        "Perfect, exactly what I expected from this brand"
    ]
    
    for example in sarcastic_examples:
        processed = preprocess_text(example)
        print(f"Original: {example}")
        print(f"Processed: {processed}")
        print()
    
    # 3. Multi-word Expressions
    print("3. Multi-word Expressions:")
    phrase_examples = [
        "break a leg",     # Positive (good luck) but contains "break"
        "piece of cake",   # Easy, but contains neutral words
        "over the moon",   # Very happy, but spatially scattered
    ]
    
    for example in phrase_examples:
        processed = preprocess_text(example)
        print(f"Original: {example}")
        print(f"Processed: {processed}")
        print()


def plot_training_curves(results):
    """
    Plot training curves for all models
    """
    print_section("Training Curve Visualization")
    
    plt.figure(figsize=(12, 8))
    
    # Plot accuracy curves
    plt.subplot(2, 2, 1)
    for model_name, model_data in results.items():
        plt.plot(model_data['training_curve'], label=model_name)
    plt.title('Validation Accuracy Over Training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot final accuracies as bar chart
    plt.subplot(2, 2, 2)
    model_names = list(results.keys())
    accuracies = [results[name]['final_accuracy'] for name in model_names]
    plt.bar(model_names, accuracies)
    plt.title('Final Model Accuracies')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    
    # Add accuracy values on top of bars
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Training curves saved as 'training_results.png'")


def analyze_vocabulary_overlap(vectorizers):
    """
    Analyze vocabulary overlap between different vectorizers
    """
    print_section("Vocabulary Analysis")
    
    # Get vocabularies
    vocabs = {}
    for name, vectorizer in vectorizers.items():
        if hasattr(vectorizer, 'word_to_idx'):
            vocabs[name] = set(vectorizer.word_to_idx.keys())
    
    # Compare vocabularies
    for name1, vocab1 in vocabs.items():
        for name2, vocab2 in vocabs.items():
            if name1 != name2:
                overlap = len(vocab1.intersection(vocab2))
                total = len(vocab1.union(vocab2))
                jaccard = overlap / total if total > 0 else 0
                print(f"{name1} vs {name2}: {overlap} shared words, Jaccard: {jaccard:.3f}")


def create_sample_predictions(results, test_texts, test_labels):
    """
    Show sample predictions from each model
    """
    print_section("Sample Predictions")
    
    # Select a few test samples
    sample_indices = [0, 1, 2, 3, 4]
    
    for idx in sample_indices:
        text = test_texts[idx]
        true_label = "Positive" if test_labels[idx] == 1 else "Negative"
        
        print(f"\nSample {idx + 1}:")
        print(f"Text: {text[:100]}...")
        print(f"True Label: {true_label}")
        
        # Get predictions from each model
        for model_name, model_data in results.items():
            model = model_data['model']
            vectorizer = model_data['vectorizer']
            
            # Prepare input
            processed_text = preprocess_text(text)
            
            if hasattr(vectorizer, 'transform'):
                # BoW/TF-IDF
                import torch
                vec = vectorizer.transform([processed_text])
                input_tensor = torch.FloatTensor(vec)
            else:
                # Sequential
                import torch
                seq = vectorizer.transform([processed_text])
                input_tensor = torch.LongTensor(seq)
            
            # Get prediction
            model.eval()
            with torch.no_grad():
                output = model(input_tensor)
                prediction = torch.argmax(output, dim=1).item()
                confidence = torch.softmax(output, dim=1).max().item()
            
            pred_label = "Positive" if prediction == 1 else "Negative"
            print(f"  {model_name}: {pred_label} (confidence: {confidence:.3f})")


def main():
    """
    Main execution function for the NLP session
    """
    print_section("NLP Session 5: Introduction to NLP")
    
    # Step 1: Check if data exists, prepare if not
    data_dir = 'movie_reviews'
    processed_file = 'processed_data.pkl'
    
    if not os.path.exists(processed_file):
        if os.path.exists(data_dir):
            print(f"Preparing dataset from {data_dir}...")
            prepare_dataset(data_dir, processed_file)
        else:
            print(f"Error: Data directory {data_dir} not found!")
            print("Please ensure the movie_reviews directory exists with positive/ and negative/ subdirectories")
            sys.exit(1)
    else:
        print(f"Loading existing processed data from {processed_file}")
    
    # Step 2: Load processed data
    with open(processed_file, 'rb') as f:
        data = pickle.load(f)
    
    texts = data['texts']
    labels = data['labels']
    
    print(f"Dataset loaded: {len(texts)} samples")
    print(f"Positive samples: {sum(labels)}")
    print(f"Negative samples: {len(labels) - sum(labels)}")
    
    # Step 3: Split data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training set: {len(train_texts)} samples")
    print(f"Test set: {len(test_texts)} samples")
    
    # Step 4: Train all models
    print_section("Training All Models")
    results = comprehensive_evaluation(train_texts, train_labels, test_texts, test_labels)
    
    # Step 5: Display results summary
    print_section("Results Summary")
    for model_name, model_data in results.items():
        print(f"{model_name}: {model_data['final_accuracy']:.4f}")
    
    # Step 6: Plot training curves
    plot_training_curves(results)
    
    # Step 7: Analyze vocabularies
    vectorizers = {name: data['vectorizer'] for name, data in results.items()}
    analyze_vocabulary_overlap(vectorizers)
    
    # Step 8: Demonstrate failure cases
    demonstrate_failure_cases()
    
    # Step 9: Analyze semantic similarity failures
    # Use the best performing model for analysis
    best_model_name = max(results.keys(), key=lambda x: results[x]['final_accuracy'])
    best_model = results[best_model_name]['model']
    best_vectorizer = results[best_model_name]['vectorizer']
    
    print(f"\nUsing best model ({best_model_name}) for detailed analysis...")
    
    test_cases = [
        ("The movie was fantastic", "The film was excellent"),
        ("I love this product", "I adore this item"),
        ("The service was terrible", "The service was awful"),
        ("Very good quality", "Outstanding quality")
    ]
    
    analyze_semantic_similarity_failures(best_model, best_vectorizer, test_cases)
    
    # Step 10: Analyze word order sensitivity
    word_order_cases = [
        ("The dog chased the cat", "The cat chased the dog"),
        ("I really don't like this movie", "I don't really like this movie"),
        ("The restaurant has good food", "Good food has the restaurant")
    ]
    
    analyze_word_order_sensitivity(best_vectorizer, word_order_cases)
    
    # Step 11: Analyze morphological variations
    word_groups = [
        ["run", "running", "ran", "runner"],
        ["good", "better", "best"],
        ["write", "writing", "wrote", "written"]
    ]
    
    analyze_morphological_variations(best_vectorizer, word_groups)
    
    # Step 12: Show sample predictions
    create_sample_predictions(results, test_texts, test_labels)
    
    # Step 13: Final summary and next steps
    print_section("Session Summary and Next Steps")
    
    print("Key Findings:")
    print("1. TF-IDF typically outperforms simple Bag of Words")
    print("2. CNNs may struggle with text due to padding and fixed filters")
    print("3. All methods struggle with:")
    print("   - Semantic similarity (synonyms treated as different)")
    print("   - Word order sensitivity (especially for BoW)")
    print("   - Morphological variations (related words treated separately)")
    print("   - Negation, sarcasm, and complex linguistic phenomena")
    
    print("\nNext Session Preview:")
    print("- Word embeddings (Word2Vec, GloVe)")
    print("- Dense vector representations that capture semantic relationships")
    print("- Pretrained embeddings and transfer learning")
    print("- Visualizing word relationships in embedding space")
    
    print("\nFiles Generated:")
    print("- training_results.png: Training curves and accuracy comparison")
    print("- processed_data.pkl: Preprocessed dataset for future use")
    
    print("\nSession Complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSession interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)