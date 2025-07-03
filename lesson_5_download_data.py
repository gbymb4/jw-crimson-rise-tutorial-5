# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 19:58:31 2025

@author: Gavin
"""

# Movie Review Dataset Setup
# This code will help you get the dataset the exercise is expecting

import os
import urllib.request
import tarfile
import shutil

def download_movie_review_dataset():
    """
    Download and setup the Cornell Movie Review Dataset
    This is likely what the exercise is referencing
    """
    # Cornell Movie Review Dataset (Pang & Lee, 2004)
    url = "http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz"
    
    print("Downloading Cornell Movie Review Dataset...")
    
    # Create directory
    os.makedirs("data", exist_ok=True)
    
    # Download file
    filename = "data/review_polarity.tar.gz"
    urllib.request.urlretrieve(url, filename)
    
    # Extract
    with tarfile.open(filename, 'r:gz') as tar:
        tar.extractall("data/")
    
    # Reorganize to match expected structure
    os.makedirs("movie_reviews", exist_ok=True)
    os.makedirs("movie_reviews/positive", exist_ok=True)
    os.makedirs("movie_reviews/negative", exist_ok=True)
    
    # Move files to expected structure
    pos_source = "data/txt_sentoken/pos"
    neg_source = "data/txt_sentoken/neg"
    
    if os.path.exists(pos_source):
        for file in os.listdir(pos_source):
            shutil.copy(os.path.join(pos_source, file), 
                       os.path.join("movie_reviews/positive", file))
    
    if os.path.exists(neg_source):
        for file in os.listdir(neg_source):
            shutil.copy(os.path.join(neg_source, file), 
                       os.path.join("movie_reviews/negative", file))
    
    # Clean up
    shutil.rmtree("data/txt_sentoken")
    os.remove(filename)
    
    print("Dataset ready! Structure:")
    print("movie_reviews/")
    print("├── positive/ (1000 files)")
    print("└── negative/ (1000 files)")

# Alternative: Use sklearn's built-in dataset
def use_sklearn_dataset():
    """
    Use scikit-learn's built-in dataset and save to expected format
    """
    # Or use a different built-in dataset
    
    # This is a workaround if download doesn't work
    print("Using sklearn alternative...")
    
    # You could also manually create sample data for testing
    sample_positive = [
        "This movie was absolutely fantastic! Great acting and plot.",
        "Loved every minute of it. Highly recommend!",
        "Best film I've seen this year. Outstanding performances.",
        "Brilliant storytelling and amazing cinematography.",
        "Perfect entertainment. Worth watching multiple times."
    ]
    
    sample_negative = [
        "Terrible movie. Waste of time and money.",
        "Boring plot, bad acting. Completely disappointing.",
        "One of the worst films ever made. Avoid at all costs.",
        "Poor direction and weak storyline. Very unsatisfying.",
        "Awful movie. I want my money back."
    ]
    
    # Create directory structure
    os.makedirs("movie_reviews/positive", exist_ok=True)
    os.makedirs("movie_reviews/negative", exist_ok=True)
    
    # Save sample data
    for i, review in enumerate(sample_positive):
        with open(f"movie_reviews/positive/sample_{i:03d}.txt", "w") as f:
            f.write(review)
    
    for i, review in enumerate(sample_negative):
        with open(f"movie_reviews/negative/sample_{i:03d}.txt", "w") as f:
            f.write(review)
    
    print("Sample dataset created for testing!")

if __name__ == "__main__":
    try:
        download_movie_review_dataset()
    except Exception as e:
        print(f"Download failed: {e}")
        print("Creating sample dataset instead...")
        use_sklearn_dataset()