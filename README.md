# Deep Learning with PyTorch - Session 5: Introduction to NLP

## Session Timeline

| Time      | Activity                                    |
| --------- | ------------------------------------------- |
| 0:00 - 0:10 | 1. Check-in + Session 4 Recap              |
| 0:10 - 0:30 | 2. Introduction to NLP & Text Preprocessing |
| 0:30 - 0:55 | 3. Text Representation Methods             |
| 0:55 - 1:20 | 4. Building Text Classifiers (MLP & CNN)   |
| 1:20 - 1:45 | 5. Analyzing Results & Method Limitations   |
| 1:45 - 2:00 | 6. Wrap-Up + Discussion of Better Methods  |

---

## 1. Check-in + Session 4 Recap

### Goals

Review CNN interpretability concepts and transition to Natural Language Processing fundamentals.

### Quick Recap Questions

* How did your attention-based CNN perform on MNIST? What did the attention maps show?
* Can you explain the difference between attention maps and CAM visualizations?
* What insights did you gain from comparing attention-based CAM with Grad-CAM?
* Did anyone notice cases where attention focused on unexpected regions?
* What are the main advantages and limitations of attention mechanisms?

### Session 4 Key Concepts

* **Attention Mechanisms** for built-in interpretability
* **Class Activation Mapping (CAM)** for understanding model decisions
* **Spatial attention** and global average pooling
* **Channel attention** and adaptive average pooling
* **Model interpretability** trade-offs and validation strategies

### What's New Today

Today we're shifting from computer vision to **Natural Language Processing (NLP)**! We'll explore how to represent text numerically and apply neural networks to language tasks.

**Key Focus Areas:**
- **Text Preprocessing** - cleaning, tokenization, and normalization
- **Numerical Text Representation** - bag-of-words, TF-IDF, and indexing
- **Python Text Processing** - regular expressions, file handling, exceptions
- **Text Classification** - sentiment analysis with multiple approaches
- **Network Architecture Comparison** - MLP vs CNN for text data
- **Method Limitations** - why basic representations fall short

**Why This Matters:**
- **Foundation for NLP** - understanding basic text processing pipeline
- **Practical Python Skills** - regex, file I/O, error handling
- **Architecture Intuition** - why different networks suit different data types
- **Problem Recognition** - identifying when methods are inadequate
- **Preparation for Advanced NLP** - motivation for embeddings and transformers

---

## 2. Introduction to NLP & Text Preprocessing

### Goals

* Understand the unique challenges of processing natural language
* Learn essential Python text processing techniques
* Implement robust text preprocessing pipelines
* Handle common text data issues with proper error handling

---

### The Challenge of Natural Language

**Why Text is Different from Images:**
- **Discrete vs Continuous**: Words are discrete tokens, pixels are continuous values
- **Sequential Structure**: Order matters critically in text
- **Variable Length**: Sentences have different lengths
- **Ambiguity**: Same word can have different meanings in different contexts
- **Sparsity**: Most words don't appear in most documents

**Example Challenges:**
```
"The bank is closed" vs "The river bank is steep"
"Running water" vs "I am running" vs "Running shoes"
"Great!" vs "great" vs "GREAT" vs "greaat"
```

### Python Text Processing Essentials

**Regular Expressions for Text Cleaning:**
```python
import re

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
```

**File Handling with Error Management:**
```python
import os
from pathlib import Path

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
```

### Text Preprocessing Pipeline

**Complete Preprocessing Function:**
```python
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
```

### Dataset: Movie Review Sentiment Analysis

**Dataset Structure:**
```
movie_reviews/
├── positive/
│   ├── review_001.txt
│   ├── review_002.txt
│   └── ...
└── negative/
    ├── review_001.txt
    ├── review_002.txt
    └── ...
```

**Example Reviews:**
```
Positive: "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout."
Negative: "Terrible film. Poor acting, confusing storyline, and completely boring. Waste of time."
```

**Loading and Preprocessing:**
```python
# Load dataset
texts, labels = load_dataset_from_directory('movie_reviews')

# Preprocess texts
processed_texts = [preprocess_text(text) for text in texts]

# Convert labels to binary
label_map = {'positive': 1, 'negative': 0}
binary_labels = [label_map[label] for label in labels]

print(f"Dataset size: {len(processed_texts)} reviews")
print(f"Positive reviews: {sum(binary_labels)}")
print(f"Negative reviews: {len(binary_labels) - sum(binary_labels)}")
```

---

## 3. Text Representation Methods

### Goals

* Implement multiple numerical text representation methods
* Understand the mathematics behind bag-of-words and TF-IDF
* Handle variable-length sequences with padding
* Compare different vectorization approaches

---

### Method 1: Simple Word Indexing (Bag of Words)

**Concept:**
Each unique word gets an index, documents become vectors of word counts.

**Mathematical Foundation:**
For vocabulary $V = \{w_1, w_2, ..., w_n\}$ and document $d$:
$$\text{BoW}(d) = [c_1, c_2, ..., c_n]$$
where $c_i$ is the count of word $w_i$ in document $d$.

**Implementation:**
```python
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
```

### Method 2: TF-IDF (Term Frequency-Inverse Document Frequency)

**Mathematical Foundation:**
$$\text{TF}(t,d) = \frac{\text{count}(t,d)}{\text{total terms in } d}$$

$$\text{IDF}(t,D) = \log\left(\frac{|D|}{|\{d \in D : t \in d\}|}\right)$$

$$\text{TF-IDF}(t,d,D) = \text{TF}(t,d) \times \text{IDF}(t,D)$$

**Implementation:**
```python
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
```

### Method 3: Sequential Indexing for CNNs

**Concept:**
Represent text as sequences of word indices, pad to fixed length.

**Implementation:**
```python
class SequentialIndexer:
    def __init__(self, max_features=5000, max_length=200):
        self.max_features = max_features
        self.max_length = max_length
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}  # Special tokens
        self.vocabulary_size = 2  # Start with special tokens
    
    def fit(self, texts):
        """Build vocabulary from training texts"""
        word_counts = {}
        
        for text in texts:
            for word in text.split():
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency and take top max_features-2 (account for special tokens)
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        vocab_words = [word for word, count in sorted_words[:self.max_features-2]]
        
        # Add to vocabulary
        for word in vocab_words:
            self.word_to_idx[word] = self.vocabulary_size
            self.vocabulary_size += 1
        
        return self
    
    def transform(self, texts):
        """Convert texts to padded sequences"""
        sequences = []
        
        for text in texts:
            # Convert words to indices
            sequence = []
            for word in text.split():
                if word in self.word_to_idx:
                    sequence.append(self.word_to_idx[word])
                else:
                    sequence.append(self.word_to_idx['<UNK>'])  # Unknown word
            
            # Pad or truncate to max_length
            if len(sequence) < self.max_length:
                sequence.extend([0] * (self.max_length - len(sequence)))  # Pad with 0
            else:
                sequence = sequence[:self.max_length]  # Truncate
            
            sequences.append(sequence)
        
        return np.array(sequences)
    
    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)
```

### Comparison of Methods

**Advantages and Disadvantages:**

**Bag of Words:**
- ✅ Simple, interpretable
- ✅ Works well for short texts
- ❌ Ignores word order
- ❌ Sparse, high-dimensional

**TF-IDF:**
- ✅ Reduces impact of common words
- ✅ Highlights discriminative terms
- ❌ Still ignores word order
- ❌ Assumes independence of terms

**Sequential Indexing:**
- ✅ Preserves word order
- ✅ Compatible with CNNs/RNNs
- ❌ Requires fixed-length padding
- ❌ Padding introduces noise

---

## 4. Building Text Classifiers (MLP & CNN)

### Goals

* Implement MLP classifier for bag-of-words representations
* Build CNN classifier for sequential text data
* Compare architecture choices for different text representations
* Understand why traditional architectures struggle with text

---

### MLP Classifier for BoW/TF-IDF

**Architecture Design:**
```python
class TextMLP(nn.Module):
    def __init__(self, vocab_size, hidden_size=128, num_classes=2, dropout=0.5):
        super(TextMLP, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(vocab_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)
```

**Training Setup:**
```python
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
```

### CNN Classifier for Sequential Text

**Architecture Design:**
```python
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, num_filters=100, 
                 filter_sizes=[3, 4, 5], num_classes=2, dropout=0.5):
        super(TextCNN, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Convolutional layers with different filter sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        # Dropout and final classifier
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(len(filter_sizes) * num_filters, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        x = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)
        x = x.transpose(1, 2)  # (batch_size, embedding_dim, sequence_length)
        
        # Apply convolutions
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(x))  # (batch_size, num_filters, conv_length)
            pooled = torch.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)
        
        # Concatenate all conv outputs
        x = torch.cat(conv_outputs, dim=1)  # (batch_size, len(filter_sizes) * num_filters)
        x = self.dropout(x)
        
        return self.classifier(x)
```

**Training Setup:**
```python
def train_cnn_classifier(train_texts, train_labels, val_texts, val_labels, 
                        max_length=200, epochs=50):
    """
    Train CNN classifier with sequential text representation
    """
    # Vectorize text as sequences
    indexer = SequentialIndexer(max_features=5000, max_length=max_length)
    X_train = indexer.fit_transform(train_texts)
    X_val = indexer.transform(val_texts)
    
    # Convert to tensors
    X_train = torch.LongTensor(X_train)
    X_val = torch.LongTensor(val_val)
    y_train = torch.LongTensor(train_labels)
    y_val = torch.LongTensor(val_labels)
    
    # Initialize model
    model = TextCNN(vocab_size=indexer.vocabulary_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop (similar to MLP)
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
    
    return model, indexer, train_losses, val_accuracies
```

### Why These Architectures Struggle

**MLP Limitations:**
1. **Bag of Words Assumption**: Treats text as unordered collection of words
2. **No Spatial/Sequential Understanding**: Cannot capture word order or local patterns
3. **High Dimensionality**: Sparse, high-dimensional input vectors
4. **No Shared Parameters**: Each word position learned independently

**CNN Limitations for Text:**
1. **Fixed Receptive Fields**: Convolution filters have fixed sizes
2. **Local Patterns Only**: Cannot capture long-range dependencies
3. **Translation Invariance**: Assumes patterns are equally important everywhere
4. **Word Order Rigidity**: Sensitive to exact word positioning

**Fundamental Text Challenges:**
```python
# Examples showing why basic methods fail:

# 1. Synonyms treated as completely different
"The movie was great" vs "The film was excellent"
# BoW: [0,1,0,1,1,0,0,1,0] vs [0,1,0,0,0,1,1,0,1]
# No similarity captured!

# 2. Word order ignored
"The dog bit the man" vs "The man bit the dog"
# BoW representation is identical!

# 3. Morphological variations
"run", "running", "ran", "runner"
# Treated as completely different words

# 4. Typos and variations
"great", "grate", "gr8", "greaat"
# No similarity captured
```

---

## 5. Analyzing Results & Method Limitations

### Goals

* Compare performance across different text representations
* Identify specific failure modes of each approach
* Understand why certain architectures perform better than others
* Prepare motivation for advanced NLP methods

---

### Comprehensive Performance Analysis

**Model Comparison Framework:**
```python
def comprehensive_evaluation(train_texts, train_labels, test_texts, test_labels):
    """
    Compare all methods and provide detailed analysis
    """
    results = {}
    
    # 1. MLP with Bag of Words
    print("Training MLP with Bag of Words...")
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
    print("Training MLP with TF-IDF...")
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
    
    # 3. CNN with Sequential Indexing
    print("Training CNN with Sequential Indexing...")
    cnn_model, seq_indexer, cnn_losses, cnn_acc = train_cnn_classifier(
        train_texts, train_labels, test_texts, test_labels
    )
    results['CNN_Sequential'] = {
        'model': cnn_model,
        'vectorizer': seq_indexer,
        'final_accuracy': cnn_acc[-1],
        'training_curve': cnn_acc
    }
    
    return results
```

### Failure Mode Analysis

**1. Synonym/Semantic Similarity Issues:**
```python
def analyze_semantic_similarity_failures(model, vectorizer, test_cases):
    """
    Test how models handle semantically similar but lexically different texts
    """
    test_cases = [
        ("The movie was fantastic", "The film was excellent"),
        ("I love this product", "I adore this item"),
        ("The service was terrible", "The service was awful"),
        ("Very good quality", "Outstanding quality")
    ]
    
    for text1, text2 in test_cases:
        if hasattr(vectorizer, 'transform'):
            # For BoW/TF-IDF
            vec1 = vectorizer.transform([text1])
            vec2 = vectorizer.transform([text2])
            
            # Cosine similarity
            similarity = cosine_similarity(vec1, vec2)[0][0]
            
        # Get model predictions
        pred1 = model(torch.FloatTensor(vec1))
        pred2 = model(torch.FloatTensor(vec2))
        
        print(f"Text 1: {text1}")
        print(f"Text 2: {text2}")
        print(f"Vector similarity: {similarity:.4f}")
        print(f"Prediction difference: {torch.abs(pred1 - pred2).max().item():.4f}")
        print("---")
```

**2. Word Order Sensitivity:**
```python
def analyze_word_order_sensitivity(model, vectorizer, test_cases):
    """
    Test how models handle word order changes
    """
    test_cases = [
        ("The dog chased the cat", "The cat chased the dog"),
        ("I really don't like this movie", "I don't really like this movie"),
        ("The restaurant has good food", "Good food has the restaurant")
    ]
    
    for original, reordered in test_cases:
        # Test with different vectorizers
        if isinstance(vectorizer, BagOfWordsVectorizer):
            # BoW should be identical
            vec1 = vectorizer.transform([original])
            vec2 = vectorizer.transform([reordered])
            print(f"BoW vectors identical: {np.array_equal(vec1, vec2)}")
        
        elif isinstance(vectorizer, SequentialIndexer):
            # Sequential should be different
            seq1 = vectorizer.transform([original])
            seq2 = vectorizer.transform([reordered])
            print(f"Sequential vectors identical: {np.array_equal(seq1, seq2)}")
```

**3. Morphological Variation Issues:**
```python
def analyze_morphological_variations(vectorizer, word_groups):
    """
    Test how vectorizers handle morphological variations
    """
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
```

### Expected Performance Patterns

**Typical Results:**
- **MLP + BoW**: ~75-80% accuracy, fast training
- **MLP + TF-IDF**: ~80-85% accuracy, slightly better than BoW
- **CNN + Sequential**: ~70-75% accuracy, slower training

**Why CNN Often Underperforms:**
1. **Padding Noise**: Zero-padding introduces artificial patterns
2. **Fixed Filter Sizes**: Cannot adapt to variable phrase lengths
3. **Local Focus**: Misses long-range dependencies crucial for sentiment
4. **Word Order Brittleness**: Small changes in word order affect filters differently

**Why TF-IDF Often Wins:**
1. **Discriminative Features**: Highlights words that distinguish classes
2. **Noise Reduction**: Reduces impact of common, uninformative words
3. **Mature Method**: Well-tuned for document classification tasks
4. **No Architecture Constraints**: Direct mapping from features to classes

---

## 6. Wrap-Up + Discussion of Better Methods

### Goals

* Synthesize learnings about text representation limitations
* Motivate the need for more sophisticated NLP methods
* Preview upcoming techniques that address current limitations
* Discuss the evolution of NLP and its trajectory

---

### Summary of Key Limitations

**Representational Issues:**
1. **Vocabulary Explosion**: Each unique word form gets separate representation
2. **Semantic Blindness**: No understanding of word meaning or relationships
3. **Context Ignorance**: Same word treated identically in different contexts
4. **Morphological Insensitivity**: Related word forms treated as unrelated

**Architectural Mismatches:**
1. **MLPs**: Designed for fixed-size, dense features - not sparse text
2. **CNNs**: Designed for spatial locality - not sequential dependencies
3. **Both**: Cannot model long-range dependencies essential for language

### Concrete Examples of Failures

**1. Negation Handling:**
```python
# These should have opposite sentiments but might be classified similarly
examples = [
    "The movie was not bad",  # Positive (double negative)
    "The movie was bad",      # Negative
    "The movie was not good", # Negative
    "The movie was good"      # Positive
]

# BoW representation ignores word order entirely
# CNN might miss the negation if "not" and "bad" are in different filters
```

**2. Sarcasm and Irony:**
```python
# Sarcastic reviews that basic methods will misclassify
sarcastic_examples = [
    "Oh great, another terrible movie",  # Negative sentiment, positive words
    "Just what I needed, a broken product",  # Negative, but hard to detect
    "Perfect, exactly what I expected from this brand"  # Context-dependent
]
```

**3. Multi-word Expressions:**
```python
# Phrases where individual words don't capture meaning
phrase_examples = [
    "break a leg",     # Positive (good luck) but contains "break"
    "piece of cake",   # Easy, but contains neutral words
    "over the moon",   # Very happy, but spatially scattered in sequence
]
```

### The Path Forward: Advanced NLP Methods

**What We Need:**
1. **Semantic Understanding**: Word representations that capture meaning
2. **Context Awareness**: Same word, different meanings in different contexts
3. **Compositional Semantics**: Understanding how words combine to create meaning
4. **Long-range Dependencies**: Modeling relationships across entire sequences

**Preview of Solutions:**

**Word Embeddings (Next Session):**
- Dense vector representations that capture semantic relationships
- Words with similar meanings have similar vectors
- Trained on large corpora to capture distributional semantics

**Recurrent Neural Networks:**
- Designed for sequential data with variable lengths
- Can model dependencies across entire sequences
- Memory mechanisms to retain important information

**Attention Mechanisms:**
- Allow models to focus on relevant parts of input
- Overcome fixed receptive field limitations
- Enable direct modeling of long-range dependencies

**Transformer Architecture:**
- Self-attention mechanisms for parallel processing
- Positional encoding to understand sequence order
- Foundation for modern NLP breakthroughs

### Practical Takeaways

**When to Use Basic Methods:**
- Small datasets where advanced methods might overfit
- Interpretability is crucial (TF-IDF weights are human-readable)
- Computational resources are extremely limited
- Baseline comparisons for more complex methods

**Red Flags for Basic Methods:**
- Tasks requiring understanding of word order (syntax, negation)
- Semantic similarity is important (synonyms, paraphrases)
- Long documents with complex discourse structure
- Multilingual or morphologically rich languages

**Transition Strategy:**
1. Start with TF-IDF baseline for any text classification task
2. Identify specific failure modes through error analysis
3. Choose advanced methods that address those specific issues
4. Always compare against the simple baseline

### Discussion Questions

1. **Architecture Choice**: Given our results, why might CNNs work better for computer vision than NLP? What fundamental differences between images and text drive this?

2. **Feature Engineering**: How could we improve our basic text representations? What preprocessing steps might help address some limitations?

3. **Evaluation Metrics**: Are accuracy scores sufficient for understanding model performance on text? What other metrics might be more informative?

4. **Real-world Applications**: Where might these basic methods still be preferred over more complex approaches? What are the trade-offs?

5. **Data Requirements**: How do you think the amount of training data affects the relative performance of these methods?

### Homework and Further Exploration

**Beginner Level:**
- Implement a simple spell-checker using edit distance
- Experiment with different preprocessing strategies (stemming, lemmatization)
- Try the methods on a different text classification dataset
- Analyze prediction errors to identify common failure patterns

**Intermediate Level:**
- Implement n-gram features (bigrams, trigrams) and compare performance
- Build a simple language model using character-level representations
- Create visualizations showing the most important TF-IDF features per class
- Experiment with different CNN architectures (varying filter sizes, depths)

**Advanced Level:**
- Implement a basic word2vec-style embedding training algorithm
- Build a hierarchical attention mechanism for document classification
- Create an ensemble model combining multiple text representations
- Analyze how performance varies with document length and vocabulary size

**Research Directions:**
- Investigate why CNNs work better for some NLP tasks (like sentence classification) than others
- Explore domain adaptation: how do models trained on one type of text perform on another?
- Study the relationship between text preprocessing choices and model performance
- Examine bias in text representations and its impact on downstream tasks

### Preparing for Next Session

**What's Coming:**
- **Word Embeddings**: Dense vector representations of words
- **Word2Vec and GloVe**: Methods for learning semantic representations
- **Pretrained Embeddings**: Leveraging large-scale linguistic knowledge
- **Embedding Spaces**: Understanding and visualizing semantic relationships

**Prerequisites to Review:**
- Linear algebra: dot products, cosine similarity, vector spaces
- Probability: conditional probability, language modeling basics
- Optimization: gradient descent, backpropagation through embeddings

**Practical Preparation:**
- Ensure you have a text editor that can handle large files (we'll work with embeddings)
- Familiarize yourself with dimensionality reduction techniques (PCA, t-SNE)
- Think about semantic relationships between words that you'd like to see captured

---

## Additional Resources

**Python Text Processing:**
- **Regular Expressions**: Python `re` module documentation
- **NLTK**: Natural Language Toolkit for advanced preprocessing
- **spaCy**: Industrial-strength NLP library
- **scikit-learn**: `CountVectorizer` and `TfidfVectorizer` implementations

**Datasets for Practice:**
- **IMDB Movie Reviews**: Larger sentiment analysis dataset
- **20 Newsgroups**: Multi-class text classification
- **Amazon Product Reviews**: Domain-specific sentiment analysis
- **Yelp Reviews**: Business review classification

**Theoretical Background:**
- **"Foundations of Statistical Natural Language Processing"** by Manning & Schütze
- **"Speech and Language Processing"** by Jurafsky & Martin
- **Information Retrieval**: Understanding TF-IDF and vector space models
- **"The Elements of Statistical Learning"**: Mathematical foundations

**Tools and Libraries:**
- **Matplotlib/Seaborn**: For visualizing training curves and performance metrics
- **Pandas**: For data manipulation and analysis
- **NumPy**: For efficient numerical operations
- **Scikit-learn**: For baseline implementations and evaluation metrics

This session establishes the foundation for understanding why traditional ML approaches struggle with natural language, setting up the motivation for the word embeddings and more sophisticated architectures that follow. The hands-on implementation of multiple approaches gives students direct experience with the limitations we'll be addressing in future sessions.