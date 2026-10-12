# Document Clustering Using TF-IDF and K-means

## 1. What is document clustering?

Document clustering is an unsupervised-learning task that groups text documents according to similarities in their content or representation. The algorithm is not given the correct category for each document in advance. Instead, it attempts to discover structure in the collection.

Examples include grouping news articles by topic, organizing customer reviews, exploring a collection of research abstracts, and grouping support tickets with similar wording.

## 2. Why do we need TF-IDF?

Machine-learning algorithms such as K-means operate on numerical features, not raw sentences. A text representation is therefore needed. A simple word count treats every occurrence as equally informative. TF-IDF adjusts word weights so that terms that are common across nearly all documents are usually less influential than terms that distinguish particular documents.

For example, words such as “the” and “is” may appear in many documents, while “earthquake” may be especially useful for distinguishing documents about earthquakes. TF-IDF estimates the importance of a term in one document relative to the whole collection.

## 3. Understanding TF-IDF

### 3.1 Term Frequency (TF)

Term Frequency measures how often a term occurs within a document. One common version divides the term count by the total number of terms in that document:

TF(t, d) = count of term t in document d / total terms in document d

Example: if a document has 10 words and “fraud” occurs twice, its normalized TF is 2/10 = 0.2. Other TF definitions are also used, including raw counts and logarithmically scaled counts.

### 3.2 Inverse Document Frequency (IDF)

Inverse Document Frequency reflects how widely a term is distributed across the document collection. A term found in nearly every document is less useful for distinguishing documents than a term found in only a few.

IDF(t) = log(N / DF(t))

Here, N is the total number of documents and DF(t) is the number of documents containing term t. With 10 documents, if “machine” appears in 8 documents, its IDF is log(10/8). If “fraud” appears in 2 documents, its IDF is log(10/2), which is larger. The logarithm base changes the numerical scale, not the basic idea.

Implementations often use smoothing, such as scikit-learn’s default formula log((1 + N)/(1 + DF(t))) + 1, to avoid zero or extreme values and to keep observed terms’ IDF values positive.

### 3.3 Combining TF and IDF

TF-IDF(t, d) = TF(t, d) × IDF(t)

A term tends to receive a high TF-IDF weight when it appears relatively often in a document but appears in relatively few documents overall. A term can still have a low score if it is rare in the collection but does not occur in the particular document, or if its within-document frequency is low.

## 4. Example collection of documents

Consider the following six short documents:

|              |                                                |
|--------------|------------------------------------------------|
| **Document** | **Text**                                       |
| **D1**       | Football players scored goals in the match     |
| **D2**       | The team won the football championship         |
| **D3**       | Cricket players scored runs in the match       |
| **D4**       | The government announced a new policy          |
| **D5**       | Parliament discussed government policy reforms |
| **D6**       | Ministers debated the new economic policy      |

If we choose K = 2, we might expect one cluster to contain sports-related documents and another to contain government/politics-related documents. This is only an expectation: clustering is based on feature similarity, so actual assignments depend on the representation, algorithm settings, and data.

## 5. The complete workflow

1. Collect documents: Gather the text items to be clustered.
2. Preprocess (optional): For example, normalize case and remove common stop words. Preprocessing choices affect results.
3. Vectorize: Use TF-IDF to turn documents into a document-term matrix.
4. Cluster: Fit K-means to the TF-IDF matrix and assign each document a cluster ID.
5. Interpret: Review cluster members and high-weight terms to understand the groups.
6. Evaluate: Try plausible K values and inspect whether clusters are useful and coherent.

## 6. Python implementation

Install scikit-learn and Matplotlib if needed. In Google Colab, they are often already available.

```python
# If required, run in a notebook:
%pip install scikit-learn matplotlib
```

### 6.1 Create the TF-IDF matrix

```python
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "Football players scored goals in the match",
    "The team won the football championship",
    "Cricket players scored runs in the match",
    "The government announced a new policy",
    "Parliament discussed government policy reforms",
    "Ministers debated the new economic policy"
]

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(documents)

print("Matrix shape:", X.shape)
print("Vocabulary:", vectorizer.get_feature_names_out())
```

TfidfVectorizer learns the vocabulary and document frequencies, then creates the TF-IDF representation. fit_transform() performs fitting and transformation together. X is usually a sparse matrix: most entries are zero because each document contains only a small fraction of the full vocabulary.

### 6.2 Fit K-means

```python
from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

labels = kmeans.fit_predict(X)

for i, label in enumerate(labels):
    print(f"Document {i + 1}: Cluster {label}")
```

n_clusters=2 asks for two clusters. random_state makes the initialization reproducible. n_init=10 runs multiple initializations and keeps the best result according to the K-means objective. fit_predict() fits the model and returns a cluster label for each document.

### 6.3 Display the document assignments

```python
for i, document in enumerate(documents):
    print(f"D{i + 1} | Cluster {labels[i]} | {document}")
```

Cluster IDs such as 0 and 1 are arbitrary. Do not assume that Cluster 0 always means sports or that Cluster 1 always means politics. Interpret each cluster by examining its documents and characteristic terms.

### 6.4 Inspect the top terms in each cluster

```python
terms = vectorizer.get_feature_names_out()
centers = kmeans.cluster_centers_

for cluster_id, center in enumerate(centers):
    top_indices = center.argsort()[::-1][:5]
    top_words = [terms[i] for i in top_indices]
    print(f"Cluster {cluster_id}: {top_words}")
```

The centroid contains a weight for each TF-IDF feature. Printing the largest weights can help summarize a cluster. These terms are clues, not guaranteed labels; always inspect the documents too.

## 7. Choosing K: the elbow method

K-means requires the number of clusters, K, to be selected in advance. The elbow method fits models for several values of K and plots the within-cluster sum of squares (WCSS), called inertia\_ in scikit-learn. WCSS measures the sum of squared Euclidean distances from each point to its assigned cluster centroid. It generally decreases as K increases, so look for a point where additional clusters yield diminishing improvements.

```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

wcss = []
k_values = range(1, 7)

for k in k_values:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    model.fit(X)
    wcss.append(model.inertia_)

plt.plot(list(k_values), wcss, marker="o")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Within-cluster sum of squares (WCSS)")
plt.title("Elbow Method for Document Clustering")
plt.xticks(list(k_values))
plt.show()
```

With only six documents, the elbow may not be clear and the dataset is too small for reliable conclusions. The elbow method is a heuristic rather than a definitive test. Also compare cluster interpretability and, where appropriate, metrics such as silhouette score.

## 8. Where is TF-IDF used?

| **Use case**                         | **How TF-IDF helps**                                                                     |
|---|---|
| **Document clustering**              | Group news articles, research papers, or support tickets by word-use similarity.         |
| **Search and information retrieval** | Help rank documents according to how well their terms match a query.                     |
| **Text classification**              | Provide features for models such as logistic regression, linear SVM, or Naive Bayes.     |
| **Keyword extraction**               | Identify terms that are relatively distinctive in a document compared with a collection. |

## 9. TF-IDF compared with other text representations

| **Method**      | **Main idea**                                                         | **Typical use**                        |
|-----------------|-----------------------------------------------------------------------|----------------------------------------|
| Bag of Words    | Represents word counts                                                | Basic text features                    |
| TF-IDF          | Weights words by within-document frequency and collection-wide rarity | Search, clustering, classification     |
| Word2Vec        | Learns dense word vectors from context                                | Word relationships and downstream NLP  |
| BERT embeddings | Produces contextual representations of text                           | Semantic similarity, search, NLP tasks |

## 10. Limitations and practical considerations

- TF-IDF mainly represents word importance, not meaning. “Car” and “automobile” may be treated as separate features.
- Word order and most syntax are not represented by the standard bag-of-words TF-IDF approach.
- The feature matrix is often high-dimensional and sparse. Keep it sparse; do not automatically call toarray() for large collections.
- K-means uses Euclidean distance and centroid-based clusters. This may not always match the similarity notion desired for text.
- For many text tasks, cosine similarity is useful. Alternatives include spherical K-means or clustering dense sentence embeddings.
- Preprocessing decisions, stop-word lists, n-grams, min_df, max_df, and normalization can change the results.
- A cluster number is not a semantic category. Validate clusters by reviewing their documents and terms.
