# 📰 News Topic Modeling: Uncovering Hidden Themes

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Library](https://img.shields.io/badge/Lib-Gensim-orange) ![Library](https://img.shields.io/badge/Lib-pyLDAvis-red) ![Topic](https://img.shields.io/badge/NLP-Topic%20Modeling-green)

### 🔍 Project Overview
In large archives of unstructured text, finding specific information can be like looking for a needle in a haystack. This project uses **Topic Modeling** to automatically organize and summarize a dataset of news articles.

By applying **Latent Dirichlet Allocation (LDA)**, the model "reads" thousands of articles and groups them into distinct topics (e.g., "Politics," "Sports," "Technology") based on the statistical probability of word occurrences.

---

### ⚙️ The NLP Pipeline

#### 1. Data Preprocessing
Raw text is messy. Before modeling, I cleaned the dataset to ensure high-quality features:
* **Tokenization:** Breaking text into individual words.
* **Stop-Word Removal:** Removing common noise words (e.g., "the", "is", "and").
* **Lemmatization:** Reducing words to their root form (e.g., "running" -> "run") to consolidate meaning.
* **Bigram/Trigram Generation:** Grouping common phrases (e.g., "New York", "Machine Learning").

#### 2. Feature Engineering
* **Dictionary Creation:** Mapping unique words to IDs.
* **Bag of Words (BoW):** Converting documents into a matrix of word frequencies.

#### 3. Model Training (LDA)
I utilized **Gensim's LDA Model**, a generative probabilistic model that assumes:
1.  Each document is a mixture of various topics.
2.  Each topic is a mixture of various words.

The model iterates through the data to find the optimal distribution of words that define these hidden topics.

---

### 📊 Visualization & Results
To interpret the results, I implemented **pyLDAvis**, an interactive visualization tool.

* **Intertopic Distance Map:** Shows how distinct or similar the discovered topics are (clusters that are far apart represent very different subjects).
* **Top-30 Saliency Terms:** Bar charts that display the most relevant keywords for any selected topic.

> *Note: The interactive visualization allows users to slide the lambda parameter ($\lambda$) to adjust the weight of specific terms vs. frequent terms.*

---

### 🛠️ Setup & Usage
1.  **Install Dependencies:**
    ```bash
    pip install gensim pyLDAvis nltk pandas numpy
    ```
2.  **Run the Analysis:**
    Open `Karthik_News Modeling.ipynb` in Jupyter Notebook.
3.  **Input Data:**
    The notebook expects a corpus of text documents (e.g., the 20 Newsgroups dataset or similar news archives).

---

### 👨‍💻 About the Author
**Karthik Kunnamkumarath**
*Aerospace Engineer | Project Management Professional (PMP) | AI Solutions Developer*

I combine engineering precision with data science to solve complex problems.
* 📍 Toronto, ON
* 💼 [LinkedIn Profile](https://linkedin.com/in/4karthik95)
* 📧 Aero13027@gmail.com

---

### 💻 Code Snippet: Visualizing Topics
Here is how I generated the interactive dashboard to explore the results:

```python
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# Prepare the visualization data
# lda_model: The trained model
# corpus: The Bag of Words corpus
# id2word: The dictionary mapping
lda_vis_data = gensimvis.prepare(lda_model, corpus, id2word)

# Render the interactive HTML graph
pyLDAvis.display(lda_vis_data)
