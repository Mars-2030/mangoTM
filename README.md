
# Multilingual Topic Modeling Dashboard

An interactive web application built with Streamlit and Gensim for performing and visualizing topic modeling on textual data. This dashboard supports multiple languages and offers a rich set of preprocessing, modeling, and evaluation options to help users discover hidden thematic structures in their documents.



---

## ✨ Features

*   **🌐 Multilingual Support:** Analyze text in English, Spanish, Chinese, and Japanese right out of the box.
*   **🚀 Powered by Gensim:** Leverages the high-performance, industry-standard Gensim library for LDA modeling (`LdaMulticore`).
*   **🧩 Advanced Preprocessing Pipeline:**
    *   Comprehensive text cleaning (URLs, HTML, special characters, numbers).
    *   POS-tag aware **lemmatization** for English.
    *   Custom stopword lists and interactive stopword selection.
*   **🧠 Intelligent Feature Engineering:**
    *   **Phrase Detection:** Automatically detects and combines common bigrams (e.g., "machine learning" becomes "machine_learning") using `gensim.models.Phrases`.
    *   **TF-IDF Feature Selection:** Option to filter the vocabulary to keep only the most significant terms before training the LDA model.
*   **📊 Rigorous Model Evaluation:**
    *   Calculates and displays three key metrics: **C_v Coherence**, **U_Mass Coherence**, and **Perplexity**.
    *   Provides human-readable interpretations of scores.
*   **🎨 Interactive Visualizations:**
    *   Dynamic Word Clouds for each topic.
    *   Interactive bar charts of top words.
    *   Topic similarity heatmap.
*   **👤 Deep-Dive User Analysis:**
    *   Scatter plot to profile user engagement vs. topic focus.
    *   Per-user topic distribution donut charts.
    *   Topic evolution area charts over time for individual users.
*   **透明 Transparency & Export:**
    *   View the preprocessed text (tokens) before they are fed into the model.
    *   Download the complete preprocessed dataset as a CSV for offline analysis.

---

## 🛠️ Technical Stack

*   **Framework:** Streamlit
*   **Topic Modeling:** Gensim
*   **Core Libraries:** Pandas, NumPy, Scikit-learn
*   **NLP & Tokenization:** NLTK, Jieba (Chinese), Janome (Japanese)
*   **Visualization:** Plotly, Matplotlib, Seaborn, WordCloud

---

## ⚙️ Setup and Installation

Follow these steps to get the application running on your local machine.

### 1. Prerequisites
*   Python 3.8+
*   `pip` and `venv`

### 2. Clone the Repository
```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 3. Set Up a Virtual Environment (Recommended)
```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

### 4. Install Dependencies
Create a file named `requirements.txt` in your project folder and paste the following content into it:
```txt
streamlit
pandas
numpy
scikit-learn
gensim
matplotlib
seaborn
nltk
emoji
jieba
janome
wordcloud
plotly
```
Now, install all the libraries from this file:
```bash
pip install -r requirements.txt
```

### 5. Download Necessary Data Files
Place the following files in the root of your project directory:
*   `NotoSansSC-Regular.ttf` (for Chinese word clouds)
*   `NotoSansJP-Regular.ttf` (for Japanese word clouds)
*   `cn_stopwords.txt` (a text file with one Chinese stopword per line)
*   `jp_stopwords.txt` (a text file with one Japanese stopword per line)

The first time you run the application, it will automatically download the necessary NLTK data packages (`punkt`, `stopwords`, `wordnet`, etc.).

### 6. Run the Application
```bash
streamlit run app.py
```
Your browser should automatically open a new tab with the dashboard.

---

## 📁 File Structure

The project is organized into modular, single-responsibility files:

*   **`app.py`**: The main Streamlit script that defines the user interface, handles user input, and orchestrates calls to other modules.
*   **`modeling.py`**: Contains the core `run_analysis` function. It handles the entire LDA modeling pipeline, from phrase detection to model training and evaluation.
*   **`preprocessing.py`**: Includes the `clean_and_tokenize` function and its helpers. This file is responsible for all text cleaning, tokenization, and lemmatization.
*   **`visualization.py`**: Contains all functions for generating plots and visuals, such as word clouds, heatmaps, and Plotly charts.
*   **`utils.py`**: A utility module for handling setup tasks like ensuring NLTK resources are downloaded and loading local data files (fonts, stopwords).

---

## 🚀 How to Use the App

1.  **Upload Data**: Click "Browse files" to upload your CSV data. The file must contain columns for a user ID, a timestamp, and the text content.

2.  **Configure Parameters**:
    *   **Main Configuration**: Select the appropriate columns from your CSV for User ID, Text Content, and Date. Choose the language of your text. Set the desired number of topics and the minimum token length. Here you can also enable and configure **Phrase Detection (bigrams)**.
    *   **Vocabulary & Feature Selection**: Fine-tune how the model's vocabulary is created. You can enable **TF-IDF Feature Selection** to automatically select the most important words, or manually set frequency filters (`Min/Max Document Frequency`) to exclude very rare or very common words.
    *   **Text Cleaning**: Choose from a wide range of preprocessing steps, including lemmatization (for English), stopword removal, and how to handle URLs, emojis, hashtags, and mentions. You can also add your own custom stopwords.

3.  **Run Analysis**: Click the "🚀 Run Topic Modeling Analysis" button to start the process. A progress bar will show the current step.

4.  **Interpret the Results**:
    *   **Overall Summary**: Get a quick overview of your dataset (number of users, posts, etc.).
    *   **Model Evaluation**: Assess the quality of your model with three key metrics:
        *   **Coherence (C_v)**: Measures topic quality based on the co-occurrence of words within topics. **Higher is better.**
        *   **Coherence (U_Mass)**: Measures topic quality based on word co-occurrence across the whole dataset. Values are negative; **closer to 0 is better.**
        *   **Perplexity**: A measure of how well the model predicts unseen data. **Lower is better.**
    *   **View & Download Preprocessed Text**: Inspect the tokenized text that was fed into the model and download it for further analysis.
    *   **Topic Visualization**: Explore the discovered topics using either **Word Clouds** or interactive **Top Word Lists**. You can click on words in the lists to add them to your custom stopwords.
    *   **User Analysis**: Dive into user behavior with the interactive scatter plot, per-user donut charts, and topic evolution charts.