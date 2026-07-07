This project implements a semantic search engine and visualises key thematic clusters within Jane Austen's classic novel, Pride and Prejudice. 

Features:

## Features
Automated Text Ingestion: Downloads Pride and Prejudice directly from Project Gutenberg and cleans and segments the text into distinct semantic passages.

Semantic Search: Allows the user to query the novel using natural language phrases (such as "a woman looks for a husband") and retrieves the top five most contextually-relevant passages using Cosine Similarity.

2D Theme Visualisation: Maps the entire book's narrative landscape into a 2D space using Principal Component Analysis (PCA).

Thematic Clusters: Identifies and highlights top passages corresponding to four major themes in the book: Marriage, Pride, Money, and Scandal.

## How It Works

Embedding Generation: The text is split into paragraphs and passed through the all-MiniLM-L6-v2 Sentence Transformer, converting text into 384-dimensional dense vector embeddings.

Semantic Search: The query text is embedded into the same vector space. Cosine similarity calculates the "angle" between the original query and every passage in the book, sorting by the highest semantic overlap.

Dimensionality Reduction: Because humans can't visualise 384 dimensions, PCA reduces the embedding vectors to 2 principal components (X and Y coordinates) while preserving as much data variance as possible.

Plotting: Matplotlib then generates a scatter plot where every passage is a light grey dot, overlaying larger, colour-coded markers for the top 10 passages most related to each core theme.

### Success within the model: 

The all-MiniLM-L6-v2 model tends to push general text matches into a tighter band of between 0.20 and 0.50, so a score of over 0.30 on this model signals a highly relevant match. 

For concrete phrases like “Mr Darcy’s estate’, the scores will likely reach 0.70 or 0.80 because the vocabulary is so distinct. But for abstract concepts like feelings or affection, the model spreads the mathematical weight across a variety of synonyms which results in a lower, but still accurate, number. 

Within this model, for the query “a woman searches for a husband”, the top passage is actually the inverse: it talks of a man wanting a wife, in the book’s famous introduction. Basic keyword searches would ignore this concept because the sentence doesn’t actually contain any of the exact words in the query. The semantic search model, however, recognises the phrases are two sides of the same semantic coin: marriage and matchmaking. 
