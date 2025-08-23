# Reading notes for the paper

### Model architectures

 - CBOW - Continuous Bag-of-Words Model
    - predicts word w[t] based on w[t-1], ..., w[t-C] and w[t+1], ..., w[t+C]

- Skip-gram - Continuous Skip-gram Model
    - Words w[t-1], ..., w[t-C] and w[t+1], ..., w[t+C] are predicted based on w[t]


### Training complexity Q

How much work (number of operations) does the model have to do for one training example?

(total) Complexity O = ExTxQ where 
E number of epochs, T number of trianing words (tokens), Q training complexity (cost per training example).

### Tokenization 
In the word2vec (2013) tokenizztion is primitive, using lower case words as the tokens. 
Importqant to keep only words - no punctation etc. 
Drop rare words (e.g. first nouns and concatinated long words) to make training more efficient. 

Later research has made modern LLM's use subword tokenization (Byte pair encoding or SentencePiece algorithms) because it reduces vocab size (makes the embedding matrices much smaller) and handles rare words better, in addition to handling punctuation.

### Subsampling (Mikolov's trick)
Very frequent words like the, and, of appear very frequently, but they add little training signal (too generic) even though they dominate training. 
Mikolov's trick is randomly discarding frequent words with a probability defined based on their frequency. 

P_\text{discard}(w) = 1 - \sqrt{\frac{t}{f(w)}}

where
- f(w) = frequency of word w / total tokens
- t = threshold (usually 10^{-5})

This balances the corpus (large, structured collection of texts), and reduces training time by 2-10x. 

### Negative sampling (for Skip-gram) (Loss function)
A trick to make training more efficient - instead of computing softmax over the entire vocab for the loss function, you can train against only a few negative examples. 

" For each positive pair (cat, sat) you sample, say, 5 random “noise” words (banana, car, philosophy, …) and tell the model "cat should not predict these." "

\mathcal{L} = -\log \sigma(v_\text{center} \cdot v_\text{context})
- \sum_{k=1}^K \log \sigma(-v_\text{center} \cdot v_{\text{neg},k})

This trains the model to assosiate each center word to the surrounding context words, and NOT assosiate the center word with the random (negative) words. 
Forces words with similar usage to have similar embeddings. 



### Why two embedding matrices in the training?

In other words - why two embeddings for each word based on if its the center or a context word?
"""
Why two matrices help
	•	If “cat” is the center, we only touch in[cat].
	•	If “cat” is the context, we only touch out[cat].
	•	The two representations can specialize to their roles.
"""
such that words being used as negative samples for other words will not affect their embeddings (in / center) that will be used in inference. 
