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

