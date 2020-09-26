# Hate Tweets Detection

This repository contains different models compaired in a large hate tweets [database](https://www.kaggle.com/kazanova/sentiment140) from Kaggle  
It makes comparison of:
1. LSTM
2. BERT
3. GPT-2

### Table of contents
- LSTM
- BERT finetune
- GPT-2  

**Note:** Some preprocessing is needed and it is performed in the LSTM script(so I you replicate the experiment run it first)  
The other 2 scripts assume that the input is the preprocessed file. Also because the original dataset contains 1.6 million tweets  
so a smaller dataset is created with 10000 tweets for fast experimentation. 

Also the generative possibilities of the GPT-2 are explored.  So you can generate human-like text after the model is trained with the whole dataset.  
The interesting thing is that some generated tweets are hatefull like the original's and some are neutral or not hatefull.


