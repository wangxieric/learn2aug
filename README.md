# learn2aug


### Stage 1: Augment SGD data with KETOD (human labelled knowledge augmentation)

KETOD offers the label of each turn about if a enrichment is required. Therefore, for the first stage, we augment KETOD to the SGD data with the labels as well as the enriched responses.

The above implementation can be conducted by running: 
```
python gen_full_ketod.py
```
Running the above data processing, we extend the SGD to further labelled data and saved to data/KETOD/ketod_release

### Stage 2: Train knowledge ranking module (e.g. BERT) or deploy TF-IDF

We also explore the use of multiple ranking strategies, such as the static methods, like TF-IDF or learned ranker, like BERT. Before ranking, we need to further process the data to prepare the context for accurate knowledge retrieval with:

```
cd kg_select
python process_data.py
```

An example context is as follow: 

```
"context": "<|context|> <|user|> Would you help me search for a bus in Anaheim, CA? <|system|> When do you want to leave? <|user|> I want to leave on the 4th. <|system|> Where will you leave from? <|user|> I will leave from San Diego. <|system|> What about a bus leaving at 10:10 am that costs $23? There are 0 transfers. <|user|> Sounds good to me. <|system|> Would you like to buy tickets? <|user|> Yes, please reserve tickets for me. <|system|> How many tickets would you like? <|user|> The tickets are for four people. <|system|> Could you confirm that you want to leave from San Diego to Anaheim on March 4th at 10:10 am for 4? <|user|> Works for me. Which bus station will I leave from? <|endofcontext|>"
```



