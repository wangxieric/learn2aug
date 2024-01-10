from xturing.models import BaseModel
from xturing.datasets.instruction_dataset import InstructionDataset
import gc

# instruction_dataset = InstructionDataset("/home/xwang/learn2aug/data/SimpleTOD/outputs/inference_only_model1_gold_action_retrieved_kg_gold_decision/results/dataset_for_gating_model")

# # initialise the model
# model = BaseModel.create("llama_lora_int8")

# model.finetune(dataset=instruction_dataset)

# model.save('/home/xwang/learn2aug/aug_gate/trained_gating_models/llama_weights')

# del model
gc.collect()
model = BaseModel.load("/home/xwang/learn2aug/aug_gate/trained_gating_models/llama_weights")
output = model.generate(texts=["Estimate the lowest logit of generating a response to the input text. The estimation examines the previous utterances, estimated attributes and especially the contribution of the augmented knowledge: Input: <|context|> <|user|> I'd like to look for several films online. <|endofcontext|> <|belief|> <|endofbelief|> <|dbresults|> <|nonewdb|> <|endofdbresults|> <|action|> media request genre <|endofaction|> <|knowledge|> Labyrinth (film) Labyrinth is a 1986 British-American adventure musical fantasy film directed by Jim Henson, executive-produced by George Lucas, and based upon conceptual designs by Brian Froud. The Lord of the Rings The Lord of the Rings is an epic high-fantasy novel written by English author J. R. R. Tolkien. The Lord of the Rings Written in stages between 1937 and 1949, \"The Lord of the Rings\" is one of the best-selling novels ever written, with over 150 million copies sold. <|endofknowledge|> <|nochitchat|> <|endofdecision|> <|response|> What kind of movies? <|endofresponse|>"])

print("generated output by the model: ", output)

"""
    Result:
    generated output by the model:  [' <|section|> <|sectiontitle|> Media Requests <|endofsection|> <|section|> <|endofsection|> <|post|> <|endofpost|> <|response|> What kind of movies? <|endofresponse|> <|response|> What kind of movies? <|endofresponse|> <|response|> What kind of movies? <|endofresponse|> <|response|> What kind of movies? <|endofresponse|> <|response|> What kind of movies? <|endofresponse|> <|response|> What kind of movies? <|endofresponse|> <|response|> What kind of movies? <|endofresponse|> <|response|> What kind of movies? <|endofresponse|> <|response|> What kind of movies? <|endofresponse|> <|response|> What kind of movies? <|endofresponse|> <|response|> What kind of movies? <|endofresponse|> <|response|> What kind of movies? <|']
"""

"""
    Result:
    generated output by the model:  [' 9.29000']
"""