from angle_emb import AnglE
import json
import torch
import torch.nn as nn

class GateFunctionAngle():
    '''
        estimate the logits to decide whether to use the augmented knowledge
    '''
    def __init__(self):
        # initiate the text embedding method
        self.angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
        # initiate a multi-layer perceptron to estimate the logits
        self.mlp = nn.Sequential(
            nn.Linear(1024, 768),
            nn.ReLU(),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).cuda()


    def estimate_logits(self, context, knowledge):
        '''
            estimate the logits
        '''
        # encode the context and knowledge
        context_emb = self.angle.encode(context)
        knowledge_emb = self.angle.encode(knowledge)
        
        # estimate the logits based on context and knowledge
        logits = self.mlp(context_emb + knowledge_emb)
        return logits

def train_model(train_data, batch_size=32):
    '''
        train a gating model
    '''
    # initiate the model
    gate_function = GateFunctionAngle()
    # initiate the optimizer
    optimizer = torch.optim.Adam(gate_function.mlp.parameters(), lr=0.001)
    # initiate the loss function
    loss_function = nn.BCEWithLogitsLoss()

    # batch processing the training data
    for i in range(0, len(train_data), batch_size):
        batch = train_data[i:i+batch_size]
        context = batch['context']
        knowledge = batch['knowledge']
        target = batch['logits']
        
        # convert the target to float
        target = [float(t) for t in target]

        # train the model
        for i in range(100):
            # estimate the logits
            logits = gate_function.estimate_logits(context, knowledge)
            # calculate the loss
            loss = loss_function(logits, torch.tensor(target).unsqueeze(1).cuda())
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("loss: ", loss)
    
    # save the model
    torch.save(gate_function.mlp.state_dict(), 'gate_function_angle.pt')

