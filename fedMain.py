from aggregate import get_state_dicts, fed_avg
from client.clientSrc.train import *
import torch

# Initialize the global model
global_model = Model()
global_round = 10

# Get client's info


# Split data among clients

# Broadcast global model to clients and start Federated Learning
for round in range(0, global_round):
    # Send global model to clients and wait

    # Get local models' state dict files from clients

    # Get local model's state dicts from files
    state_dicts = get_state_dicts('client/clientModelOutputs/', round)

    # Aggregate the weights, biases by FedAvg algorithm
    model_state_dict = fed_avg(state_dicts)

    # Update the global model with the aggregated weights, biases
    global_model.load_state_dict(model_state_dict)

    # Save the global model and state_
    torch.save(global_model, f'globalModelOutputs/model_{round}.pickle')
