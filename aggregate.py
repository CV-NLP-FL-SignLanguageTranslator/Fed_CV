import torch


def get_state_dicts(models_dir):
    """
    Returns each weight per model of the model files in the given directory address.
    :param models_dir: directory address of the models

    :return: client's state_dicts containing the weights and biases
    """
    state_dicts = []
    clients_id = [1, 2, 3, 4]
    # model = Model()

    for client in clients_id:
        uploaded_model_state_dict = torch.load(models_dir + f'model_{client}.pickle')
        model_state_dict = uploaded_model_state_dict.state_dict()
        state_dicts.append(model_state_dict)

    return state_dicts


def fed_avg(state_dicts):
    """
    Returns the average of the model's weights trained by the clients.
    :param state_dicts: list of models' state_dict

    :return: global model's state_dict
    """

    # Load Initial model's state_dict
    model_state_dict = Model().state_dict()

    # Initialize the weights of the global model
    for key in model_state_dict:
        model_state_dict[key] = torch.zeros_like(model_state_dict[key])

    # Sum all the weights of the clients and divide by the number of clients
    for key in model_state_dict:
        for state_dict in state_dicts:
            model_state_dict[key] += state_dict[key]

        model_state_dict[key] = torch.div(model_state_dict[key], len(state_dicts))

    return model_state_dict

