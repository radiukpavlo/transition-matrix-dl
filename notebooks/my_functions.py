import numpy as np
import torch


def generate_sample_dataset(input_dataset, input_n_samples):
    """
    This function creates a new dataset subset with the
    specified number of samples.
    :param input_n_samples:
    :param input_dataset: Original training dataset
    :return: Subset of the training dataset
            and a list of unique IDs (indices from the original dataset)
    """

    # Parameters
    # n_samples = 10000

    generate_sample_labels = input_dataset.targets.numpy()

    # Determine the proportion of each class in the training dataset
    _, counts = np.unique(generate_sample_labels, return_counts=True)
    proportions = counts / len(generate_sample_labels)

    # Determine the number of samples to extract for each class
    samples_per_class = (proportions * input_n_samples).astype(int)

    # Adjust samples for any rounding issues to ensure exactly 1000 samples
    while np.sum(samples_per_class) < input_n_samples:
        class_with_max_samples = np.argmax(proportions)
        samples_per_class[class_with_max_samples] += 1

    # Extract samples based on the proportions
    indices_to_extract = []

    for gs_label, n_samples in enumerate(samples_per_class):
        label_indices = np.where(generate_sample_labels == gs_label)[0]
        chosen_indices = np.random.choice(label_indices, n_samples, replace=False)
        indices_to_extract.extend(chosen_indices)

    # Shuffle the indices for randomness
    np.random.shuffle(indices_to_extract)

    # Return a subset of the dataset
    sample_dataset = torch.utils.data.Subset(input_dataset, indices_to_extract)

    return sample_dataset, indices_to_extract


'''def generate_prediction_weights(input_dataset, input_weights_path):
    """
    This function extracts the weights from the model in the prediction stage
    :param input_weights_path: 
    :param input_dataset:
    :return:
    """
    total_images = len(input_dataset)

    my_label_true = []
    my_pred_flattened = []
    my_pred_logits = []
    my_label_preds = []

    sample_model = MNISTModel(input_shape=1, hidden_units=10, output_shape=10).to(device)

    sample_model.load_state_dict(torch.load(os.path.join(input_weights_path,
                                                         f"model_epoch_{N_EPOCHS}.pth")))  # Load your trained weights
    model.eval()  # Set the model to evaluation mode

    # flattened_outputs = []

    with torch.inference_mode():
        for batch, (x_sample, y_sample) in enumerate(my_sample_dataset_train):  # we don't need labels for prediction
            x_sample = x_sample.to(device)
            preds, flattened = sample_model(x_sample.unsqueeze(dim=0).to(device), return_flattened=True)

            # Save the true labels for each image
            my_label_true.append(y_sample)

            # Save the activations after the nn.Flatten() layer for each image
            my_pred_flattened.append(flattened)

            # Save the prediction logits
            my_pred_logits.append(preds)

            # Save the prediction labels
            model_pred_labels = torch.argmax(torch.softmax(preds, dim=1), dim=1)
            my_label_preds.append(model_pred_labels)

            # At this point, `flattened_outputs` is a list of tensors, 
            # where each tensor corresponds to the flattened outputs for a batch of data.

        my_pred_flattened  = torch.cat(my_pred_flattened, dim=0).cpu()  # Concatenate all activations
        my_pred_logits = torch.cat(my_pred_logits).cpu()
        my_label_preds = torch.cat(my_label_preds).cpu()

    for item in range(total_images):
        # Get image and labels from the test data
        input_image = input_dataset[item][0]
        input_label = input_dataset[item][1]
        my_label_true.append(input_label)

        # Extract and save the weights after the nn.Flatten() layer
        model_pred_flattened = model.get_flatten
        # model_pred_flattened = model(input_image.unsqueeze(dim=0).get_flatten.to(device))
        my_pred_flattened.append(model_pred_flattened)
        print(f"The flattened weights:\n {my_pred_flattened}")
        # print(f"The shape of the tensor:\n {my_pred_flattened.shape}")

        # Save the prediction weights
        model_pred_logits = model_gpu(input_image.unsqueeze(dim=0).to(device))
        my_pred_logits.append(model_pred_logits)

        # Save the prediction labels
        model_pred_labels = torch.argmax(torch.softmax(model_pred_logits, dim=1), dim=1)
        my_label_preds.append(model_pred_labels)

    my_pred_flattened = torch.cat(my_pred_flattened).cpu()
    my_pred_logits = torch.cat(my_pred_logits).cpu()
    my_label_preds = torch.cat(my_label_preds).cpu()

    return my_pred_flattened, my_pred_logits, my_label_preds, my_label_true
'''

'''model = MNISTModel(input_shape=1, hidden_units=10, output_shape=10).to(device)

model_weights_path = "../model_weights/"

model.load_state_dict(torch.load(os.path.join(model_weights_path, 
                                              f"model_epoch_{N_EPOCHS}.pth")))  # Load your trained weights
model.eval()  # Set the model to evaluation mode

flattened_outputs = []

with torch.inference_mode():
    for batch, (X_sample, _) in enumerate(my_sample_dataset_train):  # we don't need labels for prediction
        X_sample = X_sample.to(device)
        preds, flattened = model(X_sample.unsqueeze(dim=0).to(device), return_flattened=True)
        flattened_outputs.append(flattened)

        # At this point, `flattened_outputs` is a list of tensors, 
        # where each tensor corresponds to the flattened outputs for a batch of data.

    all_flattened_outputs  = torch.cat(flattened_outputs, dim=0)  # Concatenate all activations
'''