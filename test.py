from data import get_dataloaders
import torchvision.models as models
import os
import torch
import numpy as np
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# First, we define a local drift variable hi for each client. In an ideal condition, the local drift variable should satisfy the restriction: hi = w − θi
# Define the objective function for each client
# F (θi; hi, Di, w) = Li(θi)+ (α / 2) * Ri(θi; wi, w)+Gi(θi; gi, g)
# where Li(θi) is the local empirical loss term, Ri(θi; wi, w) is the penalized term, and Gi(θi; gi, g) is the gradient correction term
# Gi(θi; gi, g) = (1 /ηK) * ⟨θi, gi − g⟩, where η is the learning rate, K is the amount of training iterations in one round. gi is the local update value of i-th client’s local parameters in last round, g is the average update value of all clients’ local parameters in last round    
# Define the objective function for each client
def objective_function(model, criterion, local_data, local_weights, global_weights, alpha_coef, local_grad, global_grad, learning_rate):
    # Calculate local empirical loss term
    local_loss = criterion(model(local_data), local_weights)
    # Calculate penalized term
    penalized_term = (torch.norm(local_weights - global_weights)**2) / 2
    # Calculate gradient correction term
    gradient_correction_term = torch.dot(local_grad.flatten(), (local_weights - global_weights).flatten()) / (learning_rate * len(local_data))
    # Calculate objective function
    objective = local_loss + alpha_coef * penalized_term + gradient_correction_term
    # Calculate gradient of objective function
    objective.backward()
    # Update local gradient
    local_grad += learning_rate * local_weights.grad
    # Update global gradient
    global_grad += local_grad
    # Return objective function value and updated local and global gradients
    return objective.item(), local_grad, global_grad


# Define function for training
def train(model, train_dataloaders, test_dataloaders, n_client, com_amount, save_period, weight_decay, batch_size, act_prob, suffix, lr_decay_per_round, epoch, alpha_coef, learning_rate, print_per):
    # Set model to training mode
    model.train()
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    # Define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Initialize global weights
    global_weights = torch.zeros_like(torch.cat([x.view(-1) for x in model.parameters()]))
    # Initialize global gradient
    global_grad = torch.zeros_like(torch.cat([x.view(-1) for x in model.parameters()]))
    # Initialize list to store test accuracies
    test_accs = []
    # Initialize list to store communication rounds
    com_rounds = []
    # Initialize list to store objective function values
    objective_values = []
    # Initialize list to store learning rates
    learning_rates = []
    # Initialize list to store local gradients
    local_grads = [torch.zeros_like(torch.cat([x.view(-1) for x in model.parameters()])) for _ in range(n_client)]
    # Initialize list to store local weights
    local_weights = [torch.cat([x.view(-1) for x in model.parameters()]) for _ in range(n_client)]
    
    # Train for com_amount communication rounds
    for com_round in range(com_amount):
        # Randomly select active clients
        active_clients = np.random.choice(range(n_client), size=int(n_client * act_prob), replace=False)
        # Initialize list to store objective function values for each client
        objective_values_per_client = []
        # Train on each active client
        for client_idx in active_clients:
            # Set model to training mode
            model.train()
            # Get local data and labels
            local_data, local_labels = next(iter(train_dataloaders[client_idx]))
            # Send local data and labels to device
            local_data, local_labels = local_data.to(device), local_labels.to(device)
            # Zero out gradients
            optimizer.zero_grad()
            # Calculate objective function and update local and global gradients
            objective_value, local_grad, global_grad = objective_function(model, criterion, local_data, local_weights[client_idx], global_weights, alpha_coef, local_grads[client_idx], global_grad, learning_rate)
            # Save objective function value
            objective_values_per_client.append(objective_value)
            # Update local weights
            local_weights[client_idx] -= learning_rate * local_grad
        # Update global weights
        global_weights -= (learning_rate / len(active_clients)) * global_grad
        # Decay learning rate
        learning_rate *= lr_decay_per_round
        # Save model every save_period communication rounds
        if (com_round + 1) % save_period == 0:
            torch.save(model, f"./Folder/Model/{suffix}_round{com_round+1}.pt")
        # Evaluate test accuracy every communication round
        if (com_round + 1) % 1 == 0:
            # Set model to evaluation mode
            model.eval()
            # Initialize list to store predictions and labels
            preds, labels = [], []
            # Evaluate on each test dataset
            for test_dataloader in test_dataloaders:
                # Iterate over test dataset
                for data, target in test_dataloader:
                    # Send data and target to device
                    data, target = data.to(device), target.to(device)
                    # Get model prediction
                    output = model(data)
                    # Get predicted class
                    pred = output.argmax(dim=1, keepdim=True)

                    # Continue with evaluating test accuracy
                    # Append predictions and labels to lists
                    preds += pred.cpu().numpy().tolist()
                    labels += target.cpu().numpy().tolist()
            # Calculate test accuracy
            test_acc = (np.array(preds) == np.array(labels)).mean()
            # Print test accuracy
            print(f"Communication round {com_round+1}, Test accuracy: {test_acc}")
            # Append test accuracy to list
            test_accs.append(test_acc)
            # Append communication round to list
            com_rounds.append(com_round+1)
            # Append objective function value to list
            objective_values.append(np.mean(objective_values_per_client))
            # Append learning rate to list
            learning_rates.append(learning_rate)
        # Print progress every print_per epochs
        if (com_round + 1) % (print_per * len(train_dataloaders[0])) == 0:
            print(f"Communication round {com_round+1}, Objective function value: {np.mean(objective_values_per_client)}")
            # Evaluate test accuracy
            model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in test_dataloader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    test_loss += criterion(output, target).item() # sum up batch loss
                    pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_dataloader.dataset)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_dataloader.dataset),
                100. * correct / len(test_dataloader.dataset)))


batch_size = 32
train_dataloaders, test_dataloaders = get_dataloaders("CIFAR10_iid", batch_size, shuffle=True)

# Check if model exists in ./Folder/Model directory
if os.path.exists("./Folder/Model/model.pt"):
    # Load existing model
    model = torch.load("./Folder/Model/model.pt")
else:
    # Create ResNet18 model
    model = models.resnet18()


n_client = 20

# Define model name
model_name = 'Resnet18'

# Define common hyperparameters
com_amount = 100  # Number of communication rounds
save_period = 50  # Save model every 50 rounds
weight_decay = 1e-3  # Weight decay for optimizer
batch_size = 50  # Batch size for training
act_prob = 0.15  # Probability of activation for each client
suffix = model_name  # Suffix for saved model name
lr_decay_per_round = 0.998  # Learning rate decay per round


# Define training hyperparameters
epoch = 5  # Number of epochs for training
alpha_coef = 1e-2  # Coefficient for alpha in FedProx
learning_rate = 0.1  # Learning rate for optimizer
print_per = epoch // 2  # Print progress every half epoch


train(model, train_dataloaders, test_dataloaders, n_client, com_amount, save_period, weight_decay, batch_size, act_prob, suffix, lr_decay_per_round, epoch, alpha_coef, learning_rate, print_per)

