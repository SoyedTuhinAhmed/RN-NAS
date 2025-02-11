import torch, itertools, os, json, random, sys
import numpy as np
CROSSSIM_DIR = "/home/sahme175/cross-sim-pytorch"
sys.path.append(CROSSSIM_DIR)
from applications.mvm_params import set_params
from simulator.algorithms.dnn.torch.convert import from_torch, reinitialize
from model import *
import torch.optim as optim

def set_seed(seed):
    torch.use_deterministic_algorithms(True)  # Ensures deterministic behavior
    torch.manual_seed(seed)  # Seed for PyTorch (CPU)
    torch.cuda.manual_seed(seed)  # Seed for PyTorch (single GPU)
    torch.cuda.manual_seed_all(seed)  # Seed for PyTorch (all GPUs, if applicable)
    np.random.seed(seed)  # Seed for NumPy
    random.seed(seed)  # Seed for Python random
    # For compatibility with older PyTorch versions:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def crossSim_simulation(model, config=None, val_loader=None, device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), seed=0, N_runs=20, recalibrate=False):
    set_seed(seed)
    gpu_id = device.index
    if config == None:
        config = set_params(weight_bits = 8, wtmodel = "BALANCED", 
                         error_model = "generic",
                         proportional_error = False,
                         alpha_error = 0.1, useGPU = True, gpu_id = gpu_id)
    analog_model = from_torch(model, config)
    accuracies = []
    # analog_mnist_cnn_pt.to(device)
    for i in range(N_runs):
        acc = validate(analog_model, data_loader=val_loader, device=device, recalibrate=recalibrate)
        accuracies.append(acc)
        reinitialize(analog_model)
    return np.mean(accuracies)

def grid_search_nas(train_loader, val_loader, device, epochs=5, N_runs=10, save_dir='nas_results'):
    """
    Search over the normalization configuration space, evaluate each candidate,
    and save the best model and its configuration.
    """
    # Define the search space.
    search_space = {
        'conv1': ['none', 'batchConv', 'layer', 'instanceConv', 'group2', 'group4', 'group6'],
        'conv2': ['none', 'batchConv', 'layer', 'instanceConv', 'group2', 'group4', 'group6'],
        'fc1':   ['none', 'batchFC', 'layer', 'instanceFC', 'group2', 'group4', 'group6'],
        'fc2':   ['none', 'batchFC', 'layer', 'instanceFC', 'group2', 'group4', 'group6'],
    }
    
    best_config = None
    best_score = -float('inf')
    best_model = None

    # Ensure the save directory exists.
    os.makedirs(save_dir, exist_ok=True)
    
    # Iterate over all configurations in the grid.
    for config_tuple in itertools.product(*search_space.values()):
        print(config_tuple)
        norm_config = dict(zip(search_space.keys(), config_tuple))
        score, model = evaluate_candidate(net=LeNet5_NAS, norm_config=norm_config, 
                                          train_loader=train_loader, val_loader=val_loader, 
                                          fault_injecttion_config = None, N_runs=N_runs,
                                          device=device, epochs=epochs)
        
        candidate_filename = os.path.join(save_dir, f"model_{hash(str(norm_config))}.pth")
        torch.save(model.state_dict(), candidate_filename)
        
        if score > best_score:
            best_score = score
            best_config = norm_config
            best_model = model
    
    # Save the best model and its configuration.
    best_model_path = os.path.join(save_dir, "best_model.pth")
    torch.save(best_model.state_dict(), best_model_path)
    best_config_path = os.path.join(save_dir, "best_config.json")
    with open(best_config_path, 'w') as f:
        json.dump(best_config, f, indent=4)
    
    print("Best normalization configuration found:", best_config)
    print("Best combined score:", best_score)
    print("Best model saved to:", best_model_path)
    print("Best configuration saved to:", best_config_path)
    
    return best_config, best_score, best_model

def evaluate_candidate(net, norm_config, fault_injecttion_config, train_loader, val_loader, alpha = 0.5, epochs= 50, N_runs=10, device=torch.device):
    # Instantiate the network with the given normalization config.
    model = net(norm_config).to(device)
    
    # Train model on clean I.I.D. data
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    torch.use_deterministic_algorithms(False)
    train(model, train_loader, optimizer, criterion, device, epochs=epochs)
    
    # Evaluate on clean validation set to get clean_accuracy
    clean_accuracy = validate(model, data_loader=val_loader, device=device, recalibrate=False)
    
    # Run CrossSim fault injections and compute fault_tolerance_score. calculates mean accuracies on N runs
    FI_accuracy = crossSim_simulation(model, fault_injecttion_config, device=device,
                                      val_loader=val_loader,
                                      seed=2025, N_runs=N_runs, recalibrate=False)
    
    # Combine metrics, weighted sum
    combined_score = alpha * clean_accuracy + (1-alpha) * FI_accuracy
    print('Combined Score: ', combined_score)
    return combined_score, model


def train(model, train_loader, optimizer, criterion, device, epochs=100):
    """
    Train the model for a fixed number of epochs.
    """
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

def validate(model, data_loader, device, recalibrate=False):
    """
    Evaluate the model on a clean validation/test set.
    Returns accuracy.
    """
    if recalibrate:
        model.train()
    else:
        model.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = correct / total
    return accuracy