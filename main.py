from runner.runner import run_probes
from data.mutag_dataset import load_mutag_dataset
from models.gin_model import GIN
from plotting.angle_matrix import plot_probe_direction_angles
from plotting.plot import plot_probe_directions
import torch

dataset, train_dataset, train_loader, test_loader, full_loader = load_mutag_dataset()

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_features = dataset.num_features
    num_classes = dataset.num_classes
    hidden_dims = [6, 2]
    model = GIN(num_features, hidden_dims, num_classes).to(device)


    probe_directions = run_probes(model = model,
                        train_dataset = train_dataset, 
                        full_loader = full_loader, 
                        test_loader = test_loader, 
                        train_loader= train_loader, 
                        desired_layer = 2, 
                        num_epochs = 50, 
                        dataset=dataset)
    
    plot_probe_direction_angles(probe_directions)

    plot_probe_directions(probe_directions)


