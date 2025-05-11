import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from models.linear_probe import LinearProbe
from scripts.train_probe import train_probe, evaluate_probe
from scripts.train_gin import train_gin, test_gin
from configs.concept_configs import concept_configs
from concepts.create_mask import concatenate_masks


def run_probes(model, train_dataset, full_loader, test_loader, train_loader, desired_layer, num_epochs, dataset):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(1, num_epochs+1):

        loss = train_gin(model, train_loader, optimizer, criterion, device, train_dataset)
        acc = test_gin(model, full_loader, device, return_hidden=False, layer=desired_layer)  # Evaluating on test set+
        if epoch % 10 == 0:
            print(f"Epoch {epoch:02d}: Loss {loss:.4f}, Test Acc {acc:.4f}")

    probe_directions=[]

    # Hidden embeddings (assumed to be shared across concepts)
    __, hidden_embeddings, weights = test_gin(model, full_loader, device, return_hidden=True, layer=desired_layer)
    print(f'Hidden_embeddings size: {hidden_embeddings.shape}')
    embeddings_np = hidden_embeddings.cpu().detach().numpy()
    print(embeddings_np)

    # Loop over each concept configuration
    for config in concept_configs:
        # Set the global concept variables for this configuration.
        concepts = config["concepts"]
        combination_operator = config["combination_operator"]

        # Generate the feature mask (this uses your existing concatenate_masks() function).
        feature_mask = concatenate_masks(concepts, combination_operator)
        feature_mask = torch.tensor(feature_mask)
        print("\n====================================================")
        print(f"Processing concept with combination operator: {combination_operator}")
        print(f'Hidden_embeddings size: {hidden_embeddings.shape}')
        print(f'Feature mask size: {feature_mask.shape}')

        # Create a TensorDataset and DataLoader for batching.
        dataset = TensorDataset(hidden_embeddings, feature_mask)
        probe_loader = DataLoader(dataset, batch_size=64, shuffle=True)

        # Set up the probe, optimizer, and criterion.
        input_dim = hidden_embeddings.size(1)
        probe = LinearProbe(input_dim).to(device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=0.01)
        criterion = nn.BCEWithLogitsLoss()

        num_epochs = 10
        print("=== Training Linear Probe ===")
        for epoch in range(1, num_epochs + 1):
            loss = train_probe(probe, probe_loader, optimizer, criterion, device)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{num_epochs} => Probe Loss: {loss:.4f}")

        avg_loss, accuracy, probe_direction = evaluate_probe(probe, probe_loader, device, criterion)
        print(f"\n=== Linear Probe Evaluation ===")
        print(f"Probe Loss for {config['name']}: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%")
        print(f"Probe Direction: {probe_direction}")
        probe_directions.append(probe_direction)

    print(f'Probe directions: {probe_directions}')

    return probe_directions, weights