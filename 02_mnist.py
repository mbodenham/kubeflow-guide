from kfp import dsl
from kfp import compiler
from typing import NamedTuple

# =============================================================================
# 🧠 COMPONENT: MNIST Training & Evaluation
# =============================================================================

@dsl.component(
    base_image='nvcr.io/nvidia/pytorch:26.01-py3',
    packages_to_install=['torchvision']
)
def train_test(
    batch_size: int,
    test_batch_size: int,
    epochs: int,
    lr: float,
    gamma: float,
    no_accel: bool,
    dry_run: bool,
    seed: int,
    log_interval: int,
    save_model: bool
) -> NamedTuple('Outputs', [('test_loss', float), ('test_accuracy', float)]):
    """Train a CNN on MNIST and evaluate test performance.
    
    All imports, classes, and functions must be defined inside this component
    because it executes in an isolated Docker container.
    
    Args:
        batch_size: Training batch size
        test_batch_size: Testing batch size
        epochs: Number of training epochs
        lr: Learning rate
        gamma: LR scheduler gamma
        no_accel: Disable GPU acceleration
        dry_run: Quick test mode
        seed: Random seed for reproducibility
        log_interval: Logging frequency
        save_model: Save model weights
        
    Returns:
        NamedTuple with test_loss and test_accuracy
    """

    # ── IMPORTS ─────────────────────────────────────────────────────────────
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.optim.lr_scheduler import StepLR

    # ── MODEL DEFINITION ────────────────────────────────────────────────────
    class Net(nn.Module):
        """Simple CNN for MNIST classification."""
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

    def train(model, device, train_loader, optimizer, epoch, log_interval):
        """Training loop."""
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    def test(model, device, test_loader):
        """Evaluation loop."""
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        test_accuracy = correct / len(test_loader.dataset)
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
              f'({100. * test_accuracy:.0f}%)\n')
        return test_loss, test_accuracy

    # ── EXECUTION ───────────────────────────────────────────────────────────
    use_accel = not no_accel and torch.cuda.is_available()
    torch.manual_seed(seed)
    device = torch.device("cuda" if use_accel else "cpu")
    print(f"Using device: {torch.cuda.get_device_name(0) if use_accel else 'CPU'}")

    # Data loading configuration
    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}
    if use_accel:
        train_kwargs.update({'num_workers': 1, 'persistent_workers': True, 'pin_memory': True, 'shuffle': True})
        test_kwargs.update({'num_workers': 1, 'persistent_workers': True, 'pin_memory': True, 'shuffle': True})

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download datasets
    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # Train model
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, log_interval)
        test_loss, test_accuracy = test(model, device, test_loader)
        scheduler.step()

    # Save model if requested
    if save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    return (float(test_loss), float(test_accuracy))

# =============================================================================
# 📐 PIPELINE DEFINITION
# =============================================================================

@dsl.pipeline(
    name='mnist-pipeline',
    description='Train and test CNN model on MNIST dataset.'
)
def mnist_pipeline(
    batch_size: int = 64,
    test_batch_size: int = 1000,
    epochs: int = 14,
    lr: float = 1.0,
):
    """MNIST training pipeline with GPU support."""
    training_task = train_test(
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        epochs=epochs,
        lr=lr,
        gamma=0.7,
        no_accel=False,
        dry_run=False,
        seed=1,
        log_interval=10,
        save_model=False
    )

    # Configure resources
    training_task.set_cpu_limit('4')
    training_task.set_memory_limit('8G')
    training_task.set_accelerator_limit(1)
    training_task.add_node_selector_constraint('nvidia.com/gpu')

# =============================================================================
# ⚙️ COMPILATION
# =============================================================================

if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=mnist_pipeline,
        package_path='compiled_mnist_pipeline.yaml'
    )
    print("✓ Saved to 'compiled_mnist_pipeline.yaml'.")
