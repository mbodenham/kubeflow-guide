from kfp import dsl
from kfp import compiler
from typing import NamedTuple

@dsl.component(
    base_image='nvcr.io/nvidia/pytorch:26.01-py3',
    # Explicitly requesting torchvision in case it is absent from the base image
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
    
    # 1. IMPORTS MUST BE INSIDE THE COMPONENT
    # Because this function executes inside its own isolated Docker container,
    # all imports must be scoped locally within the function itself.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.optim.lr_scheduler import StepLR

    # 2. CLASSES AND FUNCTIONS MUST BE INSIDE THE COMPONENT
    # For the same reason as imports, any helper methods or classes must be
    # defined within the boundaries of the component function.
    class Net(nn.Module):
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
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    def test(model, device, test_loader):
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

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * test_accuracy))
        return test_loss, test_accuracy

    # 3. EXECUTION LOGIC
    # Below is standard PyTorch training logic. Since it runs in the component's container,
    # it can use accelerators if the Kubernetes node provides them.
    # Adjusted accelerator logic to standard CUDA check for broader compatibility
    use_accel = not no_accel and torch.cuda.is_available()

    torch.manual_seed(seed)
    device = torch.device("cuda" if use_accel else "cpu")
    print("GPU:", torch.cuda.get_device_name(0))

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}
    if use_accel:
        accel_kwargs = {'num_workers': 1,
                        'persistent_workers': True,
                        'pin_memory': True,
                        'shuffle': True}
        train_kwargs.update(accel_kwargs)
        test_kwargs.update(accel_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Downloads dataset to the container's ephemeral storage
    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    test_loss, test_accuracy = 0.0, 0.0

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, log_interval)
        test_loss, test_accuracy = test(model, device, test_loader)
        scheduler.step()

    # Warning: This file will be deleted when the pipeline task finishes.
    if save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    return (float(test_loss), float(test_accuracy))

@dsl.pipeline(
    name='mnist-pipeline',
    description='Train and test CNN model on MNIST dataset.'
)
# Pipeline parameters can be specified via the Kubeflow UI when creating a Run!
def mnist_pipeline(batch_size: int = 64, 
                   test_batch_size: int = 1000,
                   epochs: int = 14,
                   lr: float = 1.0,):
    # 4. INSTANTIATE THE TASK (Do not unpack outputs)
    # Tasks are created by calling the component function.
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

    # 5. RESOURCE REQUESTS AND LIMITS
    # KFP enables fine-grained control over the Kubernetes Pod executing this task.
    training_task.set_cpu_limit('4')       # Request 4 CPU cores
    training_task.set_memory_limit('8G')   # Request 8 GB of Memory

    # 6. EXPLICIT GPU ALLOCATION
    # Request 1 GPU for the container 
    training_task.set_accelerator_limit(1)


    # 7. NODE SELECTOR CONSTRAINTS
    # Ensure the Kubernetes scheduler places the pod on a node with an NVIDIA GPU.
    # The label below is the standard for most Kubernetes/Kubeflow setups.
    training_task.add_node_selector_constraint('nvidia.com/gpu')
    
    # If adding a downstream component later, pass variables like this:
    # my_evaluator_component(loss=training_task.outputs['test_loss'])

if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=mnist_pipeline,
        package_path='compiled_mnist_pipeline.yaml'
    )
    print("Saved to 'compiled_mnist_pipeline.yaml'.")