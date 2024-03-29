import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import tqdm
import numpy as np
import utils
import dataloaders
import torchvision
from trainer import Trainer

torch.random.manual_seed(0)
np.random.seed(0)


# Load the dataset and print some stats
batch_size = 64

image_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
    ]
)

dataloader_train, dataloader_test = dataloaders.load_dataset(
    batch_size, image_transform
)
example_images, _ = next(iter(dataloader_train))
print(
    f"The tensor containing the images has shape: {example_images.shape} (batch size, number of color channels, height, width)",
    f"The maximum value in the image is {example_images.max()}, minimum: {example_images.min()}",
    sep="\n\t",
)


def create_model():
    """
    Initializes the mode. Edit the code below if you would like to change the model.
    """
    model = nn.Sequential(
        nn.Flatten(),  # Flattens the image from shape (batch_size, C, Height, width) to (batch_size, C*height*width)
        nn.Linear(28 * 28 * 1, 10)
        # No need to include softmax, as this is already combined in the loss function
    )
    # Transfer model to GPU memory if a GPU is available
    model = utils.to_cuda(model)
    return model


model = create_model()


# Test if the model is able to do a single forward pass
example_images = utils.to_cuda(example_images)
output = model(example_images)
print("Output shape:", output.shape)
expected_shape = (batch_size, 10)  # 10 since mnist has 10 different classes
assert (
    output.shape == expected_shape
), f"Expected shape: {expected_shape}, but got: {output.shape}"


# Hyperparameters
learning_rate = 0.0192
num_epochs = 5


# Use CrossEntropyLoss for multi-class classification
loss_function = torch.nn.CrossEntropyLoss()

# Define optimizer (Stochastic Gradient Descent)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


trainer = Trainer(
    model=model,
    dataloader_train=dataloader_train,
    dataloader_test=dataloader_test,
    batch_size=batch_size,
    loss_function=loss_function,
    optimizer=optimizer,
)
train_loss_dict, test_loss_dict = trainer.train(num_epochs)

final_loss, final_acc = utils.compute_loss_and_accuracy(
    dataloader_test, model, loss_function
)
print(f"Final Test loss: {final_loss}. Final Test accuracy: {final_acc}")

# We reset the manual seed to 0, such that the model parameters are initialized with the same random number generator.
torch.random.manual_seed(0)
np.random.seed(0)

image_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=0.5, std=0.5),
    ]
)

dataloader_train, dataloader_test = dataloaders.load_dataset(
    batch_size, image_transform
)
model = create_model()

# Redefine optimizer, as we have a new model.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
trainer = Trainer(
    model=model,
    dataloader_train=dataloader_train,
    dataloader_test=dataloader_test,
    batch_size=batch_size,
    loss_function=loss_function,
    optimizer=optimizer,
)
train_loss_dict_normalized, test_loss_dict_normalized = trainer.train(num_epochs)

# We reset the manual seed to 0, such that the model parameters are initialized with the same random number generator.
torch.random.manual_seed(0)
np.random.seed(0)

dataloader_train, dataloader_test = dataloaders.load_dataset(
    batch_size, image_transform
)
model = utils.to_cuda(
    nn.Sequential(
        nn.Flatten(),  # Flattens the image from shape (batch_size, C, Height, width) to (batch_size, C*height*width)
        nn.Linear(28 * 28 * 1, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )
)

# Redefine optimizer, as we have a new model.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
trainer = Trainer(
    model=model,
    dataloader_train=dataloader_train,
    dataloader_test=dataloader_test,
    batch_size=batch_size,
    loss_function=loss_function,
    optimizer=optimizer,
)
train_loss_dict_hidden, test_loss_dict_hidden = trainer.train(num_epochs)

# Plot loss
utils.plot_loss(train_loss_dict_hidden, label="Train Loss - Hidden layer")
utils.plot_loss(test_loss_dict_hidden, label="Test Loss - Hidden layer")
utils.plot_loss(train_loss_dict_normalized, label="Train Loss - Normalized images")
utils.plot_loss(test_loss_dict_normalized, label="Test Loss - Normalized images")
utils.plot_loss(train_loss_dict, label="Train Loss - Original model")
utils.plot_loss(test_loss_dict, label="Test Loss - Original model")

# Limit the y-axis of the plot (The range should not be increased!)
plt.ylim([0, 1])
plt.legend()
plt.xlabel("Global Training Step")
plt.ylabel("Cross Entropy Loss")
plt.savefig("image_solutions/task_4c.png")

plt.show()

torch.save(model.state_dict(), "saved_model.torch")
final_loss, final_acc = utils.compute_loss_and_accuracy(
    dataloader_test, model, loss_function
)
print(f"Final Test loss: {final_loss}. Final Test accuracy: {final_acc}")
