import time
import torch
import torch.nn as nn

def simple_train(optimizer, train_dataloader, forward, print):
    criterion = nn.CrossEntropyLoss()

    # Training loop

    start_time = time.time()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = forward(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    end_time = time.time()
    epoch_training_time = end_time - start_time

    loss = running_loss / len(train_dataloader)
    accuracy = correct / total

    print(f"Epoch training time: {epoch_training_time}")
    print(f"Loss: {loss}")
    print(f"Accuracy: {100 * accuracy:.2f}%")


    return epoch_training_time

        