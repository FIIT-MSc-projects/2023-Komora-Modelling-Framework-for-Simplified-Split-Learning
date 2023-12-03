import torch
import torch.nn as nn

def simple_evaluate(test_dataloader, output_function, print):
    criterion = nn.CrossEntropyLoss()

    correct = 0
    total = 0
    running_loss = 0.0

    # since we're not training, we don't need to calculate the gradients for our outputs
    
    with torch.no_grad():
        for data in test_dataloader:
            inputs, labels = data
            outputs = output_function(inputs)

            # the class with the highest energy is what we choose as prediction
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            

    accuracy = correct / total
    loss = running_loss / len(test_dataloader)
    print(f"Validation Loss: {loss}")
    print(f"Validation Accuracy: {100 * accuracy:.2f}%")

    return correct, total
