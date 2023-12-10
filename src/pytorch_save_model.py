from torchinfo import summary
from data_handling_experiment_3.prepare_cifar_data import load_image_datasets, prepare_data
from experiment3_resnet import ResNet


model = ResNet()

datapath = "experiment3/data"
train_dataset, test_dataset = load_image_datasets(datapath)
train_dataloader, test_dataloader = prepare_data(train_dataset, test_dataset, 128)

# Get an individual batch
for inputs, labels in train_dataloader:
    # 'inputs' is a tensor representing the input data for the batch
    input_size = inputs.size()  # Exclude batch size
    break

print("Input size:", input_size)

summary(model, [128, 3, 32, 32])