import os
import flwr as fl
import tensorflow as tf
from clients import BasicClient

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def start_client(model, data, server_address, client):
    # Load model and data (MobileNetV2, CIFAR-10)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    (x_train, y_train), (x_test, y_test) = data

    fl.client.start_numpy_client(server_address=server_address, client=client(model=model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test))


if __name__ == '__main__':
    print("Zdes")
    
    model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    data = tf.keras.datasets.cifar10.load_data()
    
    start_client(model=model, data=data, server_address="127.0.0.1:8080", client=BasicClient)

    