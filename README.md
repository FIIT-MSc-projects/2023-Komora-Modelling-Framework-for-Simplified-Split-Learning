Prototype of split learning based on [SplitLearning-alices](https://github.com/mlpotter/SplitLearning/tree/alices).

> DEMO:

* Run client 1:
```python client_main.py --rank 1 --config client1.env --port 8888```

    * ```--rank [Rank of client (needs to be unique)]```
    * ```--config [Path to the client config file]```
    * ```--port [PORT]```

* Run client 2:
```python client_main.py --rank 2 --config client2.env --port 8888```
    * ```--rank [Rank of client (needs to be unique)]```
    * ```--config [Path to the client config file]```
    * ```--port [PORT]```

* Run server:
```python server_main.py --port 8888```
    * ```--port [PORT]```

> CONFIG

* server.env:
```
log_file: "Path to log directory"
client: "Name of the clients with * being substitute for rank of client, example: alice* -> alice1, alice2, alice3...."
```

* client.env: 
```
log_file: "Path to log directory"
client_model_1_path: "Path to the serialized input model"
client_model_2_path: "Path to the serialized output model"
datapath: "Path to the data"
lr: "Learning rate"
momentum= "Momentum"
```

