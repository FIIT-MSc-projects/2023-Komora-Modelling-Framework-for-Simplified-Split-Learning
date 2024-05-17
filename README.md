Prototype of split learning based on [SplitLearning-alices](https://github.com/mlpotter/SplitLearning/tree/alices).

### DEMO - LOCAL:

1. Run client 1:
```python client_main.py --rank 1 --config env_configs/client1.env --port 8888```

    * ```--rank [Rank of client (needs to be unique)]```
    * ```--config [Path to the client config file]```
    * ```--port [PORT]```

2. Run client 2:
```python client_main.py --rank 2 --config env_configs/client2.env --port 8888```
    * ```--rank [Rank of client (needs to be unique)]```
    * ```--config [Path to the client config file]```
    * ```--port [PORT]```

3. Run server:
```python server_main.py --port 8888```
    * ```--port [PORT]```

### DEMO - DISTRIBUTED

0. Socket initialization:

```
ip addr - choose suitable interface

export TP_SOCKET_IFNAME=<suitable-interface>
export GLOO_SOCKET_IFNAME=<suitable-interface>

example for gngtx:
export TP_SOCKET_IFNAME=enp0s31f6
export GLOO_SOCKET_IFNAME=enp0s31f6

example for gna4000:
export TP_SOCKET_IFNAME=eno1
export GLOO_SOCKET_IFNAME=eno1
```

1. Run client 1:
```python client_main.py --rank 1 --config env_configs/client1.env --port 8888 --host 147.175.145.55```

    * ```--rank [Rank of client (needs to be unique)]```
    * ```--config [Path to the client config file]```
    * ```--port [PORT]```
    * ```--host [ip address of server]```

2. Run client 2:
```python client_main.py --rank 2 --config env_configs/client2.env --port 8888 --host 147.175.145.55```

    * ```--rank [Rank of client (needs to be unique)]```
    * ```--config [Path to the client config file]```
    * ```--port [PORT]```
    * ```--host [ip address of server]```

3. Run server:
```python server_main.py --port 8888 --host 147.175.145.55```
    * ```--port [PORT]```
    * ```--host [ip address of server]```

### CONFIG

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

