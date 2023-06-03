import fedml
from fedml import FedMLRunner
from mpi4py import MPI

if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    print("ID: ", args.process_id, "\n\n\n") #0
    print("Worker no: ", args.worker_num, "\n\n\n") #1
    print("Comm: ", args.comm, "\n\n\n") # MPI.COMM_WORLD

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load(args)

    # load model
    model = fedml.model.create(args, output_dim)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model)
    fedml_runner.run()