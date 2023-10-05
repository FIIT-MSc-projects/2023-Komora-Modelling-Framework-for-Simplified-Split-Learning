import abc
import logging
import sys

from torch.distributed.rpc import RRef
import torch.distributed.rpc as rpc
import logging
import os

from splearning.utils.data_structures import AbstractServer, StartServerArguments

class bob(AbstractServer):
    def __init__(self, args: StartServerArguments):

        self.server_ref = RRef(self)
        self.model = args.get_server_model()
        model_rrefs = list(map(lambda x: RRef(x),self.model.parameters()))

        client_name = os.getenv('client', default="alice")

        self.alices = {
            rank+1: rpc.remote(
                to=client_name.replace("*", rank+1), 
                func=args.get_client(), 
                args=(self.server_ref,model_rrefs,rank+1,args.get_epochs())
            ) for rank in range(args.client_num_in_total)
        }

        self.last_alice_id = None
        self.client_num_in_total  = args.get_client_num_in_total()
        self.start_logger()

    def train_request(self,client_id):
        # call the train request from alice
        self.logger.info(f"Train Request for Alice{client_id}")
        if self.last_alice_id is None:
            self.alices[client_id].rpc_sync(timeout=0).train(None,None)
        else:
            self.alices[client_id].rpc_sync(timeout=0).train(self.alices[self.last_alice_id],self.last_alice_id)
        self.last_alice_id = client_id

    def eval_request(self):
        self.logger.info("Initializing Evaluation of all Alices")
        total = []
        num_corr = []
        check_eval = [self.alices[client_id].rpc_async(timeout=0).eval() for client_id in
                      range(1, self.client_num_in_total + 1)]
        for check in check_eval:
            corr, tot = check.wait()
            total.append(tot)
            num_corr.append(corr)

        self.logger.info("Accuracy over all data: {:.3f}".format(sum(num_corr) / sum(total)))

    def train(self,x):
        return self.model(x)

    def start_logger(self):
        self.logger = logging.getLogger("bob")
        self.logger.setLevel(logging.INFO)

        format = logging.Formatter("%(asctime)s: %(message)s")
        log_file_path = os.getenv('log_file')

        if not os.path.isdir(log_file_path):
            os.mkdir(log_file_path)

        fh = logging.FileHandler(filename=f"{log_file_path}/bob.log", mode='w')
        fh.setFormatter(format)
        fh.setLevel(logging.INFO)

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(format)
        sh.setLevel(logging.DEBUG)

        self.logger.addHandler(fh)
        self.logger.addHandler(sh)
        self.logger.info("Bob Started Getting Tipsy")
