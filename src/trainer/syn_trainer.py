import torch
import numpy as np
from util.tools import get_timepc
from util.metric import topk_accuracy
from .basic_trainer import BasicTrainer


class SynTrainer(BasicTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.best_top1 = float("-inf")

    @torch.no_grad()
    def test_model_multi(self):
        t_s = get_timepc()
        self.reset(is_train=False)
        self.check_bn()

        top_1_list = []
        for k, v in self.test_loader.items():
            top_1 = 0.0
            _t_s = get_timepc()
            for batch_data in v:
                self.set_input(batch_data)
                self.forward()

                top_1 += topk_accuracy(self.output, self.target, top_k=(1,))[0]

            top_1 = 100 * top_1 / self.cfg.data.test_lengths[k]
            _t_e = get_timepc()

            t_cost = _t_e - _t_s
            self.cur_logs = "[{:^15}]\t[Time Cost:{:.3f}s]\t[Top_1:{:.3f}%]\t".format(k, t_cost, top_1)
            self.logger.log_msg(self.cur_logs) if self.master else None
            top_1_list.append(top_1)

        top_1 = np.mean(top_1_list)
        t_e = get_timepc()
        t_cost = t_e - t_s
        self.cur_logs = "Test Done!\tTime Cost:{:.3f}s\tAvg Top_1:{:.3f}%".format(t_cost, top_1)
        self.logger.log_msg(self.cur_logs) if self.master else None
        if self.best_top1 < top_1:
            self.best_top1 = top_1
            self.is_best = True
