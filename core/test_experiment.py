import pytest
import numpy as np
import torch 
import torch.nn as nn


import core.experiment as exp

class Test_SupConLoss:

    @staticmethod
    def sup_con_loss(x, y, temp):       
            loss = 0
            for i in np.arange(x.size(0)):

                sj = 0
                cnt = 0
                for j in np.arange(x.size(0)):
                    y_j = y[j].item()
                    y_i = y[i].item()

                    if i != j and y_i == y_j:
                        cnt += 1

                        t0 = torch.div(x[i,:].dot(x[j,:]), temp)
                        t0 = t0.exp()

                        t1 = 0
                        for k in np.arange(x.size(0)):
                            if i != k:
                                tmp = torch.div(x[i,:].dot(x[k,:]), temp)
                                tmp = tmp.exp()
                                t1 += tmp

                        sj += torch.log(t0/t1)

                # This is an adaption to the original loss:
                # we ignore samples which have no positive partner. 
                # The original implementation has not to consider this case 
                # as by batchconstruction the existence of an same-class partner
                # is assured for each sample. 
                if cnt > 0: 
                    sj = torch.div(sj, cnt)
                    loss = loss + sj

            loss = -1.0 * loss/x.size(0)
            return loss

    def test_SupConLoss(self):
        z = torch.randn(64, 10)
        y = torch.tensor([0, 1, 2, 3]*16)
        z = z/torch.norm(z, p=2, keepdim=True, dim=1)
        t = 0.1

        loss_fn = exp.SupConLoss(temperature=t)

        l_1 = loss_fn((None, z), y)
        l_2 = self.sup_con_loss(z, y, temp=t) 
        assert torch.isclose(l_1, l_2)

        # y[-1] = 4 
        # l_1 = loss_fn((None, z), y)
        # l_2 = self.sup_con_loss(z, y, temp=t) 
        # assert torch.isclose(l_1, l_2)