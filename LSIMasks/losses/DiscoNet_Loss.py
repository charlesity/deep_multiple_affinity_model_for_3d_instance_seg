import warnings
import gc
import torch.nn as nn
import torch
from torch.nn.parallel import data_parallel


class DiscoNetLoss(nn.Module):
    def __init__(self, model, symmetric_loss, devices):
        super(DiscoNetLoss, self).__init__()
        self.disconet_model = model.mask_decoders[0] #only one present since we sample for multiple predictions
        self.K = self.disconet_model.nb_mask_output #pick the multiple outputs from one of the mask decoders
        self.symmetric_loss = symmetric_loss
        self.devices = devices


    def forward(self, selected_embeddings, gamma, target_me_masks, ignore_masks):
        valid_masks = 1. - ignore_masks.float()
        target_me_masks = target_me_masks * valid_masks
        pq_loss_sum = 0
        qq_loss_sum = 0
        for idx in range(len(selected_embeddings)):
            temp1 = torch.tensor([self.symmetric_loss(self.disconet_model(selected_embeddings[[idx],:])*valid_masks[[idx],:], target_me_masks[[idx],:]) for i in range(self.K)], requires_grad=True).cuda()
            pq_loss_sum += torch.mean(temp1)
            temp2 =torch.tensor([ self.symmetric_loss(self.disconet_model(selected_embeddings[[idx],:])*valid_masks[[idx],:], self.disconet_model(selected_embeddings[[idx],:])) for v1 in range(self.K) for v2 in range(self.K-1)], requires_grad=True).cuda()
            qq_loss_sum +=  (1/(self.K*self.K-1))*torch.sum(temp1)
        pq_loss_sum /=selected_embeddings.shape[0]
        qq_loss_sum /=selected_embeddings.shape[0]
        loss = pq_loss_sum - gamma*qq_loss_sum
        gc.collect()
        return loss










        #
        # embeddings_stack = torch.empty(size=(self.K, *target_me_masks.shape))
        # gt_stack = torch.empty(size=(self.K, *target_me_masks.shape))
        # for i in range(self.K):
        #     embeddings_stack[i] = self.disconet_model(selected_embeddings)*valid_masks
        #     gt_stack[i] = target_me_masks
        #
        # with warnings.catch_warnings(record=True) as w:
        #     loss1 = data_parallel(self.symmetric_loss, (embeddings_stack,  gt_stack),
        #                                         self.devices)
        #
        # print("here")
        #
    # def forward(self, a_selected_embeddings, gamma, a_target_me_masks, an_ignore_masks):
    #     a_selected_embeddings = unsqueeze(a_selected_embeddings, 0)
    #     a_target_me_masks = unsqueeze(a_target_me_masks, 0)
    #     an_ignore_masks = unsqueeze(an_ignore_masks, 0)
    #
    #     a_valid_masks = 1. - an_ignore_masks.float()
    #     sum_1 = 0
    #     for i in range(self.K):
    #         a_decoded_masks = self.disconet_model(a_selected_embeddings) #sample at every prediction
    #         a_decoded_masks = a_decoded_masks * a_valid_masks
    #         a_target_me_masks = a_target_me_masks * a_valid_masks
    #         sum_1 += self.symmetric_loss(a_decoded_masks, a_target_me_masks)
    #     sum_1+=1/self.K
    #
    #     sum_2 = 0
    #     for i in range(self.K):
    #         q1 = self.disconet_model(a_selected_embeddings)
    #         q1 = q1 * a_valid_masks
    #         for j in range(self.K-1):
    #             q2 = self.disconet_model(a_selected_embeddings)
    #             q2 = q2 * a_valid_masks
    #             sum_2 +=self.symmetric_loss(q1, q2)
    #     sum_1 +=1/(self.K*(self.K-1))
    #
    #     l = sum_1 - gamma * sum_2
    #     gc.collect()
    #     print(l)
    #     return l
