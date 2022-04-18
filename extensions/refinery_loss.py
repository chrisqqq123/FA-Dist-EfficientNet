__author__ = "Hessam Bagherinezhad <hessam@xnor.ai>"

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import loss
from models.efficientnet import Arctransf, Margloss

class RefineryLoss(loss._Loss):
    """The KL-Divergence loss for the model and refined labels output.

    output must be a pair of (model_output, refined_labels), both NxC tensors.
    The rows of refined_labels must all add up to one (probability scores);
    however, model_output must be the pre-softmax output of the network."""

    def __init__(self, size_average=None, reduce=None, reduction='mean', cosln=True, scl=5.0):
        super(loss._Loss, self).__init__()
        if size_average is not None or reduce is not None:
            # self.reduction = _reduction.legacy_get_string(size_average, reduce)
            self.reduction = reduction
        else:
            self.reduction = reduction
        self.using_cos = cosln
        self.s = scl

    def fa_loss(self, feat, ref_feat):
        batch_size, ch, h, w = feat.size(0), feat.size(1), feat.size(2), feat.size(3)
        # print(feat.size(), ref_feat.size())
        feat = feat.view(batch_size, ch, -1)
        norm_feat = feat.norm(p=2,dim=1).unsqueeze(1)
        feat = torch.div(feat, norm_feat)
        tran_feat = feat.permute(0,2,1)
        ft_map = torch.matmul(tran_feat,feat)

        ref_feat = F.interpolate(ref_feat, size=(h,w),mode='bilinear', align_corners=True)
        # print(feat.size(), ref_feat.size(1))
        ref_feat = ref_feat.view(batch_size, ref_feat.size(1), -1)
        norm_ref_feat = ref_feat.norm(p=2,dim=1).unsqueeze(1)
        ref_feat = torch.div(ref_feat, norm_ref_feat)
        tran_ref_feat = ref_feat.permute(0,2,1)
        refft_map = torch.matmul(tran_ref_feat,ref_feat)

        loss = (ft_map-refft_map).norm(p=1)/(h*h*w*w)

        return  loss

    def loc_fa_loss(self, feat, ref_feat, alpha=2):

        batch_size, ch, h, w = feat.size(0), feat.size(1), feat.size(2), feat.size(3)
        ###### create mask #############
        # em = torch.ones(h,h).cuda()
        # em1 = em+alpha*(torch.diag(em.diagonal(1),1)+torch.diag(em.diagonal(-1),-1))
        # em2 = em1+alpha*torch.diag(em.diagonal())
        # m = []
        # for i in range(w):
        #     ls = []
        #     for j in range(w):
        #         if j==i:
        #             ls.append(em1)
        #         elif (j==i+1) or (j==i-1):
        #             ls.append(em2)
        #         else:
        #             ls.append(em)
        #     tmp = torch.cat(ls,1)
        #     m.append(tmp)
        # mask = torch.cat(m,0)
        # print(mask.type())
        ##########larger mask###########
        mask = torch.ones(h*w,h*w).cuda()
        dist = 10
        for i in range(h*w):
            for j in range(i+1,h*w):
                m_cur, n_cur = j//h, j%h
                m_cen, n_cen = i//h, i%h
                if abs(m_cur-m_cen)<=dist and abs(n_cur-n_cen)<=dist:
                    mask[i][j]=mask[j][i]=alpha
        ################################

        # print(feat.size(), ref_feat.size())
        feat = feat.view(batch_size, ch, -1)
        norm_feat = feat.norm(p=2,dim=1).unsqueeze(1)
        feat = torch.div(feat, norm_feat)
        tran_feat = feat.permute(0,2,1)
        ft_map = torch.matmul(tran_feat,feat)
        # ft_map *= mask

        ref_feat = F.interpolate(ref_feat, size=(h,w),mode='bilinear', align_corners=True)
        # print(feat.size(), ref_feat.size(1))
        ref_feat = ref_feat.view(batch_size, ref_feat.size(1), -1)
        norm_ref_feat = ref_feat.norm(p=2,dim=1).unsqueeze(1)
        ref_feat = torch.div(ref_feat, norm_ref_feat)
        tran_ref_feat = ref_feat.permute(0,2,1)
        refft_map = torch.matmul(tran_ref_feat,ref_feat)
        # refft_map *= mask

        loss = ((ft_map-refft_map)*mask).norm(p=1)/(h*h*w*w)

        return  loss
    
    def fast_fa_loss(self, feat, ref_feat):
        batch_size, ch, h, w = feat.size(0), feat.size(1), feat.size(2), feat.size(3)

        # generating random vector  (HW) x 1
        vec = torch.randn(h*w, 1).unsqueeze(0).repeat(batch_size,1,1).cuda()
        # print('@@@',vec.size())
        # print('###', vec[0:3,0:3,0:3])

        # print(feat.size(), ref_feat.size())
        feat = feat.view(batch_size, ch, -1)  # [batch, ch, HW]
        norm_feat = feat.norm(p=2,dim=1).unsqueeze(1)
        feat = torch.div(feat, norm_feat)
        tran_feat = feat.permute(0,2,1)    # [batch, HW, ch]

        # ft_map = torch.matmul(tran_feat,feat)
        ft_map = torch.matmul(tran_feat, torch.matmul(feat, vec) )
        # print(ft_map.size())

        ref_feat = F.interpolate(ref_feat, size=(h,w),mode='bilinear', align_corners=True)
        # print(feat.size(), ref_feat.size(1))
        ref_feat = ref_feat.view(batch_size, ref_feat.size(1), -1)
        norm_ref_feat = ref_feat.norm(p=2,dim=1).unsqueeze(1)
        ref_feat = torch.div(ref_feat, norm_ref_feat)
        tran_ref_feat = ref_feat.permute(0,2,1)

        # refft_map = torch.matmul(tran_ref_feat,ref_feat)
        refft_map = torch.matmul(tran_ref_feat, torch.matmul(ref_feat, vec) )

        loss = (ft_map-refft_map).norm(p=1)/(h*h*w*w)

        return  loss
    
    def forward(self, output, target):
        if not self.training:
            # Loss is normal cross entropy loss between the model output and the
            # target.
            if self.using_cos:
                lossfun = Margloss(s = self.s)
                return lossfun(output, target)
            else:
                return F.cross_entropy(output, target)

        assert type(output) == tuple and len(output) == 4 and output[0].size() == \
            output[1].size(), "output must a pair of tensors of same size."

        # Target is ignored at training time. Loss is defined as KL divergence
        # between the model output and the refined labels.
        model_output, refined_labels, inner_feature, refined_feature = output
        if refined_labels.requires_grad:
            raise ValueError("Refined labels should not require gradients.")
        model_output  = Arctransf(model_output, self.s)     
        model_output_log_prob = F.log_softmax(model_output, dim=1)   
        del model_output

        # Loss is -dot(model_output_log_prob, refined_labels). Prepare tensors
        # for batch matrix multiplicatio
        refined_labels = refined_labels.unsqueeze(1)
        model_output_log_prob = model_output_log_prob.unsqueeze(2)

        # Compute the loss, and average/sum for the batch.
        cross_entropy_loss = -torch.bmm(refined_labels, model_output_log_prob)

        if self.reduction == 'mean':
            cross_entropy_loss = cross_entropy_loss.mean()
        else:
            cross_entropy_loss = cross_entropy_loss.sum()
        total_loss = cross_entropy_loss
        ###### add FA loss ###########
        # regular FA
        # faloss = self.fa_loss(inner_feature, refined_feature)
        # total_loss += faloss

        # loc FA
        # locfaloss = self.loc_fa_loss(inner_feature, refined_feature)
        # total_loss += locfaloss

        # Fast FA
        fastfaloss = self.fast_fa_loss(inner_feature, refined_feature)
        total_loss += fastfaloss
        ##############################


        # Return a pair of (loss_output, model_output). Model output will be
        # used for top-1 and top-5 evaluation.
        model_output_log_prob = model_output_log_prob.squeeze(2)
        # print(model_output_log_prob.size())
        return (total_loss, model_output_log_prob)
