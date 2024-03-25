import os.path as osp
import torch
import torch.nn as nn
from torch.nn import functional as F
from config import cfg
from human_body_prior.tools.model_loader import load_vposer
import torchgeometry as tgm
from nets.layer import make_conv_layers, make_deconv_layers, make_conv1d_layers, make_linear_layers, GraphConvBlock, GraphResBlock
from utils.mano import MANO
from utils.smpl import SMPL


class Pose2Feat(nn.Module):
    def __init__(self, joint_num):
        super(Pose2Feat, self).__init__()
        self.joint_num = joint_num
        self.conv = make_conv_layers([64+joint_num,64])

    def forward(self, img_feat, joint_heatmap):
        feat = torch.cat((img_feat, joint_heatmap),1)
        feat = self.conv(feat)
        return feat


class PositionNet(nn.Module):
    def __init__(self):
        super(PositionNet, self).__init__()
        if 'FreiHAND' in cfg.trainset_3d + cfg.trainset_2d + [cfg.testset]:
            self.human_model = MANO()
            self.joint_num = self.human_model.graph_joint_num
        else:
            self.human_model = SMPL()
            self.joint_num = self.human_model.graph_joint_num

        self.hm_shape = [cfg.output_hm_shape[0] // 8, cfg.output_hm_shape[1] // 8, cfg.output_hm_shape[2] // 8]
        self.conv = make_conv_layers([2048, self.joint_num * self.hm_shape[0]], kernel=1, stride=1, padding=0, bnrelu_final=False)

    def soft_argmax_3d(self, heatmap3d):
        heatmap3d = heatmap3d.reshape((-1, self.joint_num, self.hm_shape[0] * self.hm_shape[1] * self.hm_shape[2]))
        heatmap3d = F.softmax(heatmap3d, 2)
        heatmap3d = heatmap3d.reshape((-1, self.joint_num, self.hm_shape[0], self.hm_shape[1], self.hm_shape[2]))

        accu_x = heatmap3d.sum(dim=(2, 3))
        accu_y = heatmap3d.sum(dim=(2, 4))
        accu_z = heatmap3d.sum(dim=(3, 4))

        accu_x = accu_x * torch.arange(self.hm_shape[2]).float().cuda()[None, None, :]
        accu_y = accu_y * torch.arange(self.hm_shape[1]).float().cuda()[None, None, :]
        accu_z = accu_z * torch.arange(self.hm_shape[0]).float().cuda()[None, None, :]

        accu_x = accu_x.sum(dim=2, keepdim=True)
        accu_y = accu_y.sum(dim=2, keepdim=True)
        accu_z = accu_z.sum(dim=2, keepdim=True)

        coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)
        return coord_out

    def forward(self, img_feat):
        # joint heatmap
        joint_heatmap = self.conv(img_feat).view(-1, self.joint_num, self.hm_shape[0], self.hm_shape[1], self.hm_shape[2])

        # joint coord
        joint_coord = self.soft_argmax_3d(joint_heatmap)

        # joint score sampling
        scores = []
        joint_heatmap = joint_heatmap.view(-1, self.joint_num, self.hm_shape[0] * self.hm_shape[1] * self.hm_shape[2])
        joint_heatmap = F.softmax(joint_heatmap, 2)
        joint_heatmap = joint_heatmap.view(-1, self.joint_num, self.hm_shape[0], self.hm_shape[1], self.hm_shape[2])
        for j in range(self.joint_num):
            x = joint_coord[:, j, 0] / (self.hm_shape[2] - 1) * 2 - 1
            y = joint_coord[:, j, 1] / (self.hm_shape[1] - 1) * 2 - 1
            z = joint_coord[:, j, 2] / (self.hm_shape[0] - 1) * 2 - 1
            grid = torch.stack((x, y, z), 1)[:, None, None, None, :]
            score_j = F.grid_sample(joint_heatmap[:, j, None, :, :, :], grid, align_corners=True)[:, 0, 0, 0, 0]  # (batch_size)
            scores.append(score_j)
        scores = torch.stack(scores)  # (joint_num, batch_size)
        joint_score = scores.permute(1, 0)[:, :, None]  # (batch_size, joint_num, 1)
        return joint_coord, joint_score


class RotationNet(nn.Module):
    def __init__(self):
        super(RotationNet, self).__init__()

        if 'FreiHAND' in cfg.trainset_3d + cfg.trainset_2d + [cfg.testset]:
            self.human_model = MANO()
            self.joint_num = self.human_model.graph_joint_num
            self.graph_adj = torch.from_numpy(self.human_model.graph_adj).float()
        else:
            self.human_model = SMPL()
            self.joint_num = self.human_model.graph_joint_num
            self.graph_adj = torch.from_numpy(self.human_model.graph_adj).float()

        # graph convs
        self.graph_block = nn.Sequential(*[\
            GraphConvBlock(self.graph_adj, 2048+4, 128),
            GraphResBlock(self.graph_adj, 128),
            GraphResBlock(self.graph_adj, 128),
            GraphResBlock(self.graph_adj, 128),
            GraphResBlock(self.graph_adj, 128)])

        self.hm_shape = [cfg.output_hm_shape[0] // 8, cfg.output_hm_shape[1] // 8, cfg.output_hm_shape[2] // 8]

        self.root_pose_out = make_linear_layers([self.joint_num*128, 6], relu_final=False)
        self.pose_out = make_linear_layers([self.joint_num*128, self.human_model.vposer_code_dim], relu_final=False) # vposer latent code
        self.shape_out = make_linear_layers([self.joint_num*128, self.human_model.shape_param_dim], relu_final=False)
        self.cam_out = make_linear_layers([self.joint_num*128,3], relu_final=False)

    def sample_image_feature(self, img_feat, joint_coord_img):
        img_feat_joints = []
        for j in range(self.joint_num):
            x = joint_coord_img [: ,j,0] / (self.hm_shape[2]-1) * 2 - 1
            y = joint_coord_img [: ,j,1] / (self.hm_shape[1]-1) * 2 - 1
            grid = torch.stack( (x, y),1) [:,None,None,:]
            img_feat = img_feat.float()
            img_feat_j = F.grid_sample(img_feat, grid, align_corners=True) [: , : , 0, 0] # (batch_size, channel_dim)
            img_feat_joints.append(img_feat_j)
        img_feat_joints = torch.stack(img_feat_joints) # (joint_num, batch_size, channel_dim)
        img_feat_joints = img_feat_joints.permute(1, 0 ,2) # (batch_size, joint_num, channel_dim)
        return img_feat_joints

    def forward(self, img_feat, joint_coord_img, joint_score):
        # pose parameter
        img_feat_joints = self.sample_image_feature(img_feat, joint_coord_img)
        feat = torch.cat((img_feat_joints, joint_coord_img, joint_score),2)
        feat = self.graph_block(feat)
        root_pose = self.root_pose_out(feat.view(-1,self.joint_num*128))
        pose_param = self.pose_out(feat.view(-1,self.joint_num*128))
        # shape parameter
        shape_param = self.shape_out(feat.view(-1,self.joint_num*128))
        # camera parameter
        cam_param = self.cam_out(feat.view(-1,self.joint_num*128))

        return root_pose, pose_param, shape_param, cam_param


class Vposer(nn.Module):
    def __init__(self):
        super(Vposer, self).__init__()
        self.vposer, _ = load_vposer(osp.join(cfg.human_model_path, 'smpl', 'VPOSER_CKPT'), vp_model='snapshot')
        self.vposer.eval()

    def forward(self, z):
        batch_size = z.shape[0]
        body_pose = self.vposer.decode(z, output_type='aa').view(batch_size,-1 ).view(-1,24-3,3) # without root, R_Hand, L_Hand
        zero_pose = torch.zeros((batch_size,1,3)).float().cuda()

        # attach zero hand poses
        body_pose = torch.cat((body_pose, zero_pose, zero_pose),1)
        body_pose = body_pose.view(batch_size,-1)
        return body_pose
