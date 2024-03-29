import torch
import torch.nn as nn
import torch.nn.functional as F

class GuidedCxtAtten(nn.Layer):
    def __init__(self,
                 out_channels,
                 guidance_channels,
                 kernel_size=3,
                 stride=1,
                 rate=2):
        super().__init__()

        self.kernel_size = kernel_size
        self.rate = rate
        self.stride = stride
        self.guidance_conv = nn.Conv2d(
            in_channels=guidance_channels,
            out_channels=guidance_channels // 2,
            kernel_size=1)

        self.out_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias_attr=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, img_feat, alpha_feat, unknown=None, softmax_scale=1.):

        img_feat = self.guidance_conv(img_feat)
        img_feat = F.interpolate(
            img_feat, scale_factor=1 / self.rate, mode='nearest')

        # process unknown mask
        unknown, softmax_scale = self.process_unknown_mask(unknown, img_feat,
                                                           softmax_scale)

        img_ps, alpha_ps, unknown_ps = self.extract_feature_maps_patches(
            img_feat, alpha_feat, unknown)

        self_mask = self.get_self_correlation_mask(img_feat)

        # split tensors by batch dimension; tuple is returned
        img_groups = torch.split(img_feat, 1, dim=0)
        img_ps_groups = torch.split(img_ps, 1, dim=0)
        alpha_ps_groups = torch.split(alpha_ps, 1, dim=0)
        unknown_ps_groups = torch.split(unknown_ps, 1, dim=0)
        scale_groups = torch.split(softmax_scale, 1, dim=0)
        groups = (img_groups, img_ps_groups, alpha_ps_groups, unknown_ps_groups,
                  scale_groups)

        y = []

        for img_i, img_ps_i, alpha_ps_i, unknown_ps_i, scale_i in zip(*groups):
            # conv for compare
            similarity_map = self.compute_similarity_map(img_i, img_ps_i)

            gca_score = self.compute_guided_attention_score(
                similarity_map, unknown_ps_i, scale_i, self_mask)

            yi = self.propagate_alpha_feature(gca_score, alpha_ps_i)

            y.append(yi)

        y = torch.cat(y, dim=0)  # back to the mini-batch
        y = torch.reshape(y, alpha_feat.shape)

        y = self.out_conv(y) + alpha_feat

        return y

    def extract_feature_maps_patches(self, img_feat, alpha_feat, unknown):

        # extract image feature patches with shape:
        # (N, img_h*img_w, img_c, img_ks, img_ks)
        img_ks = self.kernel_size
        img_ps = self.extract_patches(img_feat, img_ks, self.stride)

        # extract alpha feature patches with shape:
        # (N, img_h*img_w, alpha_c, alpha_ks, alpha_ks)
        alpha_ps = self.extract_patches(alpha_feat, self.rate * 2, self.rate)

        # extract unknown mask patches with shape: (N, img_h*img_w, 1, 1)
        unknown_ps = self.extract_patches(unknown, img_ks, self.stride)
        unknown_ps = unknown_ps.squeeze(dim=2)  # squeeze channel dimension
        unknown_ps = unknown_ps.mean(dim=[2, 3], keepdim=True)

        return img_ps, alpha_ps, unknown_ps

    def extract_patches(self, x, kernel_size, stride):
        n, c, _, _ = x.shape
        x = self.pad(x, kernel_size, stride)
        x = F.unfold(x, [kernel_size, kernel_size], strides=[stride, stride])
        x = torch.transpose(x, (0, 2, 1))
        x = torch.reshape(x, (n, -1, c, kernel_size, kernel_size))

        return x

    def pad(self, x, kernel_size, stride):
        left = (kernel_size - stride + 1) // 2
        right = (kernel_size - stride) // 2
        pad = (left, right, left, right)
        return F.pad(x, pad, mode='reflect')

    def compute_guided_attention_score(self, similarity_map, unknown_ps, scale,
                                       self_mask):
        # scale the correlation with predicted scale factor for known and
        # unknown area
        unknown_scale, known_scale = scale[0]
        out = similarity_map * (
            unknown_scale * torch.gt(unknown_ps,
                                                torch.tensor([0.])) +
            known_scale * torch.le(unknown_ps, torch.tensor([0.])))
        # mask itself, self-mask only applied to unknown area
        out = out + self_mask * unknown_ps
        gca_score = F.softmax(out, dim=1)

        return gca_score

    def propagate_alpha_feature(self, gca_score, alpha_ps):

        alpha_ps = alpha_ps[0]  # squeeze dim 0
        if self.rate == 1:
            gca_score = self.pad(gca_score, kernel_size=2, stride=1)
            alpha_ps = torch.transpose(alpha_ps, (1, 0, 2, 3))
            out = F.conv2d(gca_score, alpha_ps) / 4.
        else:
            out = F.conv_transpose2d(
                gca_score, alpha_ps, stride=self.rate, padding=1) / 4. 

        return out

    def compute_similarity_map(self, img_feat, img_ps):
        img_ps = img_ps[0]  # squeeze dim 0
        # convolve the feature to get correlation (similarity) map
        img_ps_normed = img_ps / torch.clamp(self.l2_norm(img_ps), 1e-4)
        img_feat = F.pad(img_feat, (1, 1, 1, 1), mode='reflect')
        similarity_map = F.conv2d(img_feat, img_ps_normed)

        return similarity_map

    def get_self_correlation_mask(self, img_feat):
        _, _, h, w = img_feat.shape
        self_mask = F.one_hot(
            torch.reshape(torch.arange(h * w), (h, w)),
            num_classes=int(h * w))

        self_mask = torch.transpose(self_mask, (2, 0, 1))
        self_mask = torch.reshape(self_mask, (1, h * w, h, w))

        return self_mask * (-1e4)

    def process_unknown_mask(self, unknown, img_feat, softmax_scale):

        n, _, h, w = img_feat.shape

        if unknown is not None:
            unknown = unknown.copy()   ##paddle.clone
            unknown = F.interpolate(
                unknown, scale_factor=1 / self.rate, mode='nearest')
            unknown_mean = unknown.mean(axis=[2, 3])
            known_mean = 1 - unknown_mean
            unknown_scale = torch.clamp(
                torch.sqrt(unknown_mean / known_mean), 0.1, 10)
            known_scale = torch.clamp(
                torch.sqrt(known_mean / unknown_mean), 0.1, 10)
            softmax_scale = torch.cat([unknown_scale, known_scale], dim=1)
        else:
            unknown = torch.ones([n, 1, h, w])
            softmax_scale = torch.reshape(
                torch.tensor([softmax_scale, softmax_scale]), (1, 2))
            softmax_scale = softmax_scale.expand((n, 2))

        return unknown, softmax_scale

    @staticmethod
    def l2_norm(x:torch.Tensor):
        x = x**2
        x = x.sum(dim=[1, 2, 3], keepdim=True)
        return torch.sqrt(x)
