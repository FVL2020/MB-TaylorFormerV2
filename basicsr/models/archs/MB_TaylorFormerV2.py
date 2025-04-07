## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torchvision.ops.deform_conv import DeformConv2d
from pdb import set_trace as stx
import numbers
import math

from einops import rearrange
import numpy as np
import torchvision


freqs_dict = dict()

##########################################################################
## Layer Norm
##rotary_pos_embed


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape,path):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (path, normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        #assert len(normalized_shape) == 1
        self.path=path
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        x = x / torch.sqrt(sigma + 1e-5)
        x = rearrange(x, '(p b) n c -> b n p c',p=self.path)

        return rearrange(x * self.weight, 'b n p c -> (p b) n c')


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape, path):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (path, normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        #assert len(normalized_shape) == 1
        self.path = path
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(sigma + 1e-5)

        x=rearrange(x,'(p b) n p c -> b n p c',p=self.path)
        return rearrange(x * self.weight + self.bias, 'b n p c -> (p b) n c')


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type,path):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim,path)
        else:
            self.body = WithBias_LayerNorm(dim,path)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, num_path):
        super(FeedForward, self).__init__()

        hidden_features = int(dim//num_path * ffn_expansion_factor)*num_path

        self.project_in_1 = nn.Conv2d(dim, hidden_features , kernel_size=1,groups=num_path, bias=bias)
        self.project_in_2 = nn.Conv2d(dim, hidden_features, kernel_size=1, groups=num_path, bias=bias)

        self.dwconv_1 = nn.Conv2d(hidden_features , hidden_features , kernel_size=3, stride=1, padding=1,
                                groups=hidden_features , bias=bias)
        self.dwconv_2 = nn.Conv2d(hidden_features , hidden_features , kernel_size=3, stride=1, padding=1,
                                groups=hidden_features , bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1,groups=num_path, bias=bias)
        self.num_path=num_path
    def forward(self, x):
        x = rearrange(x, '(p B) c h w -> B (p c) h w',  p=self.num_path)
        x1 = self.dwconv_1(self.project_in_1(x))
        x2 = self.dwconv_2(self.project_in_2(x))
        #x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        x = rearrange(x, 'B (p c) h w -> (p B) c h w',  p=self.num_path)
        return x

class refine_att(nn.Module):
    """Convolutional relative position encoding."""
    def __init__(self, Ch, h, window,path):

        super().__init__()

        if isinstance(window, int):
            # Set the same window size for all attention heads.
            window = {window: h}
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:

            raise ValueError()

        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1  # Use dilation=1 at default.
            padding_size = (cur_window + (cur_window - 1) *
                            (dilation - 1)) // 2
            cur_conv=nn.Conv2d(
                cur_head_split * Ch*path,
                cur_head_split*path,
                kernel_size=(cur_window, cur_window),
                padding=(padding_size, padding_size),
                dilation=(dilation, dilation),
                groups=cur_head_split*path,
            )



            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.num_path=path
        self.channel_splits = [ x * Ch for x in self.head_splits]

    def forward(self, v, size):
        """foward function"""
        B, h, N, Ch = v.shape
        H, W = size

        v_img = v

        v_img = rearrange(v_img, "B h (H W) Ch -> B h Ch H W", H=H, W=W)
        v_img = rearrange(v_img , "b h Ch H W -> b (h Ch) H W", H=H, W=W)
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)
        v_img_list_reshape=[]
        for i in range(len(v_img_list)):
            v_img_list_reshape.append(rearrange(v_img_list[i], "(p B) c H W -> B (p c) H W", H=H, W=W, p=self.num_path))
        v_att_list = [
            conv(x) for conv, x in zip(self.conv_list, v_img_list_reshape)
        ]
        v_img_list_reshape=[]
        for i in range(len(v_att_list)):
            v_img_list_reshape.append(rearrange(v_att_list[i], "B (p c) H W -> (p B) c H W", H=H, W=W, p=self.num_path))
        
        v_att = torch.cat(v_img_list_reshape, dim=1)
        v_att = rearrange(v_att, "B (h Ch) H W -> B h (H W) Ch", h=h)


        return v_att

##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias,shared_refine_att=None,qk_norm=1,path=2,focusing_factor=8,N=256*256):
        super(Attention, self).__init__()
        self.norm=qk_norm
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(path,num_heads, 1, 1))
        #self.Leakyrelu=nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.qkv = nn.Conv2d(dim*path, dim * 3*path, kernel_size=1,groups=path, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3*path, dim * 3*path, kernel_size=3, stride=1, padding=1, groups=dim * 3*path, bias=bias)
        print(self.qkv_dwconv)
        self.project_out = nn.Conv2d(dim*path, dim*path, kernel_size=1,groups=path,bias=bias)
        self.num_path=path
        if num_heads == 8:
            crpe_window = {
                3: 2,
                5: 3,
                7: 3
            }
        elif num_heads == 1:
            crpe_window = {
                3: 1,
            }
        elif num_heads == 2:
            crpe_window = {
                3: 2,
            }
        elif num_heads == 4:
            crpe_window = {
                3: 2,
                5: 2,
            }
        self.refine_att = refine_att(Ch=dim // num_heads,
                                     h=num_heads,
                                     window=crpe_window,
                                     path=path)
        self.focusing_factor=focusing_factor
        self.scale = nn.Parameter(torch.ones(path,num_heads, 1, 1))
        #self.N=N
        self.one_M=nn.Parameter(torch.ones(1), requires_grad=False)
    def forward(self, x):
        b, c, h, w = x.shape

        relu = nn.ReLU(inplace=False)
        x= rearrange(x, '(p B) c h w -> B (p c) h w', B=b//self.num_path, p=self.num_path)
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = rearrange(qkv, 'B (p c) h w -> (p B) c h w', B=b // self.num_path, p=self.num_path)
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head (h w) c', head=self.num_heads)

        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head (h w) c', head=self.num_heads)

        q_norm = torch.norm(q, p=2, dim=-1, keepdim=True)+1e-8
        q_1 = torch.div(q, q_norm)
        k_norm = torch.norm(k, p=2, dim=-2, keepdim=True)+1e-8
        k_1 = torch.div(k, k_norm)

        q_2 = relu(q) ** self.focusing_factor
        k_2 = relu(k) ** self.focusing_factor

        q_2=(q_2/(q_2.norm(dim=-1, keepdim=True)+1e-8))#*q_norm
        k_2 = (k_2 / (k_2.norm(dim=-2, keepdim=True)+1e-8)) #* k_norm

        refine_weight = self.refine_att(v, size=(h, w))
        refine_weight = self.sigmoid(refine_weight)
        attn_2 = k_2@v
        attn_1=k_1@v
        scale= self.sigmoid(self.scale).repeat_interleave(b//self.num_path,0)


        out_numerator = torch.sum(v, dim=-2).unsqueeze(2)+(q_1@attn_1)+scale*(q_2@attn_2)#self.one_M \
        
        N=h*w
        one_M=self.one_M*N
        target_shape = (N, c // self.num_heads)
        expanded_tensor = one_M.expand(target_shape)
        out_denominator = expanded_tensor + q_1 @ torch.sum(k_1, dim=-1).unsqueeze(
            3).repeat(1, 1, 1, c // self.num_heads) + \
                          q_2 @ torch.sum(scale * k_2, dim=-1).unsqueeze(3).repeat(1, 1, 1, c // self.num_heads) + 1e-8

        out = torch.div(out_numerator, out_denominator)
       
        out = out* (self.temperature.repeat_interleave(b//self.num_path,0))+refine_weight
        
        out = rearrange(out, 'b head (h w) c-> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = rearrange(out, '(p b) c h w-> b (p c) h w',  h=h, w=w, p=self.num_path)
        out = self.project_out(out)
        out = rearrange(out, 'b (p c) h w-> (p b) c h w', h=h, w=w, p=self.num_path)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type,shared_refine_att=None,qk_norm=1,N=256*256, path_emb_dim=48,num_path=2):
        super(TransformerBlock, self).__init__()
        self.num_path=num_path
        self.norm1 = LayerNorm(dim, LayerNorm_type,num_path)
        self.attn = Attention(dim, num_heads, bias,shared_refine_att=shared_refine_att,qk_norm=qk_norm,path=num_path)
        self.norm2 = LayerNorm(dim, LayerNorm_type,num_path)

        self.ffn = FeedForward(dim*num_path, ffn_expansion_factor, bias, num_path)

    def forward(self, x):

        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class MHCAEncoder(nn.Module):
    """Multi-Head Convolutional self-Attention Encoder comprised of `MHCA`
    blocks."""

    def __init__(
            self,
            dim,
            num_layers=1,
            num_heads=8,
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type='BiasFree',
            qk_norm=1,
            N=256*256,
            num_path=4,
            path_emb_dim=48
    ):
        super().__init__()

        self.num_layers = num_layers
        self.MHCA_layers = nn.ModuleList([
            TransformerBlock(
                dim,
                num_heads=num_heads,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                qk_norm=qk_norm,
                N=N,
                path_emb_dim=path_emb_dim,
                num_path=num_path
            ) for idx in range(self.num_layers)
        ])
        


    def forward(self, x, size):
        b,_,_,_=x[0].shape
        """foward function"""


                    
        x=torch.cat(x,dim=0)
        x = x.flatten(2).transpose(1, 2).contiguous()
        H, W = size
        B = x.shape[0]
        # return x's shape : [B, N, C] -> [B, C, H, W]
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        for i in range(len(self.MHCA_layers)):
            x = self.MHCA_layers[i](x)

        return x





class MHCA_stage(nn.Module):
    """Multi-Head Convolutional self-Attention stage comprised of `MHCAEncoder`
    layers."""

    def __init__(
            self,
            embed_dim,
            out_embed_dim,
            num_layers=1,
            num_heads=8,
            ffn_expansion_factor=2.66,
            num_path=4,
            bias=False,
            LayerNorm_type='BiasFree',
            qk_norm=1,
            N=256*256,
            path_emb_dim=48
    ):
        super().__init__()
        self.mhca_blk=MHCAEncoder(
                embed_dim,
                num_layers,
                num_heads,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                qk_norm=qk_norm,
                N=N,
                num_path=num_path,
                path_emb_dim=path_emb_dim,
            )
        self.aggregate = SKFF(embed_dim,height=num_path)
        self.num_path=num_path

    def forward(self, inputs):
        """foward function"""
        #att_outputs = [self.InvRes(inputs[0])]
        b,_,h,w=inputs[0].shape

        x=inputs
        #for idx in range(len(inputs)):
        #    torch.cat((inputs[idx],pos[idx]),1)


        out=self.mhca_blk(x, size=(h,w))
        att_outputs=out.chunk(self.num_path, dim=0)
        out = self.aggregate(att_outputs)


        return out



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class Conv2d_BN(nn.Module):


    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            pad=0,
            dilation=1,
            groups=1,
            bn_weight_init=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=None,
    ):
        super().__init__()

        self.conv = torch.nn.Conv2d(in_ch,
                                    out_ch,
                                    kernel_size,
                                    stride,
                                    pad,
                                    dilation,
                                    groups,
                                    bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))

        self.act_layer = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x):

        x = self.conv(x)
        x = self.act_layer(x)

        return x


class SKFF(nn.Module):
    def __init__(self, in_channels, height=2, reduction=8, bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU())

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V



class DWConv2d_BN(nn.Module):

    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.Hardswish,
            bn_weight_init=1,
            offset_clamp=(-1,1)
    ):
        super().__init__()
        self.offset_clamp=offset_clamp
        self.offset_generator=nn.Sequential(nn.Conv2d(in_channels=in_ch,out_channels=in_ch,kernel_size=3,
                                                      stride= 1,padding= 1,bias= False,groups=in_ch),
                                            nn.Conv2d(in_channels=in_ch, out_channels=18,
                                                      kernel_size=1,
                                                      stride=1, padding=0, bias=False)

                                            )
        self.dcn=DeformConv2d(
                    in_channels=in_ch,
                    out_channels=in_ch,
                    kernel_size=3,
                    stride= 1,
                    padding= 1,
                    bias= False,
                    groups=in_ch
                    )#.cuda(7)
        self.pwconv = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)


        #self.bn = norm_layer(out_ch)
        self.act = act_layer() if act_layer is not None else nn.Identity()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()




    def forward(self, x):

        offset = self.offset_generator(x)

        if self.offset_clamp:
            offset=torch.clamp(offset, min=self.offset_clamp[0], max=self.offset_clamp[1])#.cuda(7)1
        x = self.dcn(x,offset)
        x=self.pwconv(x)

        x = self.act(x)
        return x


class DWCPatchEmbed(nn.Module):
    """Depthwise Convolutional Patch Embedding layer Image to Patch
    Embedding."""

    def __init__(self,
                 in_chans=3,
                 embed_dim=768,
                 patch_size=16,
                 stride=1,
                 idx=0,
                 act_layer=nn.Hardswish,
                 offset_clamp=(-1,1)):
        super().__init__()


        self.patch_conv = DWConv2d_BN(
                in_chans,
                embed_dim,
                kernel_size=patch_size,
                stride=stride,
                act_layer=act_layer,
                offset_clamp=offset_clamp
            )


    def forward(self, x):
        """foward function"""
        x = self.patch_conv(x)

        return x


class Patch_Embed_stage(nn.Module):
    """Depthwise Convolutional Patch Embedding stage comprised of
    `DWCPatchEmbed` layers."""

    def __init__(self, in_chans, embed_dim, num_path=4, isPool=False,offset_clamp=(-1,1)):
        super(Patch_Embed_stage, self).__init__()

        self.patch_embeds = nn.ModuleList([
            DWCPatchEmbed(
                in_chans=in_chans if idx == 0 else embed_dim,
                embed_dim=embed_dim,
                patch_size=3,
                stride=1,
                idx=idx,
                offset_clamp=offset_clamp
            ) for idx in range(num_path)
        ])


    def forward(self, x):
        """foward function"""
        att_inputs = []


        for idx in range(len(self.patch_embeds)):
            x = self.patch_embeds[idx](x)
            att_inputs.append(x)



        return att_inputs


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)


    def forward(self, x):
        x = self.proj(x)
        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, input_feat,out_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(input_feat, input_feat, kernel_size=3, stride=1, padding=1, groups=input_feat, bias=False, ),
            nn.Conv2d(input_feat, out_feat // 4, 1, 1, 0, bias=False),
            nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, input_feat,out_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(input_feat, input_feat, kernel_size=3, stride=1, padding=1, groups=input_feat, bias=False, ),
            nn.Conv2d(input_feat, out_feat * 4, 1, 1, 0, bias=False),
            nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)





##########################################################################
class MB_TaylorFormer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 model_name=1,
                 dim=[24,48,72,96],
                 num_blocks=[2,3,3,4],
                 num_refinement_blocks=2,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False,
                 num_path=[2,2,2,2],  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 qk_norm=1,
                 offset_clamp=(-1,1),
                 N=128**2,
                 path_emb_dim=12

                 ):
        self.path_emb_dim=path_emb_dim
        super(MB_TaylorFormer, self).__init__()


        self.patch_embed = OverlapPatchEmbed(inp_channels, dim[0])
        self.patch_embed_encoder_level1 = Patch_Embed_stage(dim[0], dim[0], num_path=num_path[0], isPool=False,offset_clamp=offset_clamp)
        self.encoder_level1 = MHCA_stage(dim[0], dim[0], num_layers=num_blocks[0], num_heads=heads[0],
                                         ffn_expansion_factor=2.66, num_path=num_path[0],
                                         bias=False, LayerNorm_type='BiasFree',qk_norm=qk_norm,N=N,path_emb_dim=path_emb_dim*4)

        self.down1_2 = Downsample(dim[0],dim[1])  ## From Level 1 to Level 2

        self.patch_embed_encoder_level2 = Patch_Embed_stage(dim[1], dim[1], num_path=num_path[1], isPool=False,offset_clamp=offset_clamp)
        self.encoder_level2 = MHCA_stage(dim[1], dim[1], num_layers=num_blocks[1], num_heads=heads[1],
                                         ffn_expansion_factor=2.66,
                                         num_path=num_path[1], bias=False, LayerNorm_type='BiasFree',qk_norm=qk_norm,N=N//4,path_emb_dim=path_emb_dim*4)

        self.down2_3 = Downsample(dim[1],dim[2])  ## From Level 2 to Level 3

        self.patch_embed_encoder_level3 = Patch_Embed_stage(dim[2], dim[2], num_path=num_path[2],
                                                            isPool=False,offset_clamp=offset_clamp)
        self.encoder_level3 = MHCA_stage(dim[2], dim[2], num_layers=num_blocks[2], num_heads=heads[2],
                                         ffn_expansion_factor=2.66,
                                         num_path=num_path[2], bias=False, LayerNorm_type='BiasFree',qk_norm=qk_norm,N=N//16,path_emb_dim=path_emb_dim*4)

        self.down3_4 = Downsample(dim[2],dim[3])  ## From Level 3 to Level 4

        self.patch_embed_latent = Patch_Embed_stage(dim[3], dim[3], num_path=num_path[3],
                                                    isPool=False,offset_clamp=offset_clamp)
        self.latent = MHCA_stage(dim[3], dim[3], num_layers=num_blocks[3], num_heads=heads[3],
                                 ffn_expansion_factor=2.66, num_path=num_path[3], bias=False,
                                 LayerNorm_type='BiasFree',qk_norm=qk_norm,N=N//64,path_emb_dim=path_emb_dim*4)


        self.up4_3 = Upsample(int(dim[3]),dim[2])  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Sequential(
            nn.Conv2d(dim[2]*2, dim[2], 1, 1, 0, bias=bias),
        )

        self.patch_embed_decoder_level3 = Patch_Embed_stage(dim[2], dim[2], num_path=num_path[2],
                                                            isPool=False,offset_clamp=offset_clamp)
        self.decoder_level3 = MHCA_stage(dim[2], dim[2], num_layers=num_blocks[2], num_heads=heads[2],
                                         ffn_expansion_factor=2.66, num_path=num_path[2], bias=False,
                                         LayerNorm_type='BiasFree',qk_norm=qk_norm,N=N//16,path_emb_dim=path_emb_dim*4)

        self.up3_2 = Upsample(int(dim[2]),dim[1])  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Sequential(
            nn.Conv2d(dim[1]*2, dim[1], 1, 1, 0, bias=bias),
        )

        self.patch_embed_decoder_level2 = Patch_Embed_stage(dim[1], dim[1], num_path=num_path[1],
                                                            isPool=False,offset_clamp=offset_clamp)
        self.decoder_level2 = MHCA_stage(dim[1], dim[1], num_layers=num_blocks[1], num_heads=heads[1],
                                         ffn_expansion_factor=2.66, num_path=num_path[1], bias=False,
                                         LayerNorm_type='BiasFree',qk_norm=qk_norm,N=N//4,path_emb_dim=path_emb_dim*4)



        self.up2_1 = Upsample(int(dim[1]),dim[0])  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.patch_embed_decoder_level1 = Patch_Embed_stage(dim[1], dim[1], num_path=num_path[0],
                                                            isPool=False,offset_clamp=offset_clamp)
        self.decoder_level1 = MHCA_stage(dim[1], dim[1], num_layers=num_blocks[0], num_heads=heads[0],
                                         ffn_expansion_factor=2.66, num_path=num_path[0], bias=False,
                                         LayerNorm_type='BiasFree',qk_norm=qk_norm,N=N,path_emb_dim=path_emb_dim*4)


        self.patch_embed_refinement = Patch_Embed_stage(dim[1], dim[1], num_path=num_path[0],
                                                        isPool=False,offset_clamp=offset_clamp)
        self.refinement = MHCA_stage(dim[1], dim[1], num_layers=num_blocks[0], num_heads=heads[0],
                                     ffn_expansion_factor=2.66, num_path=num_path[0], bias=False,
                                     LayerNorm_type='BiasFree',qk_norm=qk_norm,N=N,path_emb_dim=path_emb_dim*4)


        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim[0], dim[1], kernel_size=1, bias=bias)
        ###########################


        self.output = nn.Sequential(
            nn.Conv2d(dim[1], 3, kernel_size=3, stride=1, padding=1, bias=False, ),
        )

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        inp_enc_level1_list = self.patch_embed_encoder_level1(inp_enc_level1)

        out_enc_level1 = self.encoder_level1(inp_enc_level1_list)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        inp_enc_level2_list = self.patch_embed_encoder_level2(inp_enc_level2)
        out_enc_level2 = self.encoder_level2(inp_enc_level2_list) + inp_enc_level2

        inp_enc_level3 = self.down2_3(out_enc_level2)

        inp_enc_level3_list = self.patch_embed_encoder_level3(inp_enc_level3)
        out_enc_level3 = self.encoder_level3(inp_enc_level3_list) + inp_enc_level3

        inp_enc_level4 = self.down3_4(out_enc_level3)

        inp_latent = self.patch_embed_latent(inp_enc_level4)
        latent = self.latent(inp_latent) + inp_enc_level4

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        inp_dec_level3_list = self.patch_embed_decoder_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3_list) + inp_dec_level3

        inp_dec_level2 = self.up3_2(out_dec_level3)

        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)

        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        inp_dec_level2_list = self.patch_embed_decoder_level2(inp_dec_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2_list) + inp_dec_level2

        inp_dec_level1 = self.up2_1(out_dec_level2)

        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)


        inp_dec_level1_list = self.patch_embed_decoder_level1(inp_dec_level1)

        out_dec_level1 = self.decoder_level1(inp_dec_level1_list) + inp_dec_level1

        inp_latent_list = self.patch_embed_refinement(out_dec_level1)

        out_dec_level1 = self.refinement(inp_latent_list) + out_dec_level1

        # nn.Hardswish()

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


from thop import profile
from thop import clever_format
import time
if __name__ == "__main__":
    # Create a random input tensor
    x = torch.rand(1, 3, 128, 128).cuda()

    # Initialize the model and move it to GPU
    model = MB_TaylorFormer().cuda()
    print(f"Number of parameters: {count_param(model)}")

    # Set the model to evaluation mode
    model.eval()


    # Measure inference time
    num_runs = 10  # Number of runs to average the inference time
    total_time = 0.0

    for _ in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            _ = model(x)
        end_time = time.time()
        total_time += (end_time - start_time)

    avg_inference_time = total_time / num_runs
    print(f"Average inference time per image: {avg_inference_time:.6f} seconds")

