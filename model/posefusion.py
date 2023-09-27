import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from model.pspnet import PSPNet
from model.pointae import PointCloudAE
from model.model_utils import *
from model.model_utils import simplified_attention_forward, attn_diverse_loss
from einops import rearrange

######## UTILS ########
class ModifiedResnet(nn.Module):
    def __init__(self, out_dim=128):
        super(ModifiedResnet, self).__init__()
        self.model = PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18', pretrained=True, out_dim=out_dim)

    def forward(self, x):
        x = self.model(x)
        return x

class BaseFormer(nn.Module):
    def __init__(self, feat_dim, embedding_dim, n_heads=2, n_layers=1, encoder_type='pytorch', require_adl=False):
        super().__init__()
        self.encoder_type = encoder_type
        self.n_layers = n_layers
        self.transformer = Transformer(
            embedding_dim, 
            n_heads=n_heads, 
            n_layers=n_layers, 
            feedforward_dim=512,
            dropout=0.1
        )
        self.linear_projection = nn.Conv1d(feat_dim , embedding_dim, 1)
        # A Layernorm and a Linear layer are applied on the encoder embeddings
        self.norm = nn.LayerNorm(embedding_dim)
        self.require_adl = require_adl

    def forward(self, feat):
        embeddings = self.linear_projection(feat).transpose(2, 1).contiguous() 
        if not self.require_adl:
            embeddings = self.transformer(embeddings)
        else:
            attn_map_list = []
            for i, encoder_layer in enumerate(self.transformer.transformer.layers): 
                attn_layer = encoder_layer.self_attn
                query = key = value = embeddings.transpose(1, 0)
                x, attn_map = simplified_attention_forward(
                    query, key, value, attn_layer.num_heads,
                    attn_layer.in_proj_weight, attn_layer.in_proj_bias, only_attn=False)
                embeddings = encoder_layer.norm1(embeddings + encoder_layer.dropout1(x))
                embeddings = encoder_layer.norm2(embeddings + encoder_layer._ff_block(embeddings))

                attn_map_list.append(attn_map)

            adl = attn_diverse_loss(attn_map_list)

        embeddings = self.norm(embeddings) #B,seq_len,emb_dim 
        return embeddings if not self.require_adl else (embeddings, adl)

    def get_attention_map(self, feat, layer_id=None):
        if self.encoder_type == 'customized':
            raise NotImplementedError
        if self.encoder_type == 'pytorch':
            if layer_id is None:
                layer_id = self.n_layers - 1
            if layer_id >= self.n_layers or layer_id < 0:
                raise ValueError(
                    f"Provided layer_id: {layer_id} is not valid. 0 <= {layer_id} < {self.n_layers}."
                )
            x = self.linear_projection(feat).transpose(2, 1).contiguous()
            for i, encoder_layer in enumerate(self.transformer.transformer.layers): 
                if i < layer_id:
                    x = encoder_layer(x)
                else:
                    attn_layer = encoder_layer.self_attn
                    # for k, v in attn_layer.state_dict().items():
                    #     print(k)
                    query = key = value = x.transpose(1, 0)
                    attn_map = simplified_attention_forward(
                        query, key, value, attn_layer.num_heads,
                        attn_layer.in_proj_weight, attn_layer.in_proj_bias)
                    return attn_map



class PosePredictor(nn.Module):
    def __init__(self, feat_dim, num_points, num_obj):
        super(PosePredictor, self).__init__()
        self.num_points = num_points
        self.num_obj = num_obj
        #pose predictor
        self.conv_r = MLPs(feat_dim, num_obj*4) #quaternion
        self.conv_t = MLPs(feat_dim, num_obj*3) #translation
        self.conv_c = MLPs(feat_dim, num_obj*1) #confidence
        
    def forward(self, ap_x, obj):
        bs, _, _ = ap_x.size()
        rx = self.conv_r(ap_x).view(bs, self.num_obj, 4, self.num_points)
        tx = self.conv_t(ap_x).view(bs, self.num_obj, 3, self.num_points)
        cx = torch.sigmoid(self.conv_c(ap_x)).view(bs, self.num_obj, 1, self.num_points)
        
        b = 0
        out_rx = torch.index_select(rx[b], 0, obj[b]).contiguous().transpose(2, 1).contiguous()
        out_tx = torch.index_select(tx[b], 0, obj[b]).contiguous().transpose(2, 1).contiguous()
        out_cx = torch.index_select(cx[b], 0, obj[b]).contiguous().transpose(2, 1).contiguous()
        return out_rx, out_tx, out_cx

######## PoseFusion ########
class FusionBlock(nn.Module):
    def __init__(self, base_latent, embed_dim, n_layer1, n_layer2, require_adl=False):
        super(FusionBlock, self).__init__()
        self.embed_dim = embed_dim
        self.require_adl = require_adl

        if n_layer1>0:
            self.modality_fusion = BaseFormer(base_latent, base_latent, 4, n_layer1, require_adl=require_adl)
            modality_dim = 4*base_latent
        else:
            self.modality_fusion = None
            modality_dim = 2*base_latent
        self.point_fusion = BaseFormer(modality_dim, embed_dim, 8, n_layer2, require_adl=require_adl) if n_layer2>0 else None
        assert (self.modality_fusion is not None or self.point_fusion is not None), \
                "<Error Message> Either the number of modality or point-wise fusion layer \
                    should be larger than 0. \
                    Otherwise, the fusion block would be empty."

    def forward(self, rgb_emb, pt_emb, hidden_state=None, require_attn=False):
        cross_feat = None
        global_feat = None

        if self.modality_fusion is not None:
            if hidden_state is None:
                hidden_state = torch.cat([rgb_emb, pt_emb], dim=2)
            else:
                hidden_state = rearrange(hidden_state, 'b (k d) n -> b d (k n)', k=2)
                hidden_state += torch.cat([rgb_emb, pt_emb], dim=2)
            feat = self.modality_fusion(hidden_state)
            if self.require_adl:
                adl = feat[1]
                feat = feat[0]
            if require_attn: 
                attn1 = self.modality_fusion.get_attention_map(hidden_state)
            cross_feat = rearrange(feat, 'b (k n) d -> b n (k d)', k=2).transpose(2, 1).contiguous()

        if self.point_fusion is not None:
            feat = torch.cat([rgb_emb, pt_emb, cross_feat], dim=1) if cross_feat is not None else torch.cat([rgb_emb, pt_emb], dim=1)
            global_feat = self.point_fusion(feat)
            if self.require_adl:
                adl += global_feat[1]
                global_feat = global_feat[0]
                
            global_feat = global_feat.transpose(2, 1).contiguous()
            if require_attn: 
                attn2 = self.point_fusion.get_attention_map(feat)
            # global_feat_ = F.adaptive_max_pool1d(global_feat, 1).view(-1, self.embed_dim, 1).repeat(1, 1, num_points)
        if require_attn:
            return cross_feat, global_feat, attn1, attn2
        if self.require_adl:
            return cross_feat, global_feat, adl
        return cross_feat, global_feat 
    
class PoseFusion(nn.Module):
    def __init__(self, base_latent, embed_dim, n_block, n_layer1, n_layer2, require_adl=False):
        super(PoseFusion, self).__init__()
        self.n_block = n_block
        self.fusion_mode = (n_layer1 * n_layer2 > 0)
        self.require_adl = require_adl
        self.layers = self._make_layer(base_latent, embed_dim, n_layer1, n_layer2, require_adl)
        assert embed_dim == 2 * base_latent
        
    def forward(self, rgb_emb, pt_emb):
        if self.fusion_mode:
            result = None
            adl = 0
            for i, layer in enumerate(self.layers):
                if not self.require_adl:
                    cross_feat, global_feat = layer(rgb_emb, pt_emb)
                else:
                    cross_feat, global_feat, adl_ = layer(rgb_emb, pt_emb)
                    adl += adl_
                result = torch.cat([result, cross_feat, global_feat], dim=1) if result is not None else torch.cat([cross_feat, global_feat], dim=1)
        else:
            assert self.n_block == 1, \
                "<Warning> You only have 1 fusion stage. Please merge all layers into 1 block. To fix it, you can set n_block=1"
            cross_feat, global_feat = self.layers[0](rgb_emb, pt_emb)
            result = cross_feat if global_feat is None else global_feat
        return result if not self.require_adl else (result, adl)
    
    def _make_layer(self, base_latent, embed_dim, n_layer1, n_layer2, require_adl):
        layers = []
        for i in range(0, self.n_block):
            layers.append(FusionBlock(base_latent, embed_dim, n_layer1, n_layer2, require_adl=require_adl))
        return nn.Sequential(*layers)


######## PoseNet ########
class PoseNet(nn.Module):
    def __init__(self, num_points, num_obj, \
                 base_latent=256, embedding_dim=512, fusion_block_num=1, layer_num_m=2, layer_num_p=4, \
                 filter_enhance=True, require_adl=True):
        super(PoseNet, self).__init__()
        self.num_points = num_points
        self.num_obj = num_obj
        self.base_latent = base_latent
        self.embedding_dim = embedding_dim
        self.fusion_block_num = fusion_block_num
        self.layer_num_m = layer_num_m
        self.layer_num_p = layer_num_p
        self.fusion_mode = (layer_num_m*layer_num_p>0)
        self._init_config_check()
        
        # unimodal embedding
        self.cnn = ModifiedResnet(base_latent)
        self.ptnet = PointCloudAE(256, num_points, base_latent)
        self.filter_enhance = None if not filter_enhance else FilterLayer(num_points, base_latent, 0.0)
        
        # modality and position interaction
        self.fusion = PoseFusion(base_latent, embedding_dim, \
                                 fusion_block_num, layer_num_m, layer_num_p, \
                                 require_adl)
        self.require_adl = require_adl
        # prediction
        if self.fusion_mode:
            self.posepred = PosePredictor(base_latent * 4, num_points, num_obj)
        else:
            self.posepred = PosePredictor(base_latent * 2, num_points, num_obj)

    def _init_config_check(self):
        print("ResNet &PtNet Output Dim:", self.base_latent, '\n',\
              "Fusion Block Num:", self.fusion_block_num, '\n',\
              "Modality Fusion Layer Num:", self.layer_num_m, '\n',\
              "Point2point Fusion Layer Num:", self.layer_num_p)
        
    def forward(self, img, x, choose, obj, recon_ref=None):
        # rgb color embedding
        out_img = self.cnn(img) 
        bs, di, _, _ = out_img.size()
        emb = out_img.view(bs, di, -1)

        # selection of rgb color embedding
        choose = choose.repeat(1, di, 1) 
        rgb_emb = torch.gather(emb, 2, choose).contiguous()

        # depth map / point cloud (embedding)
        pt_feat, pt_emb, pt_recon, extra_loss = self.ptnet(x, None, recon_ref)
        pt_emb = self.ptnet.latent(pt_feat, pt_emb)
        if self.filter_enhance is not None:
            pt_emb = self.filter_enhance(pt_emb)
        
        feat = self.fusion(rgb_emb, pt_emb) 
        if self.require_adl:
            extra_loss += feat[1] 
            feat = feat[0]
            
        out_rx, out_tx, out_cx = self.posepred(feat, obj)

        return out_rx, out_tx, out_cx, rgb_emb.detach(), pt_recon.detach(), extra_loss
    
    def get_attention_map(self, img, x, choose):
        # rgb color embedding
        out_img = self.cnn(img) 
        bs, di, _, _ = out_img.size()
        emb = out_img.view(bs, di, -1)

        # selection of rgb color embedding
        choose = choose.repeat(1, di, 1) 
        rgb_emb = torch.gather(emb, 2, choose).contiguous()

        # depth map / point cloud (embedding)
        pt_feat, pt_emb, _, _ = self.ptnet(x, None, None)
        pt_emb = self.ptnet.latent(pt_feat, pt_emb)
        if self.filter_enhance is not None:
            pt_emb = self.filter_enhance(pt_emb)
        # feat = torch.cat([rgb_emb, pt_emb], dim=2)
        _, _, attn1, attn2 = self.fusion.layers[0](rgb_emb, pt_emb, require_attn=True)
    
        return attn1, attn2
 