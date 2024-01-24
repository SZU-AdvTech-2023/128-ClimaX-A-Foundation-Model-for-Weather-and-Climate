# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import lru_cache

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed, trunc_normal_

from climax.utils.pos_embed import (
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
)

from .parallelpatchembed import ParallelVarPatchEmbed
from climax.utils.Galerkin.model import make_GalerkinTransformerDeocder
from climax.utils.CBR_Decoder import make_decoder
from climax.AutoEncoder.autoencoder import AE

class ClimaX(nn.Module):
    """Implements the ClimaX model as described in the paper,
    https://arxiv.org/abs/2301.10343

    Args:
        default_vars (list): list of default variables to be used for training
        img_size (list): image size of the input data
        patch_size (int): patch size of the input data
        embed_dim (int): embedding dimension
        depth (int): number of transformer layers
        decoder_depth (int): number of decoder layers
        num_heads (int): number of attention heads
        mlp_ratio (float): ratio of mlp hidden dimension to embedding dimension
        drop_path (float): stochastic depth rate
        drop_rate (float): dropout rate
        parallel_patch_embed (bool): whether to use parallel patch embedding
    """

    def __init__(
        self,
        default_vars,
        img_size=[32, 64],
        patch_size=2,
        embed_dim=1024,
        depth=8,
        decoder_depth=2,
        num_heads=16,
        mlp_ratio=4.0,
        drop_path=0.1,
        drop_rate=0.1,
        parallel_patch_embed=False,
    ):
        super().__init__()

        # TODO: remove time_history parameter
        self.img_size = img_size
        self.patch_size = patch_size
        self.default_vars = default_vars
        self.parallel_patch_embed = parallel_patch_embed

        """our change"""
        self.root_dir = r'D:\mnt\data\5.625deg_npz'
        self.top_k = 5
        self.ae = AE(c=32)
        self.ae.load_state_dict(torch.load(r'D:\Research\Race\ClimaX\best_model_ae_last__kkkkkk30.pth'))
        # load ae model

    


        def load_kb(start, end, KB_FILE):
            skiprows = 8592
            df = pd.read_csv(KB_FILE, nrows=skiprows*(end-start), skiprows=start*skiprows)
            return torch.tensor(np.array(df))
        self.kb = []
        self.years = 9
        self.years1 = 9
        self.years2 = 9
        print("Loading KB")
        for i in range(self.years):
            print("year: ",i)
            t = load_kb(i ,i+1, 'D:\Research\Race\ClimaX\kb_2048.csv')
            t = t.to('cuda:0')
            t = t.float()
        
            self.kb.append(t)


        for i in range(self.years):
            print("year: ",i + self.years)
            t = load_kb(i ,i+1, 'D:\Research\Race\ClimaX\kb_2048_1.csv')
            t = t.to('cuda:0')
            t = t.float()
        
            self.kb.append(t)


        for i in range(self.years):
            print("year: ",i + self.years)
            t = load_kb(i ,i+1, 'D:\Research\Race\ClimaX\kb_2048_2.csv')
            t = t.to('cuda:0')
            t = t.float()
        
            self.kb.append(t)

        print("Loaded KB")

        # variable tokenization: separate embedding layer for each input variable
        if self.parallel_patch_embed:
            self.token_embeds = ParallelVarPatchEmbed(len(default_vars), img_size, patch_size, embed_dim)
            self.num_patches = self.token_embeds.num_patches
        else:
            self.token_embeds = nn.ModuleList(
                [PatchEmbed(img_size, patch_size, 1, embed_dim) for i in range(len(default_vars))]
            )
            self.num_patches = self.token_embeds[0].num_patches

        # variable embedding to denote which variable each token belongs to
        # helps in aggregating variables
        self.var_embed, self.var_map = self.create_var_embedding(embed_dim)
        # variable aggregation: a learnable query and a single-layer cross attention
        self.var_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.var_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # positional embedding and lead time embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=True)
        self.lead_time_embed = nn.Linear(1, embed_dim)

        # --------------------------------------------------------------------------

        # ViT backbone
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    drop_path=dpr[i],
                    norm_layer=nn.LayerNorm,
                    drop=drop_rate,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # --------------------------------------------------------------------------
        # our decoder

        self.CLimax_Decoder = make_decoder(N=1)
        
        
        # --------------------------------------------------------------------------

        # prediction head
        
        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Linear(embed_dim, embed_dim))
            self.head.append(nn.LayerNorm(1024))
        self.head.append(nn.Linear(embed_dim, len(self.default_vars) * patch_size**2))
        self.head = nn.Sequential(*self.head)

        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialize pos_emb and var_emb
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.img_size[0] / self.patch_size),
            int(self.img_size[1] / self.patch_size),
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        var_embed = get_1d_sincos_pos_embed_from_grid(self.var_embed.shape[-1], np.arange(len(self.default_vars)))
        self.var_embed.data.copy_(torch.from_numpy(var_embed).float().unsqueeze(0))

        # token embedding layer
        if self.parallel_patch_embed:
            for i in range(len(self.token_embeds.proj_weights)):
                w = self.token_embeds.proj_weights[i].data
                trunc_normal_(w.view([w.shape[0], -1]), std=0.02)
        else:
            for i in range(len(self.token_embeds)):
                w = self.token_embeds[i].proj.weight.data
                trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def create_var_embedding(self, dim):
        var_embed = nn.Parameter(torch.zeros(1, len(self.default_vars), dim), requires_grad=True)
        # TODO: create a mapping from var --> idx
        var_map = {}
        idx = 0
        for var in self.default_vars:
            var_map[var] = idx
            idx += 1
        return var_embed, var_map

    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device):
        #my change
        ids = np.array([self.var_map[var] for var in vars])
        return torch.from_numpy(ids).type(torch.long).to(device)

    def get_var_emb(self, var_emb, vars):
        ids = self.get_var_ids(vars, var_emb.device)
        return var_emb[:, ids, :]

    def unpatchify(self, x: torch.Tensor, h=None, w=None):
        """
        x: (B, L, V * patch_size**2)
        return imgs: (B, V, H, W)
        """
        p = self.patch_size
        c = len(self.default_vars)
        h = self.img_size[0] // p if h is None else h // p
        w = self.img_size[1] // p if w is None else w // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def aggregate_variables(self, x: torch.Tensor):
        """
        x: B, V, L, D
        """
        b, _, l, _ = x.shape
        x = torch.einsum("bvld->blvd", x)
        x = x.flatten(0, 1)  # BxL, V, D

        var_query = self.var_query.repeat_interleave(x.shape[0], dim=0)
        x, _ = self.var_agg(var_query, x, x)  # BxL, D
        x = x.squeeze()

        x = x.unflatten(dim=0, sizes=(b, l))  # B, L, D
        return x

    def forward_encoder(self, x: torch.Tensor, lead_times: torch.Tensor, variables):
        # x: `[B, V, H, W]` shape.

        if isinstance(variables, list):
            variables = tuple(variables)

        # tokenize each variable separately
        embeds = []
        var_ids = self.get_var_ids(variables, x.device)

        if self.parallel_patch_embed:
            x = self.token_embeds(x, var_ids)  # B, V, L, D
        else:
            for i in range(len(var_ids)):
                id = var_ids[i]
                embeds.append(self.token_embeds[id](x[:, i : i + 1]))
            x = torch.stack(embeds, dim=1)  # B, V, L, D

        # add variable embedding
        var_embed = self.get_var_emb(self.var_embed, variables)
        x = x + var_embed.unsqueeze(2)  # B, V, L, D

        # variable aggregation
        x = self.aggregate_variables(x)  # B, L, D

        # add pos embedding
        x = x + self.pos_embed

        # add lead time embedding
        lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1))  # B, D
        lead_time_emb = lead_time_emb.unsqueeze(1)
        x = x + lead_time_emb  # B, L, D

        x = self.pos_drop(x)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

# ---------------------------------------------------------------------
# "our change"
# ---------------------------------------------------------------------  

    # def load_kb_512_1024(self, vals, ids, start, end):
        
    #     res = IndividualForecastDataIter(
    #                 Forecast(
    #                     myNpyReader(
    #                         file_list=self.origin_train,
    #                         start_idx=0,
    #                         end_idx=10,
    #                         variables=self.variables,
    #                         out_variables=self.out_variables,
    #                         shuffle=False,
    #                         multi_dataset_training=False,
    #                     ),
    #                     max_predict_range=168,
    #                     random_lead_time=False,
    #                     hrs_each_step=1,
    #                 ),
    #                 transforms=self.transforms,
    #                 output_transforms=self.output_transforms,
    #             )
        
    #     res = DataLoader(
    #             self.v,
    #             batch_size=1,
    #             shuffle=False,
    #             drop_last=False,
    #             num_workers=1,
    #             pin_memory=False,
    #             collate_fn=collate_fn,
    #         )
    #     id = ids[0].tolist()
    #     id = [x + 1 for x in id]
    #     id_max = max(id)



    #     vals = vals[0].tolist()
    #     st = time.time()
    #     for i, batch in enumerate(res):
    #         #if self.kb_length > 10: breaks
            
    #         x, y, lead_times, variables, out_variables = batch
    #         x = x.to('cuda:0')
    #         lead_times = lead_times.to('cuda:0')
            
    #         if i > id_max: break
    #         if i in id:
    #             x_embedding = self.forward_encoder(x,lead_times, variables)
                
    #             # x_embedding = self.dimension_reduction(x_embedding, "cls")
    #             self.val2tensor[vals[id.index(i)]] = x_embedding
    #     print(time.time() - st)
        




    def dimension_reduction(self, x:torch.tensor, method: str):
        
        
        if method == "max_pool":
            #调整一下维度
            max_pooling_layer = nn.MaxPool2d(kernel_size=2, stride=2)
            return max_pooling_layer(x)
        elif method == "avg_pool":
            avg_pooling_layer = nn.AvgPool2d(kernel_size=2, stride=2)
            return avg_pooling_layer(x)
        elif method == "cls":
            return x[:, 0, :]  
        elif method == "AE":
            x = x.view(x.size(0), 512, 32 ,32)
            return self.ae.encode(x).view(x.size(0),-1).float()
    
    

    def retrieval_0(self, x: torch.Tensor): 
        x = self.dimension_reduction(x, "cls")
        
        

        val2id = {}
        for year in range(11):
            #TODO: read a year one time:
            #TODO: convert self.kb to tensor:
            self.kb = self.load_kb(year, year+1)
            
            
            self.kb = self.kb.to('cuda:0')
        
            ans = torch.matmul(self.kb, x.T) #[n, 1024] * [1024 * 1]
            ans = ans.T
            # print(ans.shape)
            vals, ids = torch.topk(ans, dim=-1, largest=True, sorted=True, k=self.top_k + 1)
            for i in range(vals.size(1)):
                if int(ids[0, i]) < 8591:
                    val2id[vals[0, i]] = self.kb[:,int(ids[0, i]) + 1]
                else: continue
            del self.kb
        
        keys = list(val2id.keys())
        keys.sort() 
        top_k = keys[:self.top_k]
        return [val2id[k] for k in top_k]
        
    def retrieval_1(self, x: torch.Tensor):
        
        x = self.dimension_reduction(x, "cls")
        
        val2tensor_batch = []
        for i in range(x.size(0)):
            val2tensor = {}
            for year in range(10):
                #TODO: read a year one time:
                #TODO: convert self.kb to tensor:
                
                

                ans = torch.matmul(self.kb[year], x[i].unsqueeze(0).T) #[n, 1024] * [1024, 1]
                # print(x[i].unsqueeze(0).T.shape)
                ans = ans.T
                
                vals, ids = torch.topk(ans, dim=-1, largest=True, sorted=True, k=self.top_k + 1)
    
                for j in range(vals.size(0)):
                    if int(ids[0][j]) < 8592 - 1:
                        val2tensor[vals[0][j]] = self.kb[year][int(ids[0][j]) + 1,:]
                    else: 
                        continue
                del self.kb

            keys = list(val2tensor.keys())
            keys.sort()
            top_k = keys[:self.top_k]
            val2tensor_batch.append([val2tensor[k] for k in top_k])
        
        
        return val2tensor_batch
    
    def retrieval_512(self, x: torch.Tensor):
        
        x = self.dimension_reduction(x, "cls")
        
        val2id_batch = []
        
        self.val2tensor = {}
        for i in range(x.size(0)):
            self.val2tensor = {}
            for year in range(10):
                #TODO: read a year one time:
                #TODO: convert self.kb to tensor:

                ans = torch.matmul(self.kb[year], x[i].unsqueeze(0).T) #[n, 1024] * [1024, 1]
                ans = ans.T
                
                vals, ids = torch.topk(ans, dim=-1, largest=True, sorted=True, k=self.top_k + 1)
                
                # kb_512 = self.load_kb_512_1024(year, year + 1)
                # for j in range(vals.size(0)):
                    # if int(ids[0][j]) < 8592 - 1:
                        # val2id[vals[0][j]] = kb_512[int(ids[0][j]) + 1,:].reshape(512, 1024)
                self.load_kb_512_1024(vals, ids, year, year+1)
                    # else: 
                    #     continue
                

            keys = list(self.val2tensor.keys())
            keys.sort()
            top_k = keys[:self.top_k]
            val2id_batch.append([self.val2tensor[k] for k in top_k])
        
        return val2id_batch

    def retrieval(self, x:torch.tensor):
        
        xn= self.dimension_reduction(x, "AE")
        x = self.dimension_reduction(x, "AE")
        xn = nn.functional.normalize(xn, p=2, dim=1)

        # x: [B, 2048]

        

        val2y_batch = []
        val2x_batch = []
        for _ in range(x.size(0)):
            val2y_batch.append({})
            val2x_batch.append({})

        for year in range(self.years + self.years1 + self.years2):
            #TODO: read a year one time:
            #TODO: convert self.kb to tensor:
            kb = self.kb[year]
            kb = nn.functional.normalize(kb, p=2, dim=1)
            ans = torch.matmul(xn, kb.T) #[n, 2048] * [2048, B]
            # print(x[i].unsqueeze(0).T.shape)

            
            # ans = ans.T #[B, n]
            
            
            vals, ids = torch.topk(ans, dim=1, largest=True, sorted=True, k=self.top_k + 1)
            #[B,top_k]
           

            for i in range(vals.size(0)):
                val2y = {}
                val2x = {}
                for j in range(vals.size(1)):
                    if int(ids[i][j]) < 8592 - 1:
                        val2y[vals[i][j]] = self.kb[year][int(ids[i][j]) + 1,:]
                        val2x[vals[i][j]] = self.kb[year][int(ids[i][j]),:]
                    else: 
                        continue
        
                keys = list(val2y.keys())
                keys.sort()
                top_k = keys[:self.top_k]

                key = list(val2x.keys())
                key.sort()
                top_kk = key[:self.top_k] 

                # print(top_k == top_kk)

                for k in top_k:
                    val2y_batch[i][k] = val2y[k]
                for k in top_kk:
                    val2x_batch[i][k] = val2x[k]



        ans = []
        res = []
        for _ in range(x.size(0)):
            ans.append([])
            res.append([])

        out_top_k = []
        for i in range(len(val2y_batch)):
            keys = list(val2y_batch[i].keys())
            keys.sort()
            top_k = keys[:self.top_k]
            out_top_k

            key = list(val2x_batch[i].keys())
            key.sort()
            top_kk = key[:self.top_k]

            
            for k in top_k:
                ans[i].append(val2y_batch[i][k])
            for k in top_kk:
                res[i].append(val2x_batch[i][k])

        b = []
        for i in ans: 
            b.append(torch.stack(i))
        c = []
        for i in res: 
            c.append(torch.stack(i))
        
        return xn, torch.stack(top_k), torch.stack(c), torch.stack(b) #  [B, top_k, 2048]


    def decode_kb(self,kb_x:torch.tensor, kb_y:torch.tensor):
        batch_n = kb_x.size(0)
        kb_x = kb_x.view(-1, 32, 8, 8)
        kb_y = kb_y.view(-1, 32, 8, 8)
        kb_x = self.ae.decode(kb_x).float()
        kb_y = self.ae.decode(kb_y).float()
        kb_x = kb_x.view(batch_n,self.top_k, 512, -1)
        kb_y = kb_y.view(batch_n,self.top_k, 512, -1)
        return kb_x.float(), kb_y.float() #[B, top_k, L， D]

    def fusion_1(self, q: torch.tensor, c: torch.tensor):
        """fuse the emprirical process
        q: [B*L*D]
        cases: [B*D*top_k] 
        Returns: fusing embedding
        """
        
        # 初始化输出tensor
        # output = torch.zeros_like(q) 
        q = self.W1(q)
        c = self.W2(c).transpose(1, 2) #[B*L*D]
        
       
        t = nn.GELU()(q + c)
        # t = nn.LayerNorm(t)
        t = self.batch_norm(t)
        t = self.W3(t)
        
        t = self.head(t)
        # for i in range(q.shape[0]):
        #     # 遍历batch维度
        #     for c in cases:
        #         # 将cases一一加到q对应的sample上
        #         q[i] += c[i] 
        
        return t



    def forward(self, x, y, lead_times, variables, out_variables, metric, lat):
        """Forward pass through the model.

        Args:
            x: `[B, Vi, H, W]` shape. Input weather/climate variables
            y: `[B, Vo, H, W]` shape. Target weather/climate variables
            lead_times: `[B]` shape. Forecasting lead times of each element of the batch.

        Returns:
            loss (list): Different metrics.
            preds (torch.Tensor): `[B, Vo, H, W]` shape. Predicted weather/climate variables.
        """
        with torch.no_grad():
            out_transformers = self.forward_encoder(x, lead_times, variables)  # B, L, D
            out_x, top_k, sim_x, sim_y = self.retrieval(out_transformers) # B * topk * L, D
            #sim_x, sim_y = self.decode_kb(sim_x, sim_y) # B * topk * L, D

        out_transformers = self.CLimax_Decoder(out_transformers, out_x, top_k, sim_x, sim_y)
        

        preds = self.head(out_transformers)  # B, L, V*p*p


        preds = self.unpatchify(preds)
        out_var_ids = self.get_var_ids(tuple(out_variables), preds.device)
        preds = preds[:, out_var_ids]

        if metric is None:
            loss = None
        else:
            loss = [m(preds, y, out_variables, lat) for m in metric]

        return loss, preds

    def evaluate(self, x, y, lead_times, variables, out_variables, transform, metrics, lat, clim, log_postfix):
        _, preds = self.forward(x, y, lead_times, variables, out_variables, metric=None, lat=lat)
        return [m(preds, y, transform, out_variables, lat, clim, log_postfix) for m in metrics]


