import numpy as np
import torch
import torch.nn as nn
# from pytorch_lightning import LightningModule
from torchvision.transforms import transforms
from climax.global_forecast.datamodule import GlobalForecastDataModule
from climax.tamende_arch import ClimaX
from climax.global_forecast.module import GlobalForecastModule
from tqdm import tqdm
#from pytorch_lightning.cli import LightningCLI
# data_loader = GlobalForecastDataModule
import pandas as pd
from sklearn.decomposition import PCA
from climax.AutoEncoder.autoencoder import AE
import torch.utils.data

class KnowledgeBase_Construction():
    def __init__(
        self,
        pretrained_path: str = r'D:\Research\Race\ClimaX\${trainer.default_root_dir}\checkpoints\epoch_002.ckpt',
        embed_dim=1024,
        top_k: int=5, #k-top relative
        lead_times: int=168,
        data_path: str = "D:\\mnt\\data\\5.625deg_npz\\",
        vars: list = [
          "land_sea_mask",
          "orography",
          "lattitude",
          "2m_temperature",
          "10m_u_component_of_wind",
          "10m_v_component_of_wind",
          "geopotential_50",
          "geopotential_250",
          "geopotential_500",
          "geopotential_600",
          "geopotential_700",
          "geopotential_850",
          "geopotential_925",
          "u_component_of_wind_50",
          "u_component_of_wind_250",
          "u_component_of_wind_500",
          "u_component_of_wind_600",
          "u_component_of_wind_700",
          "u_component_of_wind_850",
          "u_component_of_wind_925",
          "v_component_of_wind_50",
          "v_component_of_wind_250",
          "v_component_of_wind_500",
          "v_component_of_wind_600",
          "v_component_of_wind_700",
          "v_component_of_wind_850",
          "v_component_of_wind_925",
          "temperature_50",
          "temperature_250",
          "temperature_500",
          "temperature_600",
          "temperature_700",
          "temperature_850",
          "temperature_925",
          "relative_humidity_50",
          "relative_humidity_250",
          "relative_humidity_500",
          "relative_humidity_600",
          "relative_humidity_700",
          "relative_humidity_850",
          "relative_humidity_925",
          "specific_humidity_50",
          "specific_humidity_250",
          "specific_humidity_500",
          "specific_humidity_600",
          "specific_humidity_700",
          "specific_humidity_850",
          "specific_humidity_925",
        ],
        out_variables: str = '2m_temperature'
        ):
        super().__init__()
        # self.save_hyperparameters(logger=False, ignore=["net"])
        self.lead_times = lead_times
        self.vars = vars
        self.net = GlobalForecastModule(ClimaX(default_vars= self.vars), pretrained_path=pretrained_path)
        self.embed_dim = embed_dim
        self.top_k = top_k
        self.data_loader = GlobalForecastDataModule(root_dir=data_path, \
                                       out_variables= out_variables, \
                                       buffer_size=1,
                                       variables=self.vars
                                        )
# data = np.load(r'D:\mnt\data\5.625deg_npz\train\climatology.npz')
# print(data.files)
        self.encoder_ae = AE(c=32)
        self.encoder_ae.load_state_dict(torch.load(r'D:\Research\Race\ClimaX\best_model_ae_last__kkkkkk30.pth'))

        self.fuck = self.data_loader.setup_kb()
        # elf.data = self.data_loader.return_lister_train()
    def encoder(self, x: torch.Tensor, lead_times: torch.Tensor, variables):
        x = self.net.net.forward_encoder(x, lead_times, variables)
        
        return x #Shape: [B, L, D]
    
    def dimension_reduction(self, x:torch.tensor, method: str):
        
        if method == "max_pool":
            #调整一下维度
            max_pooling_layer = nn.MaxPool2d(kernel_size=2, stride=2)
            return max_pooling_layer(x)
        elif method == "avg_pool":
            avg_pooling_layer = nn.AvgPool2d(kernel_size=2, stride=2)
            return avg_pooling_layer(x)
        elif method == "cls":
            return x.flatten()
        elif method == "AE":
            x = x.view(x.size(0), 512, 32 ,32)
            return self.encoder_ae.encode(x)
        elif method == "PCA":
            X = x[0]
            X = X.cpu().numpy()
            pca=PCA(n_components=1)
            pca.fit(X)
            print(pca.transform(X))
            return x

    # Yanis
    def tensor2str(self, x:torch.Tensor):
        x = x.cpu().numpy()
        x = x.tolist()
        strNums = [str(x_i) for x_i in x]
        str1 = ','.join(strNums)
        return str1
    
    def str2tensor(self, df):
        '''
        Input:
            df: csv
        Output:
            torch.Tensor
        '''
        return torch.tensor(np.array(df))
    
    def load_data(self, start, end):

        KB_FILE = 'kbb.csv'
        skiprows = 8592
        df = pd.read_csv(KB_FILE, nrows=skiprows*(end-start), skiprows=start*skiprows)
        return df

    def len_data(self):
        for p in self.net.net.parameters():
            p.requires_grad = False
        origin_datas = self.data_loader.kb_data_()
        del self.data_loader
        # x_embeddings = []
        # self.id2vec = {}
        
        self.kb_length = 0
        
        # memroy = []
        len = 0
        for batch in tqdm(origin_datas):
            #if self.kb_length > 10: break
            len += 1

        print(len)

    def KB_construction_csv(self):
        for p in self.net.net.parameters():
            p.requires_grad = False
        origin_datas = self.data_loader.kb_data_()
        del self.data_loader
        # x_embeddings = []
        # self.id2vec = {}
        
        self.kb_length = 0
        
        memroy = []
        with torch.no_grad():
            for i, batch in tqdm(enumerate(origin_datas)):
                #if self.kb_length > 10: break
                x, y, lead_times, variables, out_variables = batch
                
                x_embedding = self.encoder(x,lead_times, variables)
                x_embedding = self.dimension_reduction(x_embedding, "AE")
                x_embedding = x_embedding[0].view(-1)
                # x_embeddings.append(x_embedding)
                if i % 1000 == 0:
                    with open('./kb_2048_2.csv', 'a+') as f:
                        for _ in memroy:
                            f.write(self.tensor2str(_))
                            f.write("\n")
                    memroy = []
                else: 
                    memroy.append(x_embedding)
                self.kb_length += 1
            
            

        # for j in range(self.kb_length - 1):
        #     self.id2vec[j] = memroy[j + 1]

        # self.kb = torch.stack(x_embeddings)
        # torch.save(self.kb, "./kb.pth")
        # torch.save(self.id2vec, "./id2vec.pth")

    def KB_construction_pt(self):
        for p in self.net.net.parameters():
            p.requires_grad = False
        origin_datas = self.data_loader.kb_data_()
        del self.data_loader
        x_embeddings = []
        # self.id2vec = {}
        
        self.kb_length = 0
        
        # memroy = []
        year = 0
        for batch in tqdm(origin_datas):
            #if self.kb_length > 10: break
            x, y, lead_times, variables, out_variables = batch
            
            x_embedding = self.encoder(x,lead_times, variables)
            # x_embedding = self.dimension_reduction(x_embedding, "cls")
            x_embeddings.append(x_embedding)

            if len(x_embeddings) >= 8760 - 168:
                self.kb = torch.stack(x_embeddings)
                torch.save(self.kb, "./kb_{}.pths".format(1979 + year))
                x_embeddings = []
                year += 1
                

            self.kb_length += 1
            
            

        # for j in range(self.kb_length - 1):
        #     self.id2vec[j] = memroy[j + 1]

        
        
        # torch.save(self.id2vec, "./id2vec.pth")
    
    def freeze_Net(self):
        for p in self.net.net.parameters():
            p.requires_grad = False
    
        self.net.net = self.net.net.to("cuda:0")
    def test_dim(self):
        for p in self.net.net.parameters():
            p.requires_grad = False
        origin_datas = self.data_loader.kb_data_()
        del self.data_loader

        for i, batch in tqdm(enumerate(origin_datas)):
            #if self.kb_length > 10: break
            x, y, lead_times, variables, out_variables = batch
            

            x_embedding = self.encoder(x,lead_times, variables)
            
    def return_AE_train_data(self):
        a= self.data_loader.train_dataloader()
        del self.data_loader
        return a

    def return_AE_test_data(self):
        a= self.data_loader.test_dataloader()
        del self.data_loader
        return a

if __name__ == "__main__":
   
    kb = KnowledgeBase_Construction()
    

    kb.KB_construction_csv()