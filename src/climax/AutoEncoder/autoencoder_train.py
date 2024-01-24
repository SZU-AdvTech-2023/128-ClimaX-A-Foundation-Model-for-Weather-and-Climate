from climax.KB_construction import KnowledgeBase_Construction
from climax.AutoEncoder.autoencoder import AE
import torch
import torch.nn as nn
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm
import logging



class Trainer_(nn.Module):
    def __init__(self,c,kb):
        super(Trainer_,self).__init__()
        
        # self.model = AE(c=c)
        # self.model.load_state_dict(torch.load(r'D:\Research\Race\ClimaX\best_model_ae_last__kkkkkk30.pth'))
        # self.model = self.model.to("cuda:0")
        self.kb = kb
        del kb
        self.kb.data_loader.setup()
        #self.train_loader = self.kb.return_AE_train_data()
        self.train_loader = self.kb.return_AE_test_data()
        #self.use_to_count = self.train_loader
        
        self.all_len = 8592 * 36
        self.include_batch_len = 2685 * 2
        self.cur_loss = 0
        self.best_loss = 10000000
        
        #del self.use_to_count
        
        
        self.kb.freeze_Net()
        # self.optimizer = optim.Adam(self.model.parameters(), lr=1e-6)
        

    
    def loss_function_BCE(self,recon_x, x):
        BCE = F.binary_cross_entropy(torch.softmax(recon_x.flatten(), dim=-1), torch.softmax(x.flatten(), dim=-1))
        return BCE

    def loss_function_L2(self, recon_x, x):
        l2_loss = F.mse_loss(recon_x, x)
        return l2_loss
        
    def loss_function_SmoothL1(self, recon_x, x):
        smoothl1_loss = F.smooth_l1_loss(recon_x, x)
        return smoothl1_loss

    def train(self, epoch, clip_value=1.2):
        
        self.model.train()
        train_loss = 0
        logger = logging.getLogger(__name__)  
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        file_handler = logging.FileHandler('train_{}.log'.format(epoch))
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        now_batch = 0
        for batch_idx, batch in enumerate(self.train_loader):
            data, y, lead_times, variables, out_variables = batch
            data = data.to("cuda:0")
            lead_times = lead_times.to("cuda:0")
            data = self.kb.encoder(data, lead_times, variables)
            self.optimizer.zero_grad()
            data = data.view(data.size(0), 512, 32 ,32)
            recon_batch = self.model(data)

            loss = self.loss_function_SmoothL1(recon_batch, data)
            loss.backward()

            # 1120change: 梯度裁剪
            # nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)

            train_loss += loss.item()
            batch_loss = loss.item()
            self.optimizer.step()
            now_batch += data.size(0)
            
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, now_batch, self.all_len,
                now_batch *100. / self.all_len,
                batch_loss))

            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, now_batch, self.all_len,
                now_batch *100. / self.all_len,
                batch_loss))




        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss))
            
        logger.info('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss))

        

        
        
        return train_loss
        
    def train_epoch(self):

        for epoch in range(50):
            self.cur_loss = self.train(epoch)
            
            
            if self.cur_loss < self.best_loss:
                self.best_loss = self.cur_loss
                best_model_wts = self.model.state_dict() # 保存模型参数
                torch.save(best_model_wts, 'best_model_ae_last__kkkkkk{}.pth'.format(epoch)) # 可以保存到文件


    def test(self, n):

        # logger = logging.getLogger(__name__)  
        # logger.setLevel(logging.INFO)
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # file_handler = logging.FileHandler('test{}.log'.format(n))
        # file_handler.setFormatter(formatter)

        model = AE(c=32)
        model.load_state_dict(torch.load(r'D:\Research\Race\ClimaX\best_model_ae_last__kkkkkk30.pth'))
        test_smoothl1_loss = 0
        test_L2_loss = 0
        test_BCE_loss = 0
        model = model.to("cuda:0")
        count = 0
        with torch.no_grad():
            for i, batch in enumerate(self.train_loader):
                data, y, lead_times, variables, out_variables = batch
                data = data.to("cuda:0")
                lead_times = lead_times.to("cuda:0")
                data = self.kb.encoder(data, lead_times, variables)
                data = data.view(data.size(0), 512, 32 ,32)
                recon_batch = model(data)
                count += 1
                
                test_smoothl1_loss += self.loss_function_SmoothL1(recon_batch, data).item()
                
                print(self.loss_function_SmoothL1(recon_batch, data).item())
                



        print('====> model:{} test_smoothl1_loss set loss: {:.4f}'.format(n, test_smoothl1_loss / count))
        # print('====> model:{} test_L2_loss set loss: {:.4f}'.format(n,test_L2_loss / count))
        # logger.info('====> model:{} test_smoothl1_loss set loss: {:.4f}'.format(n,test_smoothl1_loss / count))
        # logger.info('====> model:{} test_L2_loss set loss: {:.4f}'.format(n,test_L2_loss / count))

        # self.cur_loss =  test_L2_loss / count
        # if self.cur_loss <= self.best_loss:
        #     best_model_wts = model.state_dict() # 保存模型参数
        #     torch.save(best_model_wts, 'best_model_ae_last_best.pth'.format()) # 可以保存到文件
        
        


if __name__ == "__main__":
    torch.cuda.empty_cache()
    print("kb loading")
    kb = KnowledgeBase_Construction()
    print("kb load")
    model_train = Trainer_(c=32,kb=kb)
    model_train.test(1)
    
    