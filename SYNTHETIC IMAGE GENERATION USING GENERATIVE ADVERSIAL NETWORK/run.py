import torch
from model.GAN import Discriminator 
from model.GAN import Generator
from data.data_utils import get_dl
from train import train_model

torch.manual_seed(4)
batch_size = 128
no_of_epochs = 5
input_size = 100
train_loader,test_loader = get_dl(batch_size)
dl = {}
dl['train'] = train_loader
dl['valid'] = test_loader
disc  = Discriminator(batch_size)
gen = Generator(batch_size,input_size)
optimD  = torch.optim.Adam(disc.parameters(),lr=0.001,weight_decay=1e-05)
optimG  = torch.optim.Adam(gen.parameters(),lr=0.001,weight_decay=1e-05)
loss_fn = torch.nn.BCELoss()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
disc.to(device)
gen.to(device)
train_model(no_of_epochs,disc,gen,optimD,optimG,dl,loss_fn,input_size,batch_size)