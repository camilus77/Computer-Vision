import torch

#function to train the GAN model
def train_model(no_of_epochs,disc,gen,optimD,optimG,dataloaders,loss_fn,input_size,batch_size):
    """
    disc: Discriminator model
    gen: Generator model
    optimD: Optimizer for Discriminator
    optimG: Optimizer for Generator
    """
    
    # setting the device as cuda or cpu
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    reall = 1  # real label
    fakel = 0  # fake label
    #running each epoch
    for epoch in range(no_of_epochs):
        print('Epoch {}/{}'.format(epoch+1,no_of_epochs))
        running_loss_D = 0
        running_loss_G = 0
        for phase in ["train"]:
            #getting input and label from dataloader
            for inputs, _ in dataloaders[phase]:
                inputs = inputs.to(device)
                #converting labels into torch with proper size as per the batch size
                real_label = torch.full((batch_size,),reall,dtype=inputs.dtype,device=device)
                fake_label = torch.full((batch_size,),fakel,dtype=inputs.dtype,device=device)

                optimD.zero_grad()
                #output from discriminator
                output =disc(inputs)
                #Discriminator real loss
                # Compairing output label with real label which is loss
                D_real_loss = loss_fn(output,real_label)
                D_real_loss.backward()
                
                #random torch tensor as a noise data
                noise = torch.randn(batch_size,input_size,device=device)
                #passing noise throgh generator to get fake image
                fake = gen(noise)
                #passing fake image through discriminator with detaching(not passing gradient)
                output = disc(fake.detach())
                
                #Discriminator fake loss
                D_fake_loss = loss_fn(output,fake_label)
                #back propogation
                D_fake_loss.backward()

                # total loss for Discriminator
                Disc_loss = D_real_loss+D_fake_loss
                running_loss_D = running_loss_D+Disc_loss
                optimD.step()

                optimG.zero_grad()
                #passing fake image obtained from generator to discriminator
                output = disc(fake)
                #getting generator loss by giving fake image as input but giving real label
                Gen_loss = loss_fn(output,real_label)
                running_loss_G = running_loss_G + Gen_loss
                #backpropogation
                Gen_loss.backward()
                optimG.step()
        print("Discriminator Loss : {}".format(running_loss_D))
        print("Generator Loss : {}".format(running_loss_G))


