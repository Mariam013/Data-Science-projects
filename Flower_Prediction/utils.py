#installs


# Imports here
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import json
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
#import helper

         
def load_transforms():
    #transforms for the training, validation, and testing sets
    train_transform = transforms.Compose([transforms.Resize(224),
                                        transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    test_transform = transforms.Compose([transforms.Resize(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    return train_transform, test_transform

def image_datasets(train_dir, valid_dir,test_dir ,train_transform, test_transform):

   
    #image_datasets  
    train_data = datasets.ImageFolder(train_dir , transform = train_transform)
    valid_data = datasets.ImageFolder(valid_dir , transform = test_transform)
    test_data = datasets.ImageFolder(test_dir , transform = test_transform)
    #define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    return trainloader,validloader,testloader

def label_data(labels_file):
    #label mapping
    with open(labels_file, 'r') as f:
        cat_to_name = json.load(f)
        #print(cat_to_name)
        output_size = len(cat_to_name)
    return cat_to_name,output_size

def build_classifier(gpu_cpu,use_model,hidden_layers,output_size,learning_rate):
    input_size = 25088 if use_model == 'vgg16' else 2048 

    #use CPU/gpu 
    if gpu_cpu == 'gpu':
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    

    #building the classifier
    
    layers = []
    layer_sizes = [input_size] + hidden_layers + [output_size]
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < len(hidden_layers):  #activation and dropout for hidden layers
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.5))
    layers.append(nn.LogSoftmax(dim=1))
    classifier = nn.Sequential(*layers)
    # Build model
        
    model = models.vgg16(pretrained=True) if use_model == 'vgg16' else models.resnet50(pretrained=True)

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model.to(device)
    return model, criterion, optimizer,device,input_size,output_size,hidden_layers
    
    
    

def train_model(model, criterion, optimizer,device, epochs,trainloader,validloader,patience):
    #training the classifier


    best_val_loss = float('inf')  
    early_stop_counter = 0  

    
    for e in range(epochs):
        
        print(f"Starting Epoch {e+1}/{epochs}")
        #training loop
        model.train()
        running_loss = 0
        for images, labels in trainloader:
            
            optimizer.zero_grad()
            
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            output = model(images)
            loss = criterion(output, labels)

            # Backward pass and optimization
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation 
        model.eval()
        accuracy = 0
        val_loss = 0
        
        with torch.no_grad():
            for images, labels in validloader:
                images, labels = images.to(device), labels.to(device)

                out_val = model(images)
                loss_val = criterion(out_val, labels)
                val_loss += loss_val.item()
                
                # accuracy
                ps = torch.exp(out_val)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        
        print(f"Epoch {e+1}/{epochs}.. "
            f"Train loss: {running_loss/len(trainloader):.3f}.. "
            f"Validation loss: {val_loss/len(validloader):.3f}.. "
            f"Validation accuracy: {accuracy/len(validloader):.3f}")
        
        # Early stopping 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0 
            torch.save(model.state_dict(), 'best_model.pth')  
            print("Validation loss improved. Saving model...")
        else:
            early_stop_counter += 1
            print(f"No improvement in validation loss for {early_stop_counter} epochs.")
        
        # Stop training if patience is exceeded
        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            break
        
        # Reset the training mode
        running_loss = 0
        model.train()
    return model, optimizer, e

def save_model(model, optimizer, image_datasets, e,input_size,output_size,hidden_layers):
    state_dict = torch.load('best_model.pth')
    model.load_state_dict(state_dict)
    model.class_to_idx = image_datasets['train'].class_to_idx

    checkpoint = {
        'model': model,
        'input_size': input_size,  
        'output_size': output_size,  
        'hidden_layers': hidden_layers,
        'state_dict': model.state_dict(),  
        'class_to_idx': model.class_to_idx,  
        'classifier': model.classifier, 
        'optimizer_state': optimizer.state_dict(), 
        'epochs': e +1  
    }


    torch.save(checkpoint, 'checkpoint.pth')
    return f"Model saved at checkpoint.pth"

def test_model(model, criterion, device, testloader):
    with torch.no_grad():
        test_loss = 0
        total_samples = 0
        for images,labels in testloader:
            images,labels = images.to(device),labels.to(device)
            
            test_out = model(images)
            loss_test = criterion(test_out, labels)
            test_loss += loss_test.item()
            #accuracy
            
            ps = torch.exp(test_out)
            top_p,top_class = ps.topk(1,dim=1)
            equals = top_class == labels.view(*top_class.shape)
            #accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            # Batch accuracy and sample count
            batch_size = labels.size(0)  
            accuracy += torch.sum(equals).item()  
            total_samples += batch_size

    final_accuracy = (accuracy / total_samples) * 100
    return f"Test Accuracy: {final_accuracy:.4f}%"

def load_checkpoint(model_filepath,device):
    checkpoint = torch.load(model_filepath)
    model = models.vgg16(pretrained=True) 
    model.classifier = checkpoint['classifier']  
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx'] 
    model.to(device)

    return model

def process_image(image): 
  
    with Image.open(image) as image:

        aspect_ratio = image.width / image.height
        if aspect_ratio > 1:
            image = image.resize((int(256 * aspect_ratio), 256))
        else:
            image = image.resize((256, int(256 / aspect_ratio)))

        left = (image.width - 224) / 2
        upper = (image.height - 224) / 2
        right = left + 224
        lower = upper + 224
        image = image.crop((left, upper, right, lower))

        np_image = np.array(image) / 255.0
        means = np.array([0.485, 0.456, 0.406])
        stds = np.array([0.229, 0.224, 0.225])
        np_image = (np_image - means) / stds
        np_image = np_image.transpose((2, 0, 1))
    return np_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict_image(np_image, model, topk):
    #processed_image = process_image(image)
    processed_image = torch.tensor(np_image, dtype=torch.float32).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    processed_image = processed_image.to(device)

    model.eval()
    with torch.no_grad():
        output = model(processed_image)

    ps = torch.exp(output)
    top_ps, top_class = ps.topk(topk, dim=1)
    return top_ps[0].cpu().numpy(), top_class[0].cpu().numpy()


