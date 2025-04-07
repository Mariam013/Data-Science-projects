
#imports
import torch
from PIL import Image
import matplotlib.pyplot as plt
import json
#from utils import load_checkpoint, process_image, predict_image,imshow
import numpy as np
from torchvision import models
import argparse




#argument parser
parser = argparse.ArgumentParser(description="Run inference on an image using a pre-trained model.")

#arguments
parser.add_argument("--image_path", type=str, help="Path to the image file.")
parser.add_argument("--model_filepath", type=str, help="Path to the model checkpoint.")
parser.add_argument("--labels_file", type=str, help="Path to the directory for the labels file.")
parser.add_argument("--device", choices=["gpu", "cpu"], default="cpu", help="Choose device: 'gpu' or 'cpu'. Default is 'cpu'")
parser.add_argument("--model", choices=["vgg16", "resnet50"], required=True, help="Choose the model to use ('vgg16' or 'resnet50').")
parser.add_argument("--topk", type=int, default=5, help="Return top K most likely classes. Default is 5")

# Parse arguments
args = parser.parse_args()

# Process arguments
image_path = args.image_path
model_filepath = args.model_filepath
labels_file = args.labels_file
device = torch.device("cuda" if args.device == "gpu" else "cpu")
use_model = args.model
topk = args.topk

# Load the image
image = Image.open(image_path)

# Load the labels
with open(labels_file, 'r') as f:
    cat_to_name = json.load(f)
    


def load_checkpoint(model_filepath,device):
    checkpoint = torch.load(model_filepath)
    model = models.vgg16(pretrained=True) if use_model == 'vgg16' else models.resnet50(pretrained=True)
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
    return top_ps.cpu().numpy(), top_class.cpu().numpy()



model=load_checkpoint(model_filepath,device)
print(f"Model leaded successfully from {model_filepath}")
np_image=process_image(image_path)
print(f"Processed image shape: {np_image.shape}")
processed_image = torch.from_numpy(np_image)


top_ps,top_class=predict_image(np_image, model,topk)
top_ps = top_ps.flatten()
top_class = top_class.flatten()
top_class_names = [cat_to_name.get(str(class_idx), "Unknown") for class_idx in top_class]
#most likely class and probability
most_likely_class = top_class_names[0]
most_likely_prob = top_ps[0]
print(f"Most likely class: {most_likely_class} with probability: {most_likely_prob:.4f}")

#Top probabilities and classes
print(f"Top {topk} probabilities and classes:")
for prob, class_name in zip(top_ps, top_class_names):
    print(f"{class_name}: {prob:.4f}")












#print(f"Top probabilities: {top_ps}")
#print(f"Top classes: {top_class_names}")
#fig, ax = plt.subplots(figsize=(5, 3))
#imshow(processed_image)
#ax.barh(top_class_names, top_ps)
#ax.set_xlabel("Probability")
#ax.set_title("Predicted Classes")
#plt.show()






def predict_image_classes(image_path, model_filepath, cat_to_name, device,topk=5):
    model=load_checkpoint(model_filepath,device)

    np_image=process_image(image_path)
    processed_image = torch.from_numpy(np_image)
   
    

    top_ps,top_class=predict_image(np_image, model, topk)

    top_class_names = [cat_to_name.get(str(class_idx), "Unknown") for class_idx in top_class]

    fig, ax = plt.subplots(figsize=(5, 3))
    imshow(processed_image.squeeze().cpu().numpy())
    ax.barh(top_class_names, top_ps)
    ax.set_xlabel("Probability")
    ax.set_title("Predicted Classes")
    plt.show()

    return top_class_names, top_ps



# Call the function
#top_class_names, top_ps = predict_image_classes(image, model_filepath, cat_to_name, device, topk=5)
