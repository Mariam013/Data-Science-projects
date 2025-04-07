from utils import load_transforms, image_datasets, label_data, build_classifier, train_model,test_model,save_model
import argparse

# argument parser
parser = argparse.ArgumentParser(description="Train a neural network model.")

# Add arguments
parser.add_argument("--data_dir", type=str, help="Path to the data folder directory.")
parser.add_argument("--labels_file", type=str, help="Path to the labels file.")
parser.add_argument("--device", choices=["gpu", "cpu"], default="cpu", help="Choose device: 'gpu' or 'cpu'. Default is 'cpu'.")
parser.add_argument("--model", choices=["vgg16", "resnet50"], required=True, help="Choose the model to use ('vgg16' or 'resnet50').")

# Hidden layers
parser.add_argument("--hidden_layers", type=int, nargs="+", required=True, help="Sizes of the hidden layers, e.g., '--hidden_layers 128 64 32'.")
parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate for the optimizer.")
parser.add_argument("--epochs", type=int, required=True, help="Number of epochs for training.")
parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping.")

# Parse arguments
args = parser.parse_args()

#arguments
data_dir = args.data_dir
labels_file = args.labels_file
device = args.device
use_model = args.model
hidden_layers = args.hidden_layers
learning_rate = args.learning_rate
epochs = args.epochs
patience = args.patience


train_dir = f"{data_dir}/train"
valid_dir = f"{data_dir}/valid"
test_dir = f"{data_dir}/test"

def train(data_dir, labels_file, gpu_cpu, use_model, hidden_layers, learning_rate, epochs, patience):

    train_transform, test_transform = load_transforms()
    trainloader,validloader,testloader = image_datasets(train_dir, valid_dir, test_dir,train_transform, test_transform)
    cat_to_name,output_size = label_data(labels_file)
    model, criterion, optimizer,device,input_size,output_size,hidden_layers = \
        build_classifier(gpu_cpu,use_model,hidden_layers,output_size,learning_rate)
    model, optimizer, e = train_model(model, criterion, optimizer,device, epochs,trainloader,validloader,patience)

    

    test_model(model, criterion, device, testloader)
    save_model(model, optimizer, data_dir, e,input_size,output_size,hidden_layers)

    model,criterion, optimizer, e, testloader,input_size, output_size,device = \
        train_model(model, criterion, optimizer,device, epochs,patience)
    
    
    return f"Model training complete"



# Call the train function
print(train(data_dir, labels_file, device, use_model, hidden_layers, learning_rate, epochs, patience))
