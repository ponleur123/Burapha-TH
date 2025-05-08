import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.hub
import torchvision.transforms as transforms
from PIL import Image
import os

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# List of available model files
MODEL_FILES = {
    "lenet": "grayscale_lenet_state_dict.pt",
    "cnn": "grayscale_custom_CNN_state_dict.pt",
    "resnet": "grayscale_resnet_state_dict.pt",
    "vgg": "grayscale_vgg_state_dict.pt"
}

# Replace with your actual class names
class_names = {'49': 'SARA AA',
 '34': 'RO RUA',
 '18': 'NO NEN',
 '64': 'MAI THO',
 '37': 'LU',
 '05': 'KHO RAKHANG',
 '46': 'PAIYANNOI',
 '35': 'RU',
 '17': 'THO PHUTHAO',
 '06': 'NGO NGU',
 '09': 'CHO CHANG',
 '19': 'DO DEK',
 '28': 'FO FA',
 '24': 'NO NU',
 '57': 'SARA E',
 '23': 'THO THONG',
 '42': 'HO HIP',
 '08': 'CHO CHING',
 '20': 'TO TAO',
 '16': 'THO NANGMONTHO',
 '44': 'O ANG',
 '31': 'PHO SAMPHAO',
 '02': 'KHO KHUAT',
 '07': 'CHO CHAN',
 '29': 'PHO PHAN',
 '39': 'SO SALA',
 '60': 'SARA AI MAIMUAN',
 '11': 'CHO CHOE',
 '55': 'SARA U',
 '50': 'SARA AM',
 '53': 'SARA UE',
 '40': 'SO RUSI',
 '59': 'SARA O',
 '22': 'THO THAHAN',
 '30': 'FO FAN',
 '27': 'PHO PHUNG',
 '13': 'DO CHADA',
 '67': 'THANTHAKHAT',
 '10': 'SO SO',
 '61': 'SARA AI MAIMALAI',
 '33': 'YO YAK',
 '32': 'MO MA',
 '54': 'SARA UEE',
 '41': 'SO SUA',
 '03': 'KHO KHWAI',
 '65': 'MAI TRI',
 '00': 'KO KAI',
 '25': 'BO BAIMAI',
 '52': 'SARA II',
 '66': 'MAI CHATTAWA',
 '45': 'HO NOKHUK',
 '47': 'SARA A',
 '38': 'WO WAEN',
 '56': 'SARA UU',
 '14': 'TO PATAK',
 '58': 'SARA AE',
 '26': 'PO PLA',
 '63': 'MAI EK',
 '15': 'THO THAN',
 '12': 'YO YING',
 '21': 'THO THUNG',
 '01': 'KHO KHAI',
 '36': 'LO LING',
 '43': 'LO CHULA',
 '48': 'MAI HAN',
 '62': 'MAITAIKHU',
 '04': 'KHO KHON',
 '51': 'SARA I'}  # Update with actual class names

# Image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # if your images are grayscale
    transforms.Resize((64, 64)),                # ResNet expects 224x224
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Normalize(mean=[0.485,], std=[0.229,])
])
class LeNet5(nn.Module):
    def __init__(self, num_classes=68):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        
        self.fc1 = nn.Linear(16*13*13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, 2)
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class HandwrittenTextCNN(nn.Module):
    def __init__(self):
        super(HandwrittenTextCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        self.fc1 = nn.Linear(8192,4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048,1024)
        self.fc4 = nn.Linear(1024,68)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        x = torch.flatten(x,1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x  # Shape: [batch_size, 128, 8, 8]

def load_model(model_choice):
    model_path = MODEL_FILES[model_choice]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found.")
    

    if "cnn" in model_choice:
        # Load custom model
        model = HandwrittenTextCNN()
        
    elif "lenet" in model_choice:
        model = LeNet5()
    
    elif "vgg" in model_choice:
        model = torch.hub.load('pytorch/vision:v0.10.0','vgg11', pretrained=False)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        model.classifier[-1] = nn.Linear(in_features=4096, out_features=68, bias=True)    
    
    else:
        # Load pre-trained ResNet18 from torch.hub
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(model.fc.in_features, out_features=68, bias=True)

    # Load state dictionary
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def predict(model_choice, image):
    if image is None:
        return "Please upload an image."

    try:
        # Load the selected model
        model = load_model(model_choice)
        
        # Process the image
        image = Image.fromarray(image).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_class = class_names[predicted.item()]
        
        return f"Predicted class: {class_names[predicted_class]}"
    
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(choices=list(MODEL_FILES.keys()), label="Select Model"),
        gr.Image(type="numpy", label="Upload Image")
    ],
    outputs="text",
    title="Image Classification with PyTorch Models",
    description="Select a custom or pre-trained model and upload an image to get a classification prediction."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
    
    

