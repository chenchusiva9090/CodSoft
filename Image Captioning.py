# Import necessary libraries
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from torch.autograd import Variable

# Load pre-trained ResNet model
resnet = models.resnet50(pretrained=True)
modules = list(resnet.children())[:-1]
resnet = nn.Sequential(*modules)
for param in resnet.parameters():
    param.requires_grad = False
resnet.eval()

# Define LSTM-based captioning model
class CaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(CaptioningModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        embeddings = self.embedding(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        lstm_out, _ = self.lstm(embeddings)
        out = self.fc(lstm_out)
        return out

# Define a function to preprocess images
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image

# Define a function to generate captions
def generate_caption(image_path, model, vocab, max_len=20):
    image = preprocess_image(image_path)
    image_features = resnet(image).squeeze().unsqueeze(0)
    states = (Variable(torch.zeros(1, 1, hidden_size)).to(device),
              Variable(torch.zeros(1, 1, hidden_size)).to(device))
    
    result_caption = []
    for _ in range(max_len):
        inputs = embed_captions(result_caption).unsqueeze(0)
        inputs = torch.cat((image_features, inputs), 1)
        lstm_out, states = model.lstm(inputs, states)
        out = model.fc(lstm_out)
        _, predicted = out.max(2)
        result_caption.append(predicted.item())
        if vocab.itos[predicted.item()] == '<eos>':
            break
    
    caption = [vocab.itos[i] for i in result_caption]
    caption = ' '.join(caption[1:-1])  # Remove <start> and <eos>
    return caption

# Load pre-trained captioning model and vocabulary
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab = torch.load('vocab.pkl')
model = CaptioningModel(len(vocab), embed_size, hidden_size, num_layers).to(device)
model.load_state_dict(torch.load('captioning_model.pth'))
model.eval()

# Example usage
caption = generate_caption('example.jpg', model, vocab)
print(caption)
