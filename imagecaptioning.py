import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch import nn
from PIL import Image

# Image feature extraction
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, 256)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features

# Caption generation
class CaptionDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(CaptionDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

# Combined model
class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = ImageEncoder()
        self.decoder = CaptionDecoder(embed_size, hidden_size, vocab_size)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def generate_caption(self, image, vocab, max_length=20):
        result_caption = []
        with torch.no_grad():
            x = self.encoder(image).unsqueeze(0)
            states = None
            
            for _ in range(max_length):
                hiddens, states = self.decoder.lstm(x, states)
                output = self.decoder.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                
                result_caption.append(predicted.item())
                x = self.decoder.embed(predicted).unsqueeze(0)
                
                if vocab.itos[predicted.item()] == '<eos>':
                    break
        
        return ' '.join([vocab.itos[idx] for idx in result_caption])

# Image preprocessing
def preprocess_image(image_path, transform):
    image = image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

# Main function to demonstrate usage
def main():
    # Define hyperparameters
    embed_size = 256
    hidden_size = 512
    vocab_size = 10000  # This should be the actual size of your vocabulary

    # Initialize the model
    model = ImageCaptioningModel(embed_size, hidden_size, vocab_size)

    # Load pre-trained weights (you would need to train the model first)
    # model.load_state_dict(torch.load('path_to_your_trained_model.pth'))

    # Set up image transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Load and preprocess an image
    image_path = 'path_to_your_image.jpg'
    image = preprocess_image(image_path, transform)

    # Generate a caption
    # Note: You would need to implement or load a vocabulary for this to work
    # vocab = YourVocabularyClass()
    # caption = model.generate_caption(image, vocab)
    # print(f"Generated caption: {caption}")

if __name__ == '__main__':
    main()