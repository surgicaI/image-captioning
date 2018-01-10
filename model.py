import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init_weights()
        
    def init_weights(self):
        """Initialize the weights."""
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
        
    def forward(self, images):
        """Extract the image feature vectors."""
        features = self.resnet(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        cnn_features = features
        features = self.bn(self.linear(features))
        return features, cnn_features
    
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.attention = nn.Linear(hidden_size+embed_size, 2048)
        self.attended = nn.Linear(2048+embed_size, embed_size)
        self.softmax = nn.Softmax()
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        self.attention.weight.data.uniform_(-0.1, 0.1)
        self.attention.bias.data.fill_(0)
        self.attended.weight.data.uniform_(-0.1, 0.1)
        self.attended.bias.data.fill_(0)

    def forward(self, features, cnn_features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed, batch_sizes = pack_padded_sequence(embeddings, lengths, batch_first=True)
        
        hiddenStates = None
        start = 0
        for batch_size in batch_sizes:
            in_vector = packed[start:start+batch_size].view(batch_size, 1, -1)
            start += batch_size
            if hiddenStates is None:
                hiddenStates, (h_n , c_n) = self.lstm(in_vector)
                hiddenStates = torch.squeeze(hiddenStates)
            else:
                h_n, c_n = h_n[:,0:batch_size,:], c_n[:,0:batch_size,:]
                info_vector = torch.cat((in_vector, h_n.view(batch_size, 1, -1)), dim=2)
                attention_weights = self.attention(info_vector.view(batch_size, -1))
                attention_weights = self.softmax(attention_weights)
                attended_weights = cnn_features[0:batch_size] * attention_weights
                attended_info_vector = torch.cat((in_vector.view(batch_size, -1), attended_weights), dim=1)
                attended_in_vector = self.attended(attended_info_vector)
                attended_in_vector = attended_in_vector.view(batch_size, 1, -1)
                out, (h_n , c_n) = self.lstm(attended_in_vector, (h_n, c_n))
                hiddenStates = torch.cat((hiddenStates, out.view(batch_size, -1)))
        hiddenStates = self.linear(hiddenStates)
        return hiddenStates

    
    def sample(self, features, cnn_features, states=None):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        batch_size = features.size(0)
        inputs = features.unsqueeze(1)
        for i in range(20):                                      # maximum sampling length
            if states is None:
                hiddens, states = self.lstm(inputs, states)          # (batch_size, 1, hidden_size), 
            else:
                h_n, c_n = states
                info_vector = torch.cat((inputs, h_n.view(batch_size, 1, -1)), dim=2)
                attention_weights = self.attention(info_vector.view(batch_size, -1))
                attention_weights = self.softmax(attention_weights)
                attended_weights = cnn_features[0:batch_size] * attention_weights
                attended_info_vector = torch.cat((inputs.view(batch_size, -1), attended_weights), dim=1)
                attended_in_vector = self.attended(attended_info_vector)
                attended_in_vector = attended_in_vector.view(batch_size, 1, -1)
                hiddens, states = self.lstm(attended_in_vector, states)
            outputs = self.linear(hiddens.squeeze(1))            # (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)                         # (batch_size, 1, embed_size)
        sampled_ids = torch.cat(sampled_ids, 0)                  # (batch_size, 20)
        return sampled_ids.squeeze()
