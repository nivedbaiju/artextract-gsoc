import torch
import timm

class CNN_BiLSTM(torch.nn.Module):
    def __init__(self,num_style,num_genre,num_artists):
        super().__init__()

        #first the cnn backbone is defined. I have picked the efficientnet b3 noisy student variant here.
        self.cnn_backbone=timm.create_model('tf_efficientnet_b3_ns', pretrained=True, num_classes=0, global_pool="")

        self.feature_dim=self.cnn_backbone.num_features

        #now we define the biLSTM
        self.bilstm=torch.nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        #since the bilstm is bidirectional,o/p dimension will be 512*2=1024
        bilstm_out_dim=1024

        #inorder to get three labels as outputs,we design three fc layers for final label generations.
        self.style_head=torch.nn.Linear(bilstm_out_dim,num_style)
        self.genre_head=torch.nn.Linear(bilstm_out_dim,num_genre)  
        self.artist_head=torch.nn.Linear(bilstm_out_dim,num_artists) 

    def forward(self,x):
        cnn_features=self.cnn_backbone.forward_features(x)
        B,C,H,W=cnn_features.shape 

        #cnn_features has shape (batch_size,channels,feature_map_height,feature_map_width)
        #but lstm expects input of shape (batch_size,sequence_len,feature_dim)
        #so we reshape to (batch_size,height*width,channels)
        cnn_features=cnn_features.reshape(B,C,H*W).permute(0,2,1)

        #ignore hidden states.
        lstm_out,_=self.bilstm(cnn_features)
        
        #use mean pooling to get summary of all patches.
        fc_in = lstm_out.mean(dim=1)
        
        style_out=self.style_head(fc_in)
        genre_out=self.genre_head(fc_in)    
        artist_out=self.artist_head(fc_in)

        return style_out, genre_out, artist_out
