import datetime

import streamlit as st
from PIL import Image
import os
import time

import matplotlib.pyplot as plt
import os
from PIL import Image
from PIL import UnidentifiedImageError
import torch
from torchvision import datasets, transforms
import re
import pandas as pd
import random
import shutil
import datetime as dt
import torch.nn as nn


#LOADING TORCH MODELS
import torchvision.models as models
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import torchvision.transforms as transforms
import torchmetrics
import numpy as np
import matplotlib.pyplot as plt
import os
import dill as pickle
import torchvision.transforms.functional as TF
import os
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as Func

class Block(pl.LightningModule):
    
    def init(self, in_channels, out_channels, kernel_size=3):
        super().init()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.conv(x)

class Encoder(pl.LightningModule):

    def init(self, in_channels, out_channels):
        super().init()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Decoder(pl.LightningModule):

    def init(self, in_channels, out_channels):
        super().init()
        self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
        self.conv = Block(in_channels, out_channels)

    def forward(self, expand_input, contract_input):
        expand_output = self.up(expand_input)
        if expand_output.shape != contract_input.shape:
                expand_output = TF.resize(expand_output, size=contract_input.shape[2:])
        concat_output = torch.cat((expand_output, contract_input), dim=1)
        return self.conv(concat_output)

class UNet(pl.LightningModule):

    def init(self, chn=(3,64,128,256,512, 1024)):
        super().init()
        self.learning_rate = 1e-3 #1e-3

        self.train_pixel_acc = torchmetrics.classification.BinaryAccuracy(num_classes=2, ignore= 255, average='weighted')
        self.train_J = torchmetrics.classification.BinaryJaccardIndex(num_classes=2, ignore= 255, average='weighted')

        self.val_pixel_acc = torchmetrics.classification.BinaryAccuracy(num_classes=2, ignore= 255 , average='weighted')
        self.val_J = torchmetrics.classification.BinaryJaccardIndex(num_classes=2, ignore= 255 , average='weighted')

        self.test_pixel_acc = torchmetrics.classification.BinaryAccuracy(num_classes=2, ignore= 255, average='weighted')
        self.test_J = torchmetrics.classification.BinaryJaccardIndex(num_classes=2, ignore= 255, average='weighted')      
        # encoder
        self.first_layer = Block(chn[0], chn[1])
        self.encode_layers = nn.ModuleList()
        self.decode_layers = nn.ModuleList()
        
        for i in range(1, len(chn)-2):
            self.encode_layers.append(Encoder(chn[i],chn[i+1]))
            
        # bottleneck
        self.bottleneck = Block(chn[-2], chn[-1])
        # decoder
                    
        for i in range(len(chn)-1, 1, -1):
            self.decode_layers.append(Decoder(chn[i], chn[i-1]))
            
        self.final_layer = nn.Conv2d(chn[1], 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        
        
        #############################

    def forward(self, x):
        # TODO: Define U-Net layers #
        x = self.first_layer(x)
        connections = []
        connections.append(x)
        for encode_layer in self.encode_layers:
            x = encode_layer(x)
            connections.append(x)
            
        x = self.bottleneck(x)
        connections = connections[::-1]

        for i in range(len(self.decode_layers)):
            x = self.decode_layers[i](x, connections[i])
                           
        return self.sigmoid(self.final_layer(x))
        #############################

    def decode_segmap(self, prediction):
        label_colors = torch.tensor([(0, 64, 128)])
        r = torch.zeros_like(prediction, dtype=torch.uint8)
        g = torch.zeros_like(prediction, dtype=torch.uint8)
        b = torch.zeros_like(prediction, dtype=torch.uint8)
        for l in range(0, self.num_class):
            idx = prediction == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]
        rgb = torch.stack([r, g, b], axis=1)
        return rgb

    def training_step(self, train_batch, batch_idx):
        images, seg_mask = train_batch

        # convert the non-binary seg_mask to binary
        seg_mask_binary = torch.where(seg_mask > 0, torch.tensor(1), torch.tensor(0))
        seg_mask_binary = seg_mask_binary.squeeze(1).to(device)
        outputs = self(images)
        pred_seg_mask = torch.argmax(outputs, 1).to(device)
        outputs = outputs.squeeze(1).to(device)
        pred_seg_mask = pred_seg_mask.squeeze(1).to(device)
        # pixel-wise accuracy
        self.train_pixel_acc(pred_seg_mask, seg_mask_binary)

        # the Jaccard index (mean IoU)
        self.train_J(pred_seg_mask, seg_mask_binary)
        
        self.log('train_acc', self.train_pixel_acc, on_step=False, on_epoch=True)
        self.log('train_mIoU', self.train_J, on_step=False, on_epoch=True)

        # convert seg_mask_binary to float
        seg_mask_binary = seg_mask_binary.float()

        # loss
        loss = F.binary_cross_entropy(outputs.squeeze(1).to(device), seg_mask_binary.float())
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        print("Train :", loss, self.train_pixel_acc)
        return loss



    def validation_step(self, val_batch, batch_idx):
      images, seg_mask = val_batch

      seg_mask_binary = torch.where(seg_mask > 0, torch.tensor([1]).to(device), torch.tensor([0]).to(device))
      seg_mask_binary = seg_mask_binary.squeeze(1).to(device)
      outputs = self(images)
      pred_seg_mask = torch.argmax(outputs, 1)
      outputs = outputs.squeeze(1).to(device)
      pred_seg_mask = pred_seg_mask.squeeze(1).to(device)
      print(outputs.shape, pred_seg_mask.shape, seg_mask_binary.shape)
      # pixel-wise accuracy
      self.val_pixel_acc(pred_seg_mask, seg_mask_binary)

      # the Jaccard index (mean IoU)
      self.val_J(pred_seg_mask, seg_mask_binary)

      self.log('val_pixel_acc', self.val_pixel_acc, on_step=False, on_epoch=True)
      # self.log('val_mIoU', self.val_J, on_step=False, on_epoch=True)
      print("Valid:",  self.val_pixel_acc, self.val_J)
      loss = F.binary_cross_entropy(outputs.squeeze(1).to(device), seg_mask_binary.float())
      self.log('valid_loss', loss, on_step=False, on_epoch=True)
      return loss



    def test_step(self, batch, batch_idx):
      images, seg_mask = batch

      seg_mask_binary = torch.where(seg_mask > 0, torch.tensor([1]).to(device), torch.tensor([0]).to(device))
      seg_mask_binary = seg_mask_binary.squeeze(1).to(device)

      outputs = self(images)
      pred_seg_mask = torch.argmax(outputs, 1)
      outputs = outputs.squeeze(1).to(device)
      pred_seg_mask = pred_seg_mask.squeeze(1).to(device)
      print(outputs.shape, pred_seg_mask.shape, seg_mask_binary.shape)
      self.test_pixel_acc(pred_seg_mask, seg_mask_binary)
      self.test_J(pred_seg_mask, seg_mask_binary)
      print("Test:",  self.test_pixel_acc, self.val_J)
      self.log("test_acc", self.test_pixel_acc)
      # self.log("test_mIoU", self.test_J)

      loss = F.binary_cross_entropy(outputs.squeeze(1), seg_mask_binary.float())
      self.log('test_loss', loss, on_step=False, on_epoch=True)
      return loss
        

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr = self.learning_rate)
        return opt



class CNN_Pretrained(nn.Module):
    def __init__(self):
        super(CNN_Pretrained, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )

        self.backbone = models.resnet50(pretrained=True)
        
        #freeze pre-trained layer
        for param in self.backbone.parameters():
            param.requires_grad = False
                            
        self.linear1 = nn.Linear(1000, 249)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(249+7, 128)
        self.linear3 = nn.Linear(128,3)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, features):
        out = None
        out = self.sequential(x)
        out = self.backbone(out)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = torch.cat((out,features),dim = 1) #check dimensions
        out = self.linear2(out)
        out = self.linear3(out)
        out = self.softmax(out)
        
        return out
    


# def load(model, filename):
#     modeltorch.load(filename))
#     return model



def formTensorData(image, reshape_size, material, floor, date, duration):
    

    convert_tensor = transforms.ToTensor()

    #resize every image to input shape
    input_shape = (reshape_size, reshape_size)
    image = image.resize(input_shape[:2])
    image= convert_tensor(image)

    Material_Beton= [1 if material=="Beton" else 0] 	
    Material_Stein=	 [1 if material=="Stein" else 0]
    Material_Holz=[1 if material=="Holz" else 0]
    Material_Styropor=[1 if material=="Styropor" else 0]

    # datetime_object = datetime.datetime.strptime(year, '%Y-%M-%d')

    age=[int(dt.datetime.today().strftime("%Y"))-date.year]
    duration=[int(duration)]
    floor=[floor]

    features=floor+duration+Material_Beton+Material_Holz+Material_Stein+Material_Styropor+age

    features=torch.Tensor(features).to(DEVICE) 

    image=image.to(DEVICE) 
                
    return image, features


def reset_session():
    st._is_running = False
# Add custom CSS to change the background color

st.markdown("""
    <div style='text-align: center;'>
        <h1>Water Damage Assessment</h1>
    </div>
    <style>
    body {
        background-color: #F0F8FF;
    }
    </style>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

feature1_value = st.selectbox('Material', ['Beton', 'Holz', 'Styropor','Stein'])
feature2_value = st.slider('Floor Number',0, 20)
feature3_value = st.date_input("Construction year",datetime.datetime.now(),min_value=datetime.date(1700, 1, 1))
feature4_value = st.text_input('Damage Duration')


classes={0:"0-10000",1:"10000-20000",2:"20000-30000",3:"30000-40000",4:"40000-50000",5:"50000-60000",6:">60000"}


style = "<style>.row-widget.stButton {text-align: center;}</style>"
st.markdown(style, unsafe_allow_html=True)


if st.button('Submit'):
    if uploaded_file is not None and feature1_value is not None:
        # Load the image and display it
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Output in the python console
        print(image.size)
        print('Material:', feature1_value)
        print('Floor Number:', feature2_value)
        print('Construction year:', feature3_value)
        print('Damage Duration:', feature4_value)

        dirname=os.path.dirname(__file__)

        model = torch.load(dirname+"\cnn_pretrained_sgd.pt").to(DEVICE)
        model.eval()

        unet_model = torch.load(dirname+r"\unet_water_damage.pt").to(DEVICE)
        unet_model.eval()

        transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
        ])

        img_tensor = unet_model(transform(image).to(DEVICE).unsqueeze_(0))

        img_tensor=img_tensor.cpu()
        print(type(img_tensor))

        #img_array = img_tensor.numpy().squeeze()

        # st.image(img_array, caption='Segmented Image', use_column_width=True)


        st.write('Material:', feature1_value)
        st.write('Floor Number:', feature2_value)
        st.write('Construction year:', feature3_value)
        st.write('Damage Duration:', feature4_value)

        progress_text = "Assessing the Damage..."
        st.write(progress_text)
        my_bar = st.progress(0)

        image1,image_features=formTensorData(image,224, feature1_value,feature2_value,feature3_value,feature4_value)
        print(image1.shape)
        image1 = image1.unsqueeze_(0)
        print(image1.shape)
        image_features=image_features.unsqueeze_(0)

        out = model(image1,image_features)

        _, predictions = torch.max(out, dim = 1)
        print(predictions)

        for percent_complete in range(100):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1)

        st.write('Assessed Damage Category: ',predictions.tolist()[0])
        st.write('Assessed Damage Range: ',classes[predictions.tolist()[0]])



    else:
        st.write('Please select an image and provide values for all inputs.')

# Apply CSS styles to buttons
button_styles = """
<div style='text-align: center;'>
    <style>
        #submit_button {
            background-color: #32CD32;
            color: white;
            font-size: 20px;
            padding: 10px 20px;
            border-radius: 5px;
        }
        #clear_button {
            background-color: #FF0000;
            color: white;
            font-size: 20px;
            padding: 10px 20px;
            border-radius: 5px;
        }
        .button-container {
                display: flex;
                justify-content: center;
                margin-top: 20px;
            }
    </style>
</div>
"""
st.markdown(button_styles, unsafe_allow_html=True)

if st.button('Clear'):
    # os._exit(0)
    reset_session()