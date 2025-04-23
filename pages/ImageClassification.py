import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
from plotly import express as px
import pandas as pd
import numpy as np

import base64
from io import BytesIO
from PIL import Image

import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.transforms import transforms

dash.register_page(__name__, path='/cv', name='Image Classification', title="Image Classification", description="Image Classification using MobileNetV3 Large")

model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

layout = dbc.Container([
        dbc.Row([            
            dbc.Col(width=12, children=[
                html.H4("Upload an image to classify it using a pre-trained model.", className="text-center"), html.Br(),
                dbc.Label("The model used is MobileNet-V3-Small with IMAGENET1K_V1 weights from PyTorch.", className='text-start mb-2'),
            ]),
        ], class_name='mb-3'),
        dcc.Upload(id='upload-data',
            max_size=1024*1024*5,  # 10MB
            accept='image/*',
            children=html.Div([
                dbc.Label('Drag and Drop or Select Files', style={"cursor":"pointer"}),
            ]), 
            className='w-100 text-center p-4 rounded-1 border-1',
            style={
                'lineHeight': '30px',
                'borderStyle': 'dashed',
            },
            className_active='upload-dragdrop-active-bg'),
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    # dbc.CardHeader(id='image-upload-name', className='text-start'),
                    dbc.CardBody(dcc.Loading(html.Div(id='output-image-upload', children=[dbc.Label("No Image Uploaded!!")], className='w-100 text-center', style={"minHeight":"200px"}), type="circle")),
                ], class_name="w-100"),
                sm=12, md=8, className='text-center mt-3'),
            dbc.Col(dbc.Card(
                dbc.CardBody(id='model-output-container', children=dbc.Label("Model Output:"), class_name="text-start")),sm=12, md=4, className='text-center mt-3')
        ]),
       
    ], fluid=True)

def read_upload_image(contents):
    # Decode the base64 string and convert to an image
    header, encoded = contents.split(",", 1)  # Splits into "data:image/png;base64" and the base64 string
    image_data = base64.b64decode(encoded)
    
    # Convert to an Image object
    image = Image.open(BytesIO(image_data))
    image = image.convert('RGB')  # Ensure image is in RGB format
    return image

def eval_model(image: Image.Image, topK =5):    
    # Step 3: Resize and normalize the image
    image = image.resize((224, 224))  # Resize to match model input size
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    image = np.transpose(image, (2, 0, 1))  # Change to (C, H, W) format
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = torch.tensor(image, dtype=torch.float32)  # Convert to tensor
    
    # Step 4: Pass the image through the model
    with torch.no_grad():
        output = model(image)
    
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    topk_prob, topk_catid = torch.topk(probabilities, topK)
    
    try:
        classes = []
        imgnet_class = pd.read_csv('imagenet_class_list.csv')
        classes = imgnet_class[imgnet_class['id'].isin(topk_catid.tolist())]['label'].tolist()
        return topk_prob.tolist(), classes
    except FileNotFoundError:
        print("imagenet_class_index.csv file not found.")
        pass
    
    return topk_catid.tolist() # Return the predicted class index
    

@dash.callback(
    Output('output-image-upload', 'children'),
    Output('model-output-container', 'children'),
    # Input('upload-data-store', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output( contents, filename):
    if contents is not None:  
        image = read_upload_image(contents)
        topK_prob, topK_class = eval_model(image)
        
        output = [html.H4("Top 5 - Model Output: ", className='card-title font-weight-bold'), html.Hr(),
                *[
                    html.P([
                        dbc.Label([html.B(f"Label: "), html.Span(f" {topK_class[i]}")], class_name="mb-1 w-100"),
                        dbc.Label([html.B("Probability: "), html.Span(f" {topK_prob[i]:.4f}")], class_name="mb-1 w-100"),
                        html.Br()
                    ], className="mb-2 border w-100 rounded-1 p-1")
                    for i in range(len(topK_prob))
                ]]
        return [html.Div(dbc.Label("File Name: "+filename, class_name="card-title text-start"), className='text-start'), dbc.CardImg(src=contents, class_name='w-100', bottom=True)], output
        
    return dash.no_update