# Python - Dash App
> This is more than a simple Dashboard app.

This project is based on DASH framework.

So far in the project, we have
1. a [dashboard](pages\dashboard.py) page with interactive charts and maps with multiple filters.
2. a [Image Classification](pages\ImageClassification.py) web interface that uses MobileNetV2 model for ImageNet classification.

Install Dependencies:
```bash
# Install Dependencies
pip install -r requirements.txt
## For Running Image Classification using pytorch
pip install -r torch_requirements.txt -f https://download.pytorch.org/whl/cpu/torch_stable.html
```
To run locally:
```bash
# Run App
python main.py
```

To run in production:
```bash
# Run Gunicorn Server
gunicorn --workers 1 -b 0.0.0.0:8080 main:server
```

Checkout the app on: [https://dashboard-wrl5.onrender.com/](https://dashboard-wrl5.onrender.com/)