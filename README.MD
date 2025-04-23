# Python - Dash App
> This is more than a simple Dashboard app.

This project is based on DASH framework.

So far in the project, we have
1. a [dashboard](pages\dashboard.py) page with interactive charts and maps with multiple filters.
2. a [Image Classification](pages\ImageClassification.py) web interface that uses MobileNetV2 model for ImageNet classification.

To run locally:
```bash
# Install Dependencies
pip install -r requirements.txt
# Run App
python main.py
```

To run in production:
```bash
#!/bin/bash

# Install Dependencies
pip install -r requirements.txt

# Run App
gunicorn --workers 1 -b 0.0.0.0:8080 main:server
```
