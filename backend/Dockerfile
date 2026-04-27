FROM public.ecr.aws/lambda/python:3.10

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch CPU first (separate step for caching)
RUN pip install torch==2.1.0+cpu torchvision==0.16.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --no-cache-dir

# Install remaining dependencies
COPY requirements.txt .
RUN pip install fastapi==0.109.0 mangum==0.17.0 \
    Pillow==10.2.0 python-multipart==0.0.9 \
    --no-cache-dir

# Copy all app files
COPY main.py    ${LAMBDA_TASK_ROOT}/
COPY model.py   ${LAMBDA_TASK_ROOT}/
COPY utils.py   ${LAMBDA_TASK_ROOT}/
COPY best_resnet_deepfake.pt ${LAMBDA_TASK_ROOT}/best_resnet_deepfake.pt

# Lambda handler
CMD ["main.handler"]