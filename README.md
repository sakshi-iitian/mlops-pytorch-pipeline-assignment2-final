# MLOps PyTorch Pipeline

A production-style MLOps pipeline for training and serving a PyTorch CIFAR-10 image classification model using Docker and Kubernetes.

## Project Overview

This project demonstrates the complete ML deployment lifecycle:

**PyTorch Model → Docker → Kubernetes Training Job → Persistent Storage → Model Serving → Health Checks → Service → HPA → Prediction**

The pipeline includes:

* PyTorch ResNet-18 image classifier
* CIFAR-10 dataset
* Config-driven training
* JSON-line training metrics
* Model checkpointing
* Early stopping support
* Docker training image
* Docker serving image
* Kubernetes training Job
* Kubernetes PersistentVolumeClaims
* Kubernetes ConfigMap
* Kubernetes Deployment
* Kubernetes Service
* Liveness and readiness probes
* Horizontal Pod Autoscaler
* FastAPI prediction API

## Architecture

```text
                         ┌─────────────────────┐
                         │       Client        │
                         │   curl / API call   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Kubernetes Service  │
                         │    ClusterIP :80    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     Model Serving Deployment  │
                    │                               │
                    │   ┌─────────┐   ┌─────────┐  │
                    │   │ Pod 1   │   │ Pod 2   │  │
                    │   │ FastAPI │   │ FastAPI │  │
                    │   └────┬────┘   └────┬────┘  │
                    └────────┼──────────────┼───────┘
                             │              │
                             └──────┬───────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Model Checkpoint    │
                         │ classifier_v1.pt    │
                         │ Persistent Volume   │
                         └──────────▲──────────┘
                                    │
                                    │ saves checkpoint
                                    │
                         ┌──────────┴──────────┐
                         │ Kubernetes Training │
                         │        Job          │
                         └──────────▲──────────┘
                                    │
                                    │ configuration
                                    │
                         ┌──────────┴──────────┐
                         │      ConfigMap      │
                         │ training_config.yaml│
                         └─────────────────────┘

                         ┌─────────────────────┐
                         │        HPA          │
                         │   2 → 5 replicas    │
                         │    CPU target 70%   │
                         └─────────────────────┘
```

## Repository Structure

```text
mlops-pytorch-pipeline-assignment2-final/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── configs/
│   └── training_config.yaml
│
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.serve
│
├── k8s/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── pvc.yaml
│   ├── training-job.yaml
│   ├── serving-deployment.yaml
│   ├── serving-service.yaml
│   └── hpa.yaml
│
├── requirements/
│   ├── train.txt
│   └── serve.txt
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── serve.py
│
├── tests/
│   └── test_model.py
│
├── .gitignore
└── README.md
```

## Local Setup

Clone the repository:

```bash
git clone https://github.com/sakshi-iitian/mlops-pytorch-pipeline-assignment2-final.git
cd mlops-pytorch-pipeline-assignment2-final
```

Create a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install training dependencies:

```bash
pip install -r requirements/train.txt
```

Run training locally:

```bash
python3 src/train.py
```

The training script reads the configuration from:

```text
configs/training_config.yaml
```

The trained checkpoint is saved to:

```text
checkpoints/classifier_v1.pt
```

## Model

The project uses a ResNet-18 based image classifier trained on the CIFAR-10 dataset.

The ten CIFAR-10 classes are:

```text
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
```

## Training Configuration

Training parameters are stored in:

```text
configs/training_config.yaml
```

Example configuration:

```yaml
model:
  architecture: resnet18
  num_classes: 10

training:
  epochs: 10
  batch_size: 64
  learning_rate: 0.001
  early_stopping_patience: 3

data:
  dataset: cifar10
  data_dir: /app/data

output:
  checkpoint_dir: /app/checkpoints
  model_name: classifier_v1.pt
```

The Kubernetes training Job mounts the configuration using a ConfigMap.

## Docker

### Build Training Image

```bash
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
```

### Run Training Container

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-train:v1
```

### Build Serving Image

```bash
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
```

### Run Serving Container

```bash
docker run --rm \
  -p 8080:8080 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-serve:v1
```

## FastAPI Serving API

The model serving application is implemented using FastAPI.

### Health Endpoint

```text
GET /health
```

Test:

```bash
curl http://localhost:8080/health
```

Expected response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Prediction Endpoint

```text
POST /predict
```

Send an image:

```bash
curl -X POST http://localhost:8080/predict \
  -F "image=@test_image.png"
```

Example response:

```json
{
  "predicted_class": "frog",
  "class_index": 6
}
```

The API also returns probabilities for all ten CIFAR-10 classes.

## Kubernetes Deployment

The Kubernetes deployment uses the `ml-training` namespace.

### 1. Create Namespace

```bash
kubectl apply -f k8s/namespace.yaml
```

### 2. Apply Configuration

```bash
kubectl apply -f k8s/configmap.yaml
```

### 3. Create Persistent Storage

```bash
kubectl apply -f k8s/pvc.yaml
```

### 4. Run Training Job

```bash
kubectl apply -f k8s/training-job.yaml
```

Check the Job:

```bash
kubectl get jobs -n ml-training
```

Check training logs:

```bash
kubectl logs job/model-training -n ml-training
```

### 5. Deploy Model Serving

After training completes:

```bash
kubectl apply -f k8s/serving-deployment.yaml
```

Create the Service:

```bash
kubectl apply -f k8s/serving-service.yaml
```

### 6. Enable Autoscaling

```bash
kubectl apply -f k8s/hpa.yaml
```

## Kubernetes Verification

Check Pods:

```bash
kubectl get pods -n ml-training
```

Check PersistentVolumeClaims:

```bash
kubectl get pvc -n ml-training
```

Check Deployment:

```bash
kubectl get deployment model-serving -n ml-training
```

Check Service:

```bash
kubectl get svc model-serving -n ml-training
```

Check HPA:

```bash
kubectl get hpa model-serving-hpa -n ml-training
```

## Port Forwarding

The Kubernetes Service is a ClusterIP service.

Forward the service to localhost:

```bash
kubectl port-forward svc/model-serving 8080:80 -n ml-training
```

Then test the health endpoint:

```bash
curl http://localhost:8080/health
```

Test prediction:

```bash
curl -X POST http://localhost:8080/predict \
  -F "image=@test_image.png"
```

## Kubernetes Resources

### Training Job

```text
Job: model-training
Status: Complete
Completions: 1/1
```

### Model Serving

```text
Deployment: model-serving
Replicas: 2/2
```

Both serving Pods were successfully running.

### Persistent Storage

```text
data-storage-pvc   Bound   10Gi
model-storage-pvc  Bound   10Gi
```

### Service

```text
Service: model-serving
Type: ClusterIP
Port: 80
Target Port: 8080
```

### Horizontal Pod Autoscaler

```text
HPA: model-serving-hpa
Minimum replicas: 2
Maximum replicas: 5
CPU target: 70%
```

## Training Results

The Kubernetes training Job completed successfully after 10 epochs.

Final training metrics:

```text
Epoch: 10
Train loss: 0.6362
Train accuracy: 0.7814
Validation loss: 0.6554
Validation accuracy: 0.7738
```

Best validation accuracy:

```text
77.38%
```

Best validation loss:

```text
0.6554
```

The trained checkpoint was saved as:

```text
classifier_v1.pt
```

## End-to-End Validation

The complete workflow was successfully tested:

```text
PyTorch Training
       ↓
Docker Training Image
       ↓
Kubernetes Training Job
       ↓
PersistentVolumeClaim
       ↓
Model Checkpoint
       ↓
Kubernetes Serving Deployment
       ↓
FastAPI
       ↓
ClusterIP Service
       ↓
Health Check
       ↓
Prediction
       ↓
HPA Autoscaling
```

Example successful health response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

Example prediction result:

```text
Predicted class: frog
Class index: 6
Probability: 93.01%
```

## Git Workflow

The project follows a feature-branch and Pull Request workflow.

All feature work was developed on feature branches and merged through Pull Requests.

The project contains Pull Requests covering:

1. PyTorch CIFAR-10 training pipeline
2. Docker training and serving containers
3. Kubernetes training Job
4. Kubernetes model serving and autoscaling
5. End-to-end deployment validation
6. Final project integration

All required feature work was merged into the `develop` branch before the final deployment was merged into `main`.

## CI

GitHub Actions is configured through:

```text
.github/workflows/ci.yml
```

The workflow runs automated project checks when changes are pushed or Pull Requests are created.

## Technologies Used

* Python
* PyTorch
* Torchvision
* FastAPI
* Docker
* Kubernetes
* kubectl
* PersistentVolumeClaims
* ConfigMaps
* Horizontal Pod Autoscaler
* Git
* GitHub Actions
* CIFAR-10

## Assignment Requirements Covered

| Requirement              | Status    |
| ------------------------ | --------- |
| PyTorch image classifier | Completed |
| CIFAR-10 dataset         | Completed |
| Config-driven training   | Completed |
| JSON training metrics    | Completed |
| Model checkpointing      | Completed |
| Early stopping support   | Completed |
| Docker training image    | Completed |
| Docker serving image     | Completed |
| Kubernetes Training Job  | Completed |
| PersistentVolumeClaims   | Completed |
| ConfigMap                | Completed |
| Kubernetes Deployment    | Completed |
| Two serving replicas     | Completed |
| Health probes            | Completed |
| ClusterIP Service        | Completed |
| HPA                      | Completed |
| End-to-end prediction    | Completed |
| Git feature branches     | Completed |
| Pull Request workflow    | Completed |
| CI workflow              | Completed |
| README documentation     | Completed |
