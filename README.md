# MLOps PyTorch Pipeline

A complete MLOps pipeline for training and serving a PyTorch CIFAR-10 image classification model using Docker and Kubernetes.

## Architecture

```text
                         ┌──────────────────────┐
                         │     Client/User      │
                         └──────────┬───────────┘
                                    │
                              HTTP / Predict
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Kubernetes Service   │
                         │    ClusterIP :80     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                     ┌────────────────────────────┐
                     │  Model Serving Deployment  │
                     │        2 Replicas           │
                     │        FastAPI :8080        │
                     └─────────────┬──────────────┘
                                   │
                                   ▼
                         ┌──────────────────────┐
                         │ PyTorch Checkpoint   │
                         │   classifier_v1.pt   │
                         └──────────────────────┘
                                   ▲
                                   │
                         Model Storage PVC
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    │ Kubernetes Training Job     │
                    │                             │
                    │ PyTorch + CIFAR-10          │
                    └──────────────┬──────────────┘
                                   │
                         Training ConfigMap
                                   │
                                   ▼
                         Training Configuration
```

## Project Structure

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml
├── configs/
│   └── training_config.yaml
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.serve
├── k8s/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── pvc.yaml
│   ├── training-job.yaml
│   ├── serving-deployment.yaml
│   ├── serving-service.yaml
│   └── hpa.yaml
├── requirements/
│   ├── train.txt
│   └── serve.txt
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── serve.py
└── tests/
    └── test_model.py
```

## Model

The project uses a ResNet-18 based PyTorch image classifier trained on the CIFAR-10 dataset.

The training pipeline:

* Downloads and loads the CIFAR-10 dataset.
* Applies training and validation transforms.
* Reads hyperparameters from `configs/training_config.yaml`.
* Trains the model using PyTorch.
* Logs training and validation metrics as JSON lines.
* Saves the best model checkpoint.
* Supports early stopping.
* Stores the trained model in the configured checkpoint directory.

## Local Setup

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the training dependencies:

```bash
pip install -r requirements/train.txt
```

Run training:

```bash
python3 src/train.py
```

The training configuration is read from:

```text
configs/training_config.yaml
```

The trained checkpoint is saved according to the output configuration.

## Docker

### Training Image

Build the training image:

```bash
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
```

Run training with mounted data and checkpoint directories:

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-train:v1
```

### Serving Image

Build the serving image:

```bash
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
```

Run the serving container:

```bash
docker run --rm \
  -p 8080:8080 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-serve:v1
```

## Serving API

The serving application is implemented using FastAPI.

### Health Check

Check whether the model is loaded:

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

### Prediction

Send an image to the prediction endpoint:

```bash
curl -X POST http://localhost:8080/predict \
  -F "image=@test_image.png"
```

The endpoint returns:

* Predicted CIFAR-10 class
* Class index
* Probability for each CIFAR-10 class

## Kubernetes Deployment

Create the namespace:

```bash
kubectl apply -f k8s/namespace.yaml
```

Apply the training configuration:

```bash
kubectl apply -f k8s/configmap.yaml
```

Create the persistent storage:

```bash
kubectl apply -f k8s/pvc.yaml
```

Start the Kubernetes training Job:

```bash
kubectl apply -f k8s/training-job.yaml
```

Check the training Job:

```bash
kubectl get jobs -n ml-training
```

Check the pods:

```bash
kubectl get pods -n ml-training
```

View training logs:

```bash
kubectl logs job/model-training -n ml-training
```

After the training Job completes, deploy the model-serving layer:

```bash
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml
```

## Kubernetes Validation

### Check Training Job

```bash
kubectl get jobs -n ml-training
```

Expected result:

```text
NAME             STATUS     COMPLETIONS
model-training   Complete   1/1
```

### Check Pods

```bash
kubectl get pods -n ml-training
```

The serving pods should be running and ready.

### Check Persistent Storage

```bash
kubectl get pvc -n ml-training
```

The following PVCs should be `Bound`:

```text
data-storage-pvc
model-storage-pvc
```

### Check Deployment

```bash
kubectl get deployment model-serving -n ml-training
```

The deployment is configured to run two replicas.

### Check Service

```bash
kubectl get svc model-serving -n ml-training
```

The model-serving Service uses `ClusterIP` and exposes port `80`.

### Check HPA

```bash
kubectl get hpa model-serving-hpa -n ml-training
```

The HPA is configured with:

* Minimum replicas: 2
* Maximum replicas: 5
* CPU target: 70%

## Port Forwarding

For local testing of the Kubernetes Service:

```bash
kubectl port-forward svc/model-serving 8080:80 -n ml-training
```

Then test the health endpoint:

```bash
curl http://localhost:8080/health
```

Test the prediction endpoint:

```bash
curl -X POST http://localhost:8080/predict \
  -F "image=@test_image.png"
```

## End-to-End Validation

The complete Kubernetes workflow was successfully validated.

### Training

* Kubernetes Job: `model-training`
* Status: `Complete`
* Completion: `1/1`
* Training epochs: `10`
* Best validation accuracy: `77.38%`
* Best validation loss: `0.6554`

### Model Serving

* Deployment: `model-serving`
* Replicas: `2/2`
* Serving pods: `Running`
* Service type: `ClusterIP`
* Service port: `80`

### Persistent Storage

* `data-storage-pvc`: `Bound`, 10Gi
* `model-storage-pvc`: `Bound`, 10Gi

### Autoscaling

* HPA: `model-serving-hpa`
* Minimum replicas: `2`
* Maximum replicas: `5`
* CPU target: `70%`

### Health Check

The serving API returned:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Prediction

The `/predict` endpoint was successfully tested using `test_image.png`.

Example result:

```text
predicted_class: frog
class_index: 6
probability: 93.01%
```

The API also returned probabilities for all 10 CIFAR-10 classes.

## Testing

Run the model tests with:

```bash
pytest tests/
```

## Git Workflow

The project follows a feature-branch and Pull Request workflow.

Development work was performed on feature branches and merged into the `develop` branch through Pull Requests.

The final `develop` branch was then merged into `main` through the final Pull Request.

Merged Pull Requests cover:

1. PyTorch CIFAR-10 training pipeline
2. Docker training and serving images
3. Kubernetes training Job configuration
4. Kubernetes model serving and autoscaling
5. Final end-to-end PyTorch Kubernetes deployment and validation

## Technologies

* Python
* PyTorch
* Torchvision
* FastAPI
* Docker
* Kubernetes
* PersistentVolumeClaims
* ConfigMaps
* Horizontal Pod Autoscaler
* GitHub Actions
* Git/GitHub

## End-to-End Workflow

```text
PyTorch Model
      ↓
CIFAR-10 Training
      ↓
Docker Training Image
      ↓
Kubernetes Training Job
      ↓
PersistentVolumeClaim
      ↓
Saved PyTorch Checkpoint
      ↓
Kubernetes Model Serving
      ↓
FastAPI
      ↓
ClusterIP Service
      ↓
Health Check / Prediction
      ↓
Horizontal Pod Autoscaler
```
