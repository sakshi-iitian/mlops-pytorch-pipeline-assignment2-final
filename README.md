# MLOps PyTorch Pipeline

PyTorch CIFAR-10 image classifier with Docker training/serving and Kubernetes deployment.

## Architecture

Client -> Kubernetes Service -> FastAPI Serving Pods -> Saved PyTorch Checkpoint
                                      ^
                                      |
Persistent Volume <- Kubernetes Training Job <- Training ConfigMap

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/train.txt
python src/train.py --config configs/training_config.yaml
```

## Docker

```bash
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/checkpoints:/app/checkpoints mlops-train:v1

docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
docker run --rm -p 8080:8080 -v $(pwd)/checkpoints:/app/checkpoints mlops-serve:v1
```

## Kubernetes

Build/load images into your cluster, then:

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/training-job.yaml
kubectl get pods -n ml-training -w
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl port-forward svc/model-serving 8080:80 -n ml-training
```

Test:

```bash
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```
