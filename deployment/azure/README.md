# Azure Deployment Guide

This guide outlines how to deploy the Malaria Risk Predictor Streamlit app to Azure using a container image.

## Prerequisites

- Azure subscription with permission to create resource groups and Web Apps.
- Azure CLI installed and authenticated (`az login`).
- Docker Engine or compatible container runtime.
- The project repository cloned locally.

## 1. Build the Container Image

```bash
docker build -t malaria-risk-predictor:latest .
```

## 2. Test the Container Locally

```bash
docker run --rm -p 8501:8501 malaria-risk-predictor:latest
```

Visit `http://localhost:8501` to confirm the app is running.

## 3. Push the Image to Azure Container Registry (ACR)

1. Create a resource group (skip if it already exists):

   ```bash
   az group create --name sdg-ai-rg --location eastus
   ```

2. Create an Azure Container Registry:

   ```bash
   az acr create --resource-group sdg-ai-rg --name sdgaicr --sku Basic --admin-enabled true
   ```

3. Log in to ACR and tag the image:

   ```bash
   az acr login --name sdgaicr
   docker tag malaria-risk-predictor:latest sdgaicr.azurecr.io/malaria-risk-predictor:latest
   docker push sdgaicr.azurecr.io/malaria-risk-predictor:latest
   ```

## 4. Deploy to Azure App Service

1. Create an App Service plan (Linux):

   ```bash
   az appservice plan create \
     --name sdg-ai-plan \
     --resource-group sdg-ai-rg \
     --sku B1 \
     --is-linux
   ```

2. Create the Web App pointing to the container image:

   ```bash
   az webapp create \
     --resource-group sdg-ai-rg \
     --plan sdg-ai-plan \
     --name malaria-risk-predictor-app \
     --deployment-container-image-name sdgaicr.azurecr.io/malaria-risk-predictor:latest
   ```

3. Configure container credentials:

   ```bash
   az webapp config container set \
     --name malaria-risk-predictor-app \
     --resource-group sdg-ai-rg \
     --docker-custom-image-name sdgaicr.azurecr.io/malaria-risk-predictor:latest \
     --docker-registry-server-url https://sdgaicr.azurecr.io \
     --docker-registry-server-user $(az acr credential show --name sdgaicr --query username -o tsv) \
     --docker-registry-server-password $(az acr credential show --name sdgaicr --query passwords[0].value -o tsv)
   ```

4. Configure Streamlit specific settings:

   ```bash
   az webapp config appsettings set \
     --name malaria-risk-predictor-app \
     --resource-group sdg-ai-rg \
     --settings WEBSITES_PORT=8501
   ```

5. Restart the app to apply settings:

   ```bash
   az webapp restart --name malaria-risk-predictor-app --resource-group sdg-ai-rg
   ```

## 5. Monitor and Scale

- Use Azure Monitor for logs: `az webapp log tail`.
- Scale with `az appservice plan update --name sdg-ai-plan --resource-group sdg-ai-rg --number-of-workers 2`.
- Review costs regularly; stop or scale down when idle.

## 6. CI/CD (Optional)

Consider enabling GitHub Actions or Azure DevOps pipelines to automatically build and push new images on every commit.

Refer to Azure documentation for container-based Web App deployments for deeper customization.

