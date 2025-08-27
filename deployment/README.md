# Deployment: ChemBERTaDNN Production System


## **System Architecture**

┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐
│   User      │───▶│  API Gateway│───▶│ Model Inference     │
└─────────────┘    └─────────────┘    │   Service           │
                                      └─────────────────────┘
                                          ▲
                                          │
                                   ┌─────────────┐
                                   │   Data      │
                                   │  Storage    │
                                   │  (S3/RDS)   │
                                   └─────────────┘

```mermaid
flowchart TD
    A[User CSV Input] --> B[API Gateway]
    B --> C[ECS Model Inference Service]
    C --> D[CSV Output to User]
    C --> E[Data Storage (S3)]
    C --> F[Monitoring & Logging (CloudWatch)]
```



### **Components**
1. **User**: Sends a request with a SMILES string to the API Gateway.
2. **API Gateway**: Receives the request and forwards it to the Model Inference Service.
3. **Model Inference Service**: Loads the pre-trained ChemBERTaDNN model and processes the SMILES string to generate a prediction.
4. **Data Storage**: Stores the model and any necessary data for inference, eg AWS S3. 
5. **Monitoring**: Tracks the performance and usage of the service for maintenance and improvement (e.g AWS CloudWatch). 

# # Deployment Overview
### **Prerequisites**
- AWS Account
- Docker
- AWS CLI

### **Steps**
1. **Containerize the Application**:
   - Package the Model Inference Service into a Docker container for easy deployment and scalability.
   - Build the Docker image: `docker build -t chemberta-api -f poc/Dockerfile .`
   - Push the image to Amazon ECR.

2. **Deploy the Model Inference Service**:  
   - Deploy the containerized service on a cloud platform (e.g. AWS).
   - Use AWS ECS (Elastic Container Service) to deploy the container. 
   - ECS is a simple and scalable way to run Docker containers in AWS.

3. **Set Up API Gateway**:
   - Configure an API Gateway to route CSV requests to the Model Inference Service.
   - Create a REST API in API Gateway.
   - Connect the API to the ECS service.

4. **Set Up Data Storage**:
   - Use Amazon S3 to store raw and processed data.
   
5. **Testing**: 
   - Thoroughly test the deployed service to ensure it correctly processes requests and returns accurate predictions.

6**Set Up Monitoring and Logging**:
   - Configure CloudWatch to monitor the API and inference service.


Simplicity: Uses managed AWS services to minimize operational overhead.
Scalability: ECS can automatically scale the number of containers based on demand.
Reliability: AWS services are highly available and fault-tolerant.
Cost-Effectiveness: Pay only for the resources you use.

## Example Request
The service receives a CSV file with a column smiles:
```` csv
smiles
CCO
CCN
CCC
````

The service responds with predictions in a CSV format:
```csv
prediction
0.85
0.73
0.91
```

