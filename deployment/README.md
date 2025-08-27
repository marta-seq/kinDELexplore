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

### **Components**
1. **User**: Sends a request with a SMILES string to the API Gateway.
2. **API Gateway**: Receives the request and forwards it to the Model Inference Service.
3. **Model Inference Service**: Loads the pre-trained ChemBERTaDNN model and processes the SMILES string to generate a prediction.
4. **Data Storage**: Stores the model and any necessary data for inference, eg AWS S3. 
5. **Monitoring**: Tracks the performance and usage of the service for maintenance and improvement.

# # Deployment Overview
### **Prerequisites**
- AWS Account
- Docker
- AWS CLI
- Terraform (optional) ???? 

### **Steps**
1. **Containerize the Application**:
   - Build the Docker image: `docker build -t chemberta-api -f poc/Dockerfile .`
   - Push the image to Amazon ECR.

2. **Deploy the Model Inference Service**:
   - Use AWS ECS or Lambda to deploy the container.

3. **Set Up API Gateway**:
   - Create a REST API in API Gateway.
   - Connect the API to the ECS service or Lambda function.

4. **Set Up Data Storage**:
   - Use Amazon S3 to store raw and processed data.

5. **Set Up Monitoring and Logging**:
   - Configure CloudWatch to monitor the API and inference service.

---

## **Minimal Requirements**
   Component                | AWS Service               |
 |--------------------------|----------------------------|
 | Docker Image             | ECR                        |
 | Compute                  | ECS/Fargate or Lambda      |
 | API Gateway              | API Gateway               |
 | Data Storage             | S3, RDS, or DynamoDB       |
 | Monitoring               | CloudWatch                 |
 | Infrastructure as Code   | Terraform or AWS CDK       |

---

## **Notes**
- Use Terraform or AWS CDK for Infrastructure as Code (IaC).
- Ensure security best practices (e.g., IAM roles, encryption).
- 
[]: # The deployment architecture consists of the following components:
[]: # 
[]: # 1. **API Gateway**: Serves as the entry point for user requests. It handles incoming requests containing SMILES strings and forwards them to the Model Inference Service.
[]: # 
[]: # 2. **Model Inference Service**: This service is responsible for loading the pre-trained ChemBERTaDNN model and processing the SMILES strings to generate predictions. It can be implemented using a web framework such as Flask or FastAPI.
[]: # 
[]: # 3. **Data Storage**: A storage solution (e.g., AWS S3, RDS) to store the trained model and any necessary data for inference.
[]: # 
[]: # 4. **Monitoring & Logging**: Implement monitoring and logging to track the performance and usage of the service for maintenance and improvement.
[]: # 
[]: # ## Deployment Steps
[]: # 
[]: # 1. **Containerization**: Package the Model Inference Service into a Docker container for easy deployment and scalability.
[]: # 
[]: # 2. **Cloud Deployment**: Deploy the containerized service on a cloud platform (e.g., AWS, GCP, Azure) using services like AWS ECS, EKS, or Lambda.
[]: # 
[]: # 3. **API Gateway Setup**: Configure an API Gateway to route requests to the Model Inference Service.
[]: # 
[]: # 4. **Testing**: Thoroughly test the deployed service to ensure it correctly processes requests and returns accurate predictions.
[]: # 
[]: # 5. **Scaling**: Set up auto-scaling policies based on demand to ensure the service remains responsive under varying loads.
[]: # 
[]: # ## Example Request
[]: # A sample request to the API Gateway might look like this:
[]: # 
[]: # ```json
[]: # {
[]: #   "smiles": "CCO"
[]: # }
[]: # ```
[]: # The response would contain the predicted property value:
[]: # 
[]: # ```json
[]: # {
[]: #   "prediction": 0.85
[]: # }
[]: # ```