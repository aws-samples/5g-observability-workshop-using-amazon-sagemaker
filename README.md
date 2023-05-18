# 5g-observability-using-amazon-sagemaker

Welcome to our workshop! This is a comprehensive guide to help you understand the Machine Learning (ML) use case and get started with the workshop modules.

For step-by-step instructions, please visit AWS Workshop Studio: https://catalog.us-east-1.prod.workshops.aws/workshops/5ea0503b-e142-420d-b7ed-3b114bf77cdf/en-US

---

## Introduction to the Use Case

5G is transforming the way services are delivered to the end user. This calls for radically different approaches to network performance assurance. To support high speed connectivity to 5G-enabled devices available in the market today, NSA mode makes the most sense. 5G non-standalone (NSA) is a solution for 5G networks where the network is supported by the existing 4G infrastructure. This allows operators to leverage their existing network assets rather than deploy a completely new end-to-end 5G network.

5G non-standalone uses a 4G LTE core with a 5G RAN (radio access network). User equipment's (UEs) have dual connectivity between LTE and 5G. The inability to connect to 5G is represented by the abnormal release of the connection with 5G sgNB. This can lead to poor user experience. In this workshop, we use abnormal release rate from sample NSA 5G RAN dataset to determine such drops as anomalies and train a classification model to detect connectivity drops under known network utilization, contention rates, accessibility, health index and throughput parameters. This usecase is part of 5G performance assurance to detect possible loss in connectivity to 5G radio based on SLAs.

In this workshop, we use a 5G NSA RAN dataset monitored at different cell towers to build a machine learning model for detect anomalies resulting from connectivity loss. The dataset includes a wide set of features. Here, we explain the associated terminology in the feature set:

|Abbreviation|Stands for|Description|
|:----|:----|:----|
|ca|Carrier Aggregaion|a technology that allows simultaneous use of multiple frequency blocks (carriers) to increase the data rate per user.|
|*rrc*|Radio Resource Control|a protocol to control the allocation and management of radio resources between a UE (User Equipment) and the 5G network. The protocol consists of three main states: idle, connected, and inactive.|
|*rach*|Random Access Channel|a wireless channel used by a UE to initiate a connection with the 5G network. The UE can use contention based RACH procedure, where the UE randomly generates and sends the RACH preamble. Contention may occur when multiple UEs generates the same RACH preamble.|
|*sgnb*|Secondary gNB|another gNB base station that provides additional coverage for the UE, in addtional to the UE’s primary gNB.|
|*rssi*|Received Signal Strength Indicator|a measurement of the signal strength of a wireless signal.|
|*cce*|Control Channel Elements|a group of resources that carry control channel information between a UE and the base station.|
|*bler*|Block Error Rate|the ratio of the number of blocks (data units) received in error to the total number of blocks transmitted.|
|*drb*|Data Radio Bearer|a connection between the UE and the 5G network for user data transmission. Each DRB is associated with a QoS class of data traffic.|
|*pdu session*|Protocol Data Unit session|an end-to-end connection between the UE and one or multiple data networks for transferring user data. A PDU session can include one or multiple DRBs.|

---

## Prerequisites

Before you begin, please make sure you read the following prerequisites:

1. Familiarity with telecom 5G network performance and service assurance 
2. Basic knowledge of machine learning concepts and techniques.
3. Understanding of Python programming language and its libraries, such as NumPy, Pandas, and Scikit-learn.
4. Familiarity with cloud computing platforms and services, especially Amazon Web Services (AWS).
5. An AWS account with the necessary permissions to create and configure Amazon SageMaker instances, roles, and resources.

If you do not have experience with these prerequisites, we recommend that you complete some basic online courses or tutorials before attending the workshop to get the most out of the training. Additionally, we recommend reviewing the Amazon SageMaker documentation and examples beforehand to become familiar with the service and its features.

---
## Get Started
To get started, please clone this workshop repo:

```

git clone https://github.com/aws-samples/5g-observability-workshop-using-amazon-sagemaker.git

```
---
## Workshop Modules

Each module builds on top of the previous module and needs to be executed in sequence to get the most out of the workshop. However, if you are only interested in specific modules, you can execute the module 0: **[0_setup.ipynb](0_setup.ipynb)** to get the resources and parameters set up properly.

- [Module 1: Prepare your data with SageMaker Data Wrangler](1_dataprep.ipynb)

- [Module 2: Build Your Model with No-Code/Low-Code (NCLC): Canvas](2_canvas.ipynb)

- [Module 2: Build Your Model with No-Code/Low-Code (NCLC): Autopilot](2_autopilot.ipynb)
   
- [Module 3: Build custom models using Studio Notebook](3_studio_notebook.ipynb)

- [Module 4: Productionalize your model using SageMaker Pipelines](4_e2e_pipeline.ipynb)