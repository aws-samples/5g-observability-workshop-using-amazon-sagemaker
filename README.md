# 5g-observability-using-amazon-sagemaker

Welcome to our workshop! This is a comprehensive guide to help you understand the Machine Learning (ML) use case and get started with the workshop modules.

---

## Introduction to the Use Case

5G network slicing enables operators to create multiple independent logical networks for supporting different services over a common infrastructure. Each slice consists of a collection of virtual functions for a core network (CN) and a radio access network (RAN), and is customized to address the need of specific applications and services. 5G observability refers to the ability to monitor and analyze the performance of 5G networks. Due to the virtualization nature of network slicing, 5G observability is essential to assure and automate thousands of network slices in real-time.

5G accessibility is a crucial KPI of 5G networks that has a significant impact on both user experience and customer loyalty. 5G accessibility depends on systems related to 5G radio parameters, random access control, paging control, and admission control. By monitoring the parameters of the above systems, we can predict anomalies in 5G accessibility. This will allow the operators to proactively perform maintenance and prevent potential issues that may lead to customer dissatisfaction and service churn.

In this workshop, we use a dataset monitored at different cell towers to build a machine learning model for predicting anomalies in 5G accessibility. The dataset includes a wide set of features. Here, we explain the associated terminology in the feature set:

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

1. Basic knowledge of machine learning concepts and techniques.
2. Understanding of Python programming language and its libraries, such as NumPy, Pandas, and Scikit-learn.
3. Familiarity with cloud computing platforms and services, especially Amazon Web Services (AWS).
4. An AWS account with the necessary permissions to create and configure Amazon SageMaker instances, roles, and resources.

If you do not have experience with these prerequisites, we recommend that you complete some basic online courses or tutorials before attending the workshop to get the most out of the training. Additionally, we recommend reviewing the Amazon SageMaker documentation and examples beforehand to become familiar with the service and its features.

---
## Get Started
To get started, please clone this workshop repo:

```

git clone 

```
---
## Workshop Modules

Each module builds on top of the previous module and needs to be executed in sequence to get the most out of the workshop. However, if you are only interested in specific modules, you can execute the module 0: **[0_setup.ipynb](0_setup.ipynb)** to get the resources and parameters set up properly.

- [Module 1: Prepare your data with SageMaker Data Wrangler](1_dataprep.ipynb)

- [Module 2: Build Your Model with No-Code/Low-Code (NCLC): Canvas](2_canvas.ipynb)

- [Module 2: Build Your Model with No-Code/Low-Code (NCLC): Autopilot](2_autopilot.ipynb)
   
- [Module 3: Build custom models using Studio Notebook](3_studio_notebook.ipynb)

- [Module 4: Productionalize your model using SageMaker Pipelines](4_e2e_pipeline.ipynb)