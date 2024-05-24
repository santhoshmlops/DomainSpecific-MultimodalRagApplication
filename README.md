# Domain-Specific Multimodal Rag Application

The project aims to improve the performance of the RAG (Retrieval-Augmented Generation) model, a state-of-the-art natural language processing architecture, by fine-tuning it with domain-specific data. RAG combines the power of retrieval-based and generative models to provide contextually relevant responses to queries. However, to optimize its performance in specific domains, such as healthcare, finance, or legal, fine-tuning with domain-specific question-and-answer pairs is essential. This project involves collecting relevant data in the form of Q&A pairs, preprocessing it, and incorporating it into the training process of the RAG model. By doing so, the model can better understand and generate accurate responses within the chosen domain, thus enhancing its utility and effectiveness in real-world applications.

This Application is a versatile tool designed to provide users with interactive assistance across various modes of content, including text-based chat, PDF documents, websites, and YouTube videos. Leveraging Streamlit for its user interface, multimodal analysis techniques to offer tailored responses and recommendations to user queries. This application supports both CPU and GPU processing for model inference and offers model selection, allowing users to choose between different pre-trained models.

## Application User Interfaces:

https://github.com/santhoshmlops/DomainSpecific-MultimodalRagApplication/assets/133121635/17283b13-8ec9-489d-88ec-b2127d0ae24a

## Application Inferencing:
https://github.com/santhoshmlops/DomainSpecific-MultimodalRagApplication/assets/133121635/2b9abd30-2342-4bfd-b8ea-d57d70d8bea6

# Hugging Face Models:

### Fine-tune Gemma 2B-it for QLoRa Quantization via Supervised Fine Tuning
![sft](https://github.com/santhoshmlops/MultimodalRagApplication/assets/133121635/6228fd31-6ed1-4f84-8e33-9709ec2ad6f1)

### For CPU inference, convert the Hugging Face Model to GGUF format using the LLAMA.CPP
![GGUF](https://github.com/santhoshmlops/MultimodalRagApplication/assets/133121635/a7ccb613-15ae-40ec-88f2-df424c662c81)



# Key Features

### Conversational Chatbot
This chatbot is powered by a fine-tuned Gemma 2B-it model for QLoRa Quantization, providing a user-friendly interface for interacting with the knowledge base. The fine-tuning process leverages a dataset of question-answer pairs tailored to the specific context of the application.

### PDF Interaction

The application enables seamless interaction with PDF documents. Upon uploading a PDF file, users can navigate, extract information, and interact with the content effectively.

### Website Interaction

The application extends its capabilities to interact with websites. Users can provide website URLs, and the application cross-references extracted website content with user queries, facilitating accurate information retrieval and contextual understanding.

### YouTube Integration

Users can input YouTube video links, allowing the application to retrieve relevant information and interact with the content. This feature enhances the application's versatility, catering to diverse multimedia needs.

### Optimized Inference

For efficient CPU inference, the application converts the fine-tuned Gemma 2B-it model to the GGUF format using LLAMA.CPP. This optimization ensures smooth performance on various hardware configurations.
### Optional Ollama Integration

The application offers the flexibility to incorporate Ollama for additional inference options, catering to a wider range of user needs.


# System Requirements
The recommended system requirements for this project are:

| Device | Requirements | 
|----------|:-------------:|
| Operating System     | Windows 10 or 11 | 
| RAM | Minimum 16 GB or more|
| Disk Space | 14 GB for downloading and installing the models | 
| CPU | Any modern CPU with at least 4 cores is recommended |
| GPU(Optional) | A GPU is not required for running this project, but it can improve performance |


# How to Download and Run Project?

https://github.com/santhoshmlops/DomainSpecific-MultimodalRagApplication/assets/133121635/36f60825-ed4e-44d4-a13a-3d2d969ad3fa

### You will need to copy and paste the following code into your terminal :

### STEP 01 - Clone this repository:
This step takes some time to clone this Repository and download the required dependencies :

Dependency model and embeddings file size - 5Gb. 
```bash
git clone https://github.com/santhoshmlops/DomainSpecific-MultimodalRagApplication.git && cd DomainSpecific-MultimodalRagApplication && bash setup.sh
```

### STEP 02 - Create a conda environment or python environment:

```bash
conda create -p venv python=3.11 -y
```

```bash
conda activate venv/
```
or

```bash
python -m venv venv
```

```bash
venv\Scripts\activate
```

### STEP 03 - Install the Requirements : 
```bash
pip install -r requirements.txt
```
### STEP 04 - Download and install Ollama for additional Local Inferencing : 

Download and install Ollama.exe [Download link](https://ollama.com/download)

```bash
ollama run gemma:2b
```
### STEP 05 - Run the Streamlit Application : 
```bash
streamlit run app.py
```

# Link to view this project Github code and Hugging face model

| UseCase   |      GitHub     |  Hugging Face |
|----------|:-------------:|------:|
| Fine-tune Gemma 2B-it for QLoRa Quantization via Supervised Fine Tuning |  [Code](https://github.com/santhoshmlops/MyHF_LLM_FineTuning/blob/main/Project-Gemma-Fine-Tuning/Skai_Skai_gemma_2b_it_SFT_FineTuning.ipynb) | [Model](https://huggingface.co/santhoshmlops/Skai-gemma-2b-it-SFT) |
| For CPU inference, convert the Hugging Face Model to GGUF format using the LLAMA.CPP  |  [Code](https://github.com/santhoshmlops/MyHF_LLM_FineTuning/blob/main/Project-Gemma-Fine-Tuning/Skai_Skai_gemma_2b_it_SFT_LLAMACPP_FineTuning.ipynb) | [Model](https://huggingface.co/santhoshmlops/Skai-gemma-2b-it-SFT-GGUF) |
| Custom datasets with chat template for fine-tuning |  [Code](https://github.com/santhoshmlops/MyHF_LLM_FineTuning/blob/main/Project-Gemma-Fine-Tuning/Skai_Gemma_2B_it_ChatTemplate.ipynb) | [Model](https://huggingface.co/datasets/santhoshmlops/Skai_Gemma_Instruct_ChatTemplate) |


