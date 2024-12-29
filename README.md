# **Neural Image Captioning**

## **Problem Statement**
Automatically describing the content of an image is a fundamental problem in artificial intelligence that connects computer vision and natural language processing. This project implements a working solution inspired by the research paper [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/pdf/1502.03044.pdf).

---

## **What is Image Captioning?**
Image captioning involves:
- Building networks capable of perceiving contextual subtleties in images.
- Relating observations to both the scene and the real world.
- Producing succinct and accurate image descriptions.

---

## **Methodology**
The task of image captioning is divided into two key modules:

1. **Encoder - Image-Based Model**:
    - **Task**: Extract features from the image and generate feature vectors.
    - **Input**: Source images from the dataset.
    - **Output**: Flattened vectors of image features for the language model.
    - **Model/Technique**: Transfer Learning using Inception-ResNet-V2.

2. **Decoder - Language-Based Model**:
    - **Task**: Translate extracted features into a natural language caption.
    - **Input**: Flattened image feature vectors.
    - **Output**: Caption for the image.
    - **Model/Technique**: Bi-Directional Long Short-Term Memory (LSTM) with Bahdanau Attention.
    - **Embeddings**: GloVe Embeddings (6B.300d).

### **System Flow Diagram**
![System Flow Diagram](./docs/system_flow.png)

### **Attention Working Example**
![Attention Working Example](./docs/attention_explain.png)

---

## **Dependencies**
1. **Environment**:
    - Anaconda3 or Miniconda3.
    - Python 3.9+ (ideally compatible with 3.6+).
2. **Libraries**:
    - All libraries listed in [requirements.txt](requirements.txt).
3. **Datasets and Pretrained Models**:
    - Dataset: [Flickr30k dataset](https://www.kaggle.com/hsankesara/flickr-image-dataset).
    - Word Embeddings: [GloVe (6B or 42B or 840B)](https://nlp.stanford.edu/projects/glove/).

---

## **Project Structure**
```
    ├── data
    │   ├── flickr30k_images                <- Flickr30k Dataset.
    │   └── captions.csv                    <- Captions for the Flickr30k Dataset. Downloaded by default.
    │
    ├── notebooks
    │   ├── glove.xB.xxxd.txt               <- GloVe Embeddings.
    │   └── neural_image_captioning.ipynb   <- Main Notebook to run.
    │
    ├── models
    │   └── checkpoint                      <- Model checkpoints for reuse.
    │
    └── docs
        ├── system_flow.png                 <- System Flow Diagram.
        ├── attention_explain.png           <- Attention Mechanism Example.
        ├── sample_input.png                <- Sample input from the dataset.
        ├── sample_output.png               <- Predicted caption output.
        ├── belu.png                        <- BELU Performance Metric Visualization.
```

---

## **How to Run the Project**
1. Install all dependencies as listed in `requirements.txt`.
2. Organize data and embeddings into the folder structure shown above.
3. Run the [neural_image_captioning.ipynb](./notebooks/neural_image_captioning.ipynb) notebook in a `jupyter-notebooks` or `jupyter-lab` session.
4. Use model checkpoints in the [models/checkpoint](./models/checkpoint) folder to export or improve the model.

---

## **Sample Input**
The dataset includes 31,783 images and 158,915 captions, with 5 captions per image.

![Sample Input](./docs/sample_input.png)

---

## **Sample Output**
The output contains:
1. **Predicted Caption**
2. **Attention Map Over Image**
3. **Original Image**

![Sample Output](./docs/sample_output.png)

---

## **Performance Indicator**
The **BELU** (Bilingual Evaluation Understudy) score measures the similarity between the predicted and reference captions. BELU is calculated by comparing the n-grams of candidate captions with those of reference captions, producing a score between 0 and 1 (closer to 1 indicates higher similarity).

### **BELU Metric Example**
![BELU](./docs/belu.png)

---

## **Future Improvements**
- Adding more Bidirectional LSTM layers.
- Training the model for more epochs (e.g., 50 epochs as mentioned in the research paper).
- Incorporating state-of-the-art NLP techniques such as transformers (e.g., BERT).
- Using a larger dataset like MSCOCO or applying data augmentation.

---

## **References**
### **Papers**
1. [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/pdf/1502.03044.pdf)
2. [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/pdf/1406.1078.pdf)
3. [Long-term Recurrent Convolutional Networks for Visual Recognition and Description](https://arxiv.org/pdf/1411.4389.pdf)
4. [Deep Visual-Semantic Alignments for Generating Image Descriptions](http://proceedings.mlr.press/v37/xuc15.pdf)

### **Implementations**
- [DeepRNN/image_captioning](https://github.com/DeepRNN/image_captioning)
- [TensorFlow Tutorial](https://www.tensorflow.org/tutorials/text/image_captioning#caching_the_features_extracted_from_inceptionv3)

### **Videos**
- [Neural Image Caption Generation with Visual Attention (algorithm) | AISC](https://www.youtube.com/watch?v=ENVGHs3yw7k)
- [Building an Image Captioner with Neural Networks](https://www.youtube.com/watch?v=c_bVBYxX5EU)
