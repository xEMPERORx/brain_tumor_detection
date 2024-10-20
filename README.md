# Brain Tumor Classification

This repository contains a project for classifying brain tumors using Convolutional Neural Networks (CNNs). The project utilizes two different models: a custom CNN and a pre-trained ResNet50 model. The custom CNN is designed from scratch, while the ResNet50 model is used for transfer learning.

## Project Structure

- `Dataset/`: Contains the training and testing datasets.
- `main.py`: The main script to run the Streamlit application for predictions.
- `brain_tumor.h5`: The saved custom CNN model.
- `brain_tumor_resnet50.h5`: The saved ResNet50 model.
- `notebook.ipynb`: Jupyter notebook containing the code for training and evaluating the models.
- `README.md`: This readme file.

## Models

### Custom CNN

The custom CNN is built from scratch and consists of multiple convolutional layers followed by max-pooling layers, a flattening layer, and dense layers. The model is trained on the dataset and achieves a good accuracy.

### ResNet50

The ResNet50 model is a pre-trained model used for transfer learning. The top layers are replaced with custom layers suitable for our classification task. The model is fine-tuned on the dataset and achieves a high accuracy.

## Results

### Custom CNN

- **Accuracy**: 87.65%
- **Loss**: 0.3967

![Custom CNN Accuracy](images/custom_cnn_accuracy.png)
![Custom CNN Loss](images/custom_cnn_loss.png)
![CNN Confusion Matrix](images/confusion_cnn.png)
### ResNet50

- **Accuracy**: 93.42%
- **Loss**: 0.2345

![ResNet50 Accuracy](images/resnet50_accuracy.png)
![ResNet50 Loss](images/resnet50_loss.png)
![ResNet50 Confusion Matrix](images/confusion_resnet50.png)
### Conclusion
This project demonstrates the power of deep learning, particularly Convolutional Neural Networks (CNNs), in solving complex image classification problems. By leveraging a combination of pre-trained models like ResNet50 and custom convolutional layers, we successfully build a robust pipeline for training, evaluating, and deploying an image classification model. The use of transfer learning significantly reduces training time and improves model performance, especially when working with limited data.

## Usage

To use the model for predictions, run the `main.py` script using Streamlit:

```bash
streamlit run main.py
```

This will start a web application where you can upload images and get predictions for brain tumor classification.

## License

This project is licensed under the MIT License.

