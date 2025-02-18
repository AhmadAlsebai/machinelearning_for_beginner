# Cats or Dogs Classification

## Project Overview
This project is a deep learning model built using Convolutional Neural Networks (CNN) to classify images as either cats or dogs. The model is trained on a dataset containing labeled images of cats and dogs and aims to achieve high accuracy in distinguishing between the two.

## Dataset
The dataset used for this project consists of:
- Training set: Labeled images of cats and dogs
- Validation set: Used to fine-tune the model
- Test set: Unseen images to evaluate the model's performance
- Data Link  https://drive.google.com/drive/folders/1OFNnrHRZPZ3unWdErjLHod8Ibv2FfG1d?usp=sharing

## Requirements
To run this project, ensure you have the following dependencies installed:

```bash
pip install tensorflow keras numpy matplotlib opencv-python pandas scikit-learn
```

## Model Architecture
The CNN model consists of:
1. **Convolutional Layers**: Extract features from the input images.
2. **Pooling Layers**: Reduce spatial dimensions while maintaining key features.
3. **Fully Connected Layers**: Classify images as cats or dogs.
4. **Activation Functions**: Use ReLU for hidden layers and Softmax for the output layer.

## Training Process
- The model is trained using **cross-entropy loss** and **Adam optimizer**.
- Images are resized and augmented to improve generalization.
- The dataset is split into **training, validation, and testing sets**.

## How to Use
### Training the Model
Run the following command to train the model:

```python
python train.py
```

### Testing the Model
Run the following command to test the model:

```python
python test.py --image_path path/to/image.jpg
```

### Evaluating the Model
To check the model performance:
```python
evaluate.py
```

## Results
- The trained model achieves an accuracy of **XX%** on the test dataset.
- Sample predictions are visualized in `results/` folder.

## Future Improvements
- Experiment with deeper architectures for better accuracy.
- Fine-tune hyperparameters such as learning rate and batch size.
- Increase the dataset size for better generalization.

## License
This project is open-source and available under the MIT License.

---
**Author:** Ahmad Alsebai
