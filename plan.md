# **Project Plan: AI for Early Weather Prediction from Sky Images**

## **Introduction and Project Goal**

This project aims to leverage low-cost, upward-facing sky camera images to predict future precipitation events (storms) in Baxley, GA. The core idea is to fine-tune a pre-trained Vision Transformer (ViT) model, enabling it to recognize subtle atmospheric cues related to incoming storms.

Some technical challenges during the AI modeling include:

1. Generating accurate ground-truth labels for the images using historical weather station data (ASOS) and a defined look-ahead time  
2. Choosing the effective model for feature extraction and fine-tuning  
3. Spatial and temporal generalization

### **Prediction Modes**

The framework must support two distinct prediction modes, selectable via a task\_mode parameter:

1. **Classification (task\_mode='classification'):** Predicts a binary outcome (0: No Storm, 1: Storm Coming). This requires setting a precipitation threshold (e.g.,  inches/hour) to define what constitutes a "storm."  
2. **Regression (task\_mode='regression'):** Predicts the actual numerical value of precipitation (e.g., total inches) that will fall during the prediction window.

## **Phase 1: Data Labeling with ASOS Observations**

This phase focuses on generating the ground-truth labels by linking the sky image timestamps to future precipitation measurements from the nearby ASOS station.

| Function Name | Goal | Input Signature | Output Signature |
| :---- | :---- | :---- | :---- |
| extract\_timestamp\_from\_image | Parses the datetime object from an image filename, assuming a standard format like YYYYMMDD\_HHMMSS.jpg. | image\_path: str | image\_datetime: datetime.datetime |
| load\_and\_clean\_asos\_data | Loads the historical ASOS CSV data, ensuring the timestamp column is correctly parsed as datetime objects and cleaning any necessary precipitation codes. | asos\_file\_path: str | asos\_df: pd.DataFrame (Columns: timestamp, precipitation\_value) |
| generate\_prediction\_target | Determines the precipitation target by querying the ASOS data for the period starting at image\_datetime plus lead\_time\_min. | image\_datetime: datetime.datetime, asos\_df: pd.DataFrame, lead\_time\_min: int, mode: str ('binary' or 'regression'), precip\_threshold: float | target\_value: float or int |
| create\_final\_label\_dataset | Iterates over all images in the directory, calls the preceding functions, and compiles the final, labeled dataset CSV. | image\_dir: str, asos\_df: pd.DataFrame, lead\_time\_min: int, mode: str, precip\_threshold: float | labeled\_data\_df: pd.DataFrame (Columns: image\_path, target\_value) |

## **Phase 2: Model Building and Data Preparation**

This phase sets up the PyTorch infrastructure, defining the custom dataset and creating the data loaders necessary for fine-tuning.

| Function Name | Goal | Input Signature | Output Signature |
| :---- | :---- | :---- | :---- |
| initialize\_vit\_processor | Loads the image processing object from the Hugging Face transformers library for the chosen ViT checkpoint. | model\_checkpoint: str (e.g., 'google/vit-base-patch16-224') | processor: ViTImageProcessor |
| custom\_sky\_image\_dataset | Defines a custom PyTorch Dataset that handles image loading and applies the necessary pre-processing steps from the processor. | labeled\_data\_df: pd.DataFrame, processor: ViTImageProcessor | custom\_dataset: torch.utils.data.Dataset |
| create\_train\_test\_dataloaders | Splits the dataset into training and validation sets and wraps them in PyTorch DataLoader objects for batching and shuffling. | custom\_dataset: Dataset, batch\_size: int, test\_split: float | train\_loader: DataLoader, val\_loader: DataLoader |
| initialize\_model\_head | Loads the pre-trained ViT base and configures the final layer (the head) based on the task\_mode. (Regression: 1 output; Classification: 2 outputs). | model\_checkpoint: str, task\_mode: str ('regression' or 'classification') | model: PreTrainedModel |

## **Phase 3: Training and Optimization**

This is the core fine-tuning process. The student will implement a robust training loop with explicit separation of optimization setup and the per-epoch step.

| Function Name | Goal | Input Signature | Output Signature |
| :---- | :---- | :---- | :---- |
| define\_optimization\_setup | Instantiates the loss function (criterion) and the optimizer (e.g., AdamW) needed for training. | model: torch.nn.Module, learning\_rate: float, task\_mode: str | criterion: nn.Module, optimizer: torch.optim.Optimizer |
| run\_epoch\_step | Executes one full forward and backward pass (if training) over the provided data loader, accumulating loss and metrics. | model: torch.nn.Module, data\_loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, is\_training: bool | avg\_loss: float, epoch\_metrics: dict |
| execute\_full\_fine\_tuning | Manages the entire training process across multiple epochs, running both training and validation steps, and handling model saving and early stopping logic. | model: torch.nn.Module, train\_loader: DataLoader, val\_loader: DataLoader, num\_epochs: int, criterion, optimizer, patience: int, task\_mode: str | best\_model\_path: str |

## **Phase 4: Evaluation and Visualization**

Evaluation confirms the model's utility, and visualization (using methods like Integrated Gradients, as explored in the previous work) provides crucial interpretability .

| Function Name | Goal | Input Signature | Output Signature |
| :---- | :---- | :---- | :---- |
| compute\_performance\_metrics | Calculates and reports appropriate performance metrics on the validation/test set based on the task\_mode. | model: torch.nn.Module, test\_loader: DataLoader, task\_mode: str | metrics\_report: dict (Keys: 'Accuracy', 'F1-Score', 'RMSE', etc.) |
| generate\_attribution\_map | Uses an interpretability library (e.g., Captum) to generate a saliency map showing pixel contribution to the prediction for a single image. | model: torch.nn.Module, sample\_image: torch.Tensor, target\_index\_or\_value: float | attribution\_heatmap: np.array |
| visualize\_results | Generates and saves visual reports: metric plots (e.g., Confusion Matrix/Scatter Plot) and the attribution map overlaid on the original image. | attribution\_heatmap: np.array, original\_image: PIL.Image, metrics\_report: dict, predictions: list, targets: list | plot\_saved\_paths: dict |

