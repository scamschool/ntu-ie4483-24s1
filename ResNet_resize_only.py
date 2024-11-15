import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torchvision.datasets.folder import default_loader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd 
import os
from tqdm import tqdm
from PIL import Image

# Define default hyperparameters
default_params = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'num_epochs': 25,
    'patience': 5,
}

def main():

    os.makedirs('plots', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    #Set Up Data Preprocessing without Data Augmentation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
        'test2': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
    }

    data_dir = 'datasets/datasets'
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train']),
        'val': datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val']),
        'test': datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test']),
    }

    class_names = image_datasets['train'].classes
    print("Class names:", class_names)

    # Custom dataset for test2 (unlabeled images)
    class UnlabeledImageDataset(torch.utils.data.Dataset):
        def __init__(self, img_dir, transform=None):
            self.img_dir = img_dir
            self.transform = transform
            self.img_paths = [os.path.join(img_dir, fname) for fname in sorted(os.listdir(img_dir))
                              if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            self.loader = default_loader

        def __len__(self):
            return len(self.img_paths)

        def __getitem__(self, idx):
            path = self.img_paths[idx]
            image = self.loader(path)
            if self.transform is not None:
                image = self.transform(image)
            return image, os.path.basename(path)

    # Load test2 dataset
    test2_dir = 'datasets/datasets/test2'
    test2_dataset = UnlabeledImageDataset(
        test2_dir,
        data_transforms['test2']
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
    else:
        print("CUDA is not available. Using CPU.")

    # Create class_to_label mapping
    class_to_label = {}
    for idx, class_name in enumerate(class_names):
        if class_name.lower() in ['cat', 'cats']:
            class_to_label[idx] = 0
        elif class_name.lower() in ['dog', 'dogs']:
            class_to_label[idx] = 1
        else:
            print(f"Unknown class name: {class_name}")

    # Set hyperparameters from default_params
    learning_rate = default_params['learning_rate']
    batch_size = default_params['batch_size']
    num_epochs = default_params['num_epochs']
    patience = default_params['patience']

    #Load Data with the specified Batch Size
    dataloaders = {
        'train': torch.utils.data.DataLoader(
            image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
        ),
        'val': torch.utils.data.DataLoader(
            image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
        ),
        'test': torch.utils.data.DataLoader(
            image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
        ),
        'test2': torch.utils.data.DataLoader(
            test2_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
        ),
    }

    # Load Pretrained ResNet Model and Modify ffc layer
    model_ft = models.resnet152(pretrained=True)
    for param in model_ft.parameters():
        param.requires_grad = False

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    model_ft = model_ft.to(device)

    # Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(
        model_ft.fc.parameters(),
        lr=learning_rate
    )

    # Scheduler with default parameters
    scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # Train the Model
    model_ft, best_acc, history = train_model(
        model_ft, criterion, optimizer_ft, scheduler,
        dataloaders, device, num_epochs=num_epochs, patience=patience
    )

    # Plot Training and Validation Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(history['train_acc']) + 1), history['train_acc'], label='Training Accuracy')
    plt.plot(range(1, len(history['val_acc']) + 1), history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig('plots/training_validation_accuracy.png')
    plt.show()
    plt.close()

    # Plot Training and Validation Loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(history['train_loss']) + 1), history['train_loss'], label='Training Loss')
    plt.plot(range(1, len(history['val_loss']) + 1), history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('plots/training_validation_loss.png')
    plt.show()
    plt.close()

    # Evaluate on Validation Set
    print("\nEvaluating on Validation Set:")
    evaluate_model(model_ft, dataloaders['val'], device, class_names, dataset_type='Validation')

    # Evaluate on Test Set
    print("\nEvaluating on Test Set:")
    evaluate_model(model_ft, dataloaders['test'], device, class_names, dataset_type='Test')

    # Predict on test2 dataset and save results to excel
    print("\nPredicting on test2 Dataset:")
    predict_test2(model_ft, dataloaders['test2'], device, class_to_label)

# Adjusted train_model function
def train_model(model, criterion, optimizer, scheduler, dataloaders, device, num_epochs=25, patience=5):
    best_model_wts = model.state_dict()
    best_acc = 0.0
    best_loss = float('inf')
    early_stop_counter = 0

    # Track losses and accuracies
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            loop = tqdm(dataloaders[phase], desc=f'Epoch {epoch+1}/{num_epochs} - {phase.capitalize()}', leave=False)

            for inputs, labels in loop:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                loop.set_postfix(loss=running_loss / len(dataloaders[phase].dataset),
                                 accuracy=running_corrects.double() / len(dataloaders[phase].dataset))

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            # Save loss and accuracy for analysis
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.cpu()) 
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.cpu()) 

                # Early stopping check
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

                # Check if early stopping is triggered
                if early_stop_counter >= patience:
                    print(f"Early stopping triggered after epoch {epoch+1}")
                    model.load_state_dict(best_model_wts)
                    return model, best_acc.item(), history

        # epoch summary
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {history['train_loss'][-1]:.4f}, Train Acc: {history['train_acc'][-1]:.4f} - "
              f"Val Loss: {history['val_loss'][-1]:.4f}, Val Acc: {history['val_acc'][-1]:.4f}")

    model.load_state_dict(best_model_wts)
    return model, best_acc.item(), history

# Adjusted evaluate_model function
def evaluate_model(model, dataloader, device, class_names, dataset_type='Validation'):
    model.eval()
    y_true = []
    y_pred = []
    running_loss = 0.0
    running_corrects = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=f"{dataset_type} Progress", leave=True):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    dataset_size = len(dataloader.dataset)
    eval_loss = running_loss / dataset_size
    eval_acc = running_corrects.double() / dataset_size

    print(f'{dataset_type} Loss: {eval_loss:.4f} {dataset_type} Acc: {eval_acc:.4f}')

    # Generate and save classification report
    eval_report = classification_report(y_true, y_pred, target_names=class_names)
    with open(f'reports/{dataset_type.lower()}_classification_report.txt', 'w') as f:
        f.write(eval_report)
    print(eval_report)

    # Generate and save confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{dataset_type} Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'plots/{dataset_type.lower()}_confusion_matrix.png')
    plt.show()
    plt.close()

# Adjusted predict_test2 function
def predict_test2(model, dataloader, device, class_to_label):
    model.eval()
    file_names = []
    predictions = []

    with torch.no_grad():
        for inputs, fnames in tqdm(dataloader, desc="Predicting on test2", leave=True):
            inputs = inputs.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Map class indices to 0 and 1
            pred_labels = [class_to_label.get(p, -1) for p in preds.cpu().numpy()]

            # Extract IDs from file names
            ids = [os.path.splitext(fname)[0] for fname in fnames]

            # Append the results
            file_names.extend(ids)
            predictions.extend(pred_labels)

    # Create a DataFrame
    results_df = pd.DataFrame({
        'ID': file_names,
        'Predicted Label': predictions  # Predictions are 0 for cat, 1 for dog
    })

    try:
        results_df['ID'] = results_df['ID'].astype(int)
    except ValueError:
        pass 

    # Sort the DataFrame by ID
    results_df.sort_values(by='ID', inplace=True)

    # Reset index after sorting
    results_df.reset_index(drop=True, inplace=True)

    # Save to Excel
    results_df.to_excel('reports/test2_predictions.xlsx', index=False)
    print("\nPredictions saved to reports/test2_predictions.xlsx")

if __name__ == '__main__':
    main()
