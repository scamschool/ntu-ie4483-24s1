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

default_params = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'num_epochs': 25,
    'patience': 5, 
}

def main():
    os.makedirs('plots', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((313, 233)),      
            transforms.CenterCrop(224),         
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.Resize((313, 233)),     
            transforms.CenterCrop(224),        
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
        'test': transforms.Compose([
            transforms.Resize((313, 233)),      
            transforms.CenterCrop(224),      
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
        'test2': transforms.Compose([
            transforms.Resize((313, 233)),    
            transforms.CenterCrop(224),  
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

    learning_rate = default_params['learning_rate']
    batch_size = default_params['batch_size']
    num_epochs = default_params['num_epochs']
    patience = default_params['patience']

    #Load Data
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

    # Load pretrained ResNet model and modify ffc
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

    scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # Train model
    model_ft, _ = train_model(
        model_ft, criterion, optimizer_ft, scheduler,
        dataloaders, device, num_epochs=num_epochs, patience=patience
    )

    print("\nEvaluating on Validation Set:")
    evaluate_model(model_ft, dataloaders['val'], device, class_names, dataset_type='Validation')

    print("\nEvaluating on Test Set:")
    evaluate_model(model_ft, dataloaders['test'], device, class_names, dataset_type='Test')

    print("\nPredicting on test2 Dataset:")
    predict_test2(model_ft, dataloaders['test2'], device, class_to_label)

#train_model function
def train_model(model, criterion, optimizer, scheduler, dataloaders, device, num_epochs=25, patience=5):
    best_model_wts = model.state_dict()
    best_acc = 0.0
    best_loss = float('inf')
    early_stop_counter = 0

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
                    print(f"Early stopping triggered after {epoch+1} epochs!")
                    model.load_state_dict(best_model_wts)
                    plot_metrics(history)
                    check_overfit_underfit(history) 
                    return model, best_acc.item()

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        if early_stop_counter >= patience:
            break

    model.load_state_dict(best_model_wts)
    plot_metrics(history)
    check_overfit_underfit(history)

    return model, best_acc.item()

def plot_metrics(history):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 4))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.tight_layout()

    # Save the plot to a file
    plt.savefig('plots/training_validation_metrics.png')

    plt.show()
    plt.close()

def check_overfit_underfit(history, overfit_threshold=0.1, underfit_threshold=0.7):
    train_loss = history['train_loss'][-1]
    val_loss = history['val_loss'][-1]
    train_acc = history['train_acc'][-1]
    val_acc = history['val_acc'][-1]

    print("\nModel Analysis:")

    if train_acc > val_acc + overfit_threshold and val_loss > train_loss:
        print("Model is overfitting.")

    elif train_acc < underfit_threshold and val_acc < underfit_threshold:
        print("Model is underfitting.")

    elif abs(train_loss - val_loss) < 0.01:
        print("Model has good generalization.")

    else:
        print("Model is performing normally with balanced training and validation.")

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

# Function to predict on the test2 dataset and save results to Excel
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
            pred_labels = [class_to_label[p] for p in preds.cpu().numpy()]

            ids = [os.path.splitext(fname)[0] for fname in fnames]

            # Append the results
            file_names.extend(ids)
            predictions.extend(pred_labels)

    results_df = pd.DataFrame({
        'ID': file_names,
        'Predicted Label': predictions  # Predictions are 0 for cat, 1 for dog
    })

    try:
        results_df['ID'] = results_df['ID'].astype(int)
    except ValueError:
        pass 

    results_df.sort_values(by='ID', inplace=True)

    results_df.reset_index(drop=True, inplace=True)

    results_df.to_excel('reports/test2_predictions.xlsx', index=False)
    print("\nPredictions saved to reports/test2_predictions.xlsx")

if __name__ == '__main__':
    main()
