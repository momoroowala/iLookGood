import torch
import torchvision.models as models
from torch import nn
from torch.utils.data import DataLoader
import multiprocessing

from torchvision.models.resnet import ResNet50_Weights
from tqdm import tqdm
import csv

# ... (rest of your imports and function definitions)

def main():
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    # Load pretrained ResNet50
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1000)
    for param in model.parameters():
        param.requires_grad = False

    #unfreeze last 2 layers (including FC layer)
    #for param in model.layer2.parameters():
    #   param.requires_grad = True
    for param in model.layer3.parameters():
       param.requires_grad = True
    for param in model.layer4.parameters():
       param.requires_grad = True
    for param in model.fc.parameters():
       param.requires_grad = True

       
    train_dataset = torch.load('train_dataset.pt')
    val_dataset = torch.load('val_dataset.pt')
    #test_dataset = torch.load('test_dataset.pt')
    
    def transform_labels(labels):
        return (labels + 1) / 2
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    print("Train Data Loaded")
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)
    print("Validation Data Loaded")

    #No Testing for Now
    #test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)
    #print("Test Data Loaded")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    train_losses = []
    val_losses = []
    num_epochs = 10

    #Training Loop
    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            labels = transform_labels(labels)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")

        #Evaluation Loop
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                labels = transform_labels(labels)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item()

        val_epoch_loss = val_running_loss / len(val_loader)
        val_losses.append(val_epoch_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_epoch_loss:.4f}")

    # Save the final model
    torch.save(model.state_dict(), 'DFResNet_AllUF_300kIMG.pth')
    print("Final model saved as 'DFResNet_AllUF_300kIMG.pth'")

    # Save losses to CSV
    with open('training_validation_losses_AllUF.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Training Loss", "Validation Loss"])
        for i in range(num_epochs):
            writer.writerow([i+1, train_losses[i], val_losses[i]])
    print("Losses saved to 'training_validation_losses_300k3UF.csv'")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()