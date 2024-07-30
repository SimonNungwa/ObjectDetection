import torch
import torch.optim as optim
from src.models.yolo import YOLOv8
from src.utils.data_loader import get_dataloader

def train_model():
    model = YOLOv8()
    dataloader = get_dataloader("data/processed/images", "data/processed/annotations")
    
    criterion = torch.nn.CrossEntropyLoss() # Adjust as necessary
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 10
    for epoch in range(num_epochs):
        for images, annotations in dataloader:
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, annotations)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

if __name__ == "__main__":
    train_model()
