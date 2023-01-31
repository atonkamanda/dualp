import torch
import torch.nn as nn
import torch.optim as optim

class RLClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RLClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model, data, target, optimizer, criterion):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return loss

def predict(model, data):
    output = model(data)
    return output.argmax()

def reward(prediction, target):
    return 1 if prediction == target else -1

# Define model, optimizer and criterion
input_size = 1
hidden_size = 10
output_size = 2
model = RLClassifier(input_size, hidden_size, output_size)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Train the model
for i in range(1000):
    # Generate a random digit long int
    data = torch.randint(0, 10, (1,)).long()
    target = (data % 2).long()  # 0 for even, 1 for odd
    # 
    # Get prediction and calculate reward
    prediction = predict(model, data)
    r = reward(prediction, target)

    # Train the model with the reward as the target
    train(model, data, torch.tensor([r]), optimizer, criterion)

# Test the model
for i in range(10):
    data = torch.tensor([i])
    prediction = predict(model, data)
    #print(str(i)+ "is classified as" str(prediction))
    #print(f"{i} is classified as {"even" if prediction == 0 else "odd"}")
