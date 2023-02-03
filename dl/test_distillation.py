import torch
import torch.nn as nn
import torch.optim as optim

# Define the teacher model
class TeacherModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(30, 100)
        self.fc2 = nn.Linear(100, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Define the student model
class StudentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(30, 50)
        self.fc2 = nn.Linear(50, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Define the loss function for knowledge distillation
def distillation_loss(student_output, teacher_output, target, T):
    # Compute the soft labels from the teacher model
    soft_labels = nn.functional.log_softmax(teacher_output/T, dim=1)
    print(soft_labels.shape)
    # Compute the loss using the soft labels
    loss = nn.functional.kl_div(nn.functional.log_softmax(student_output/T, dim=1), soft_labels, reduction='batchmean')
    # Add the auxiliary loss
    
    loss += nn.functional.cross_entropy(student_output, target, reduction='mean')/T**2
    return loss

# Load the data
data = torch.randn(1000, 30)
labels = torch.randint(0, 10, size=(1000,))

# Define the models
teacher_model = TeacherModel()
student_model = StudentModel()

# Define the optimizer
optimizer = optim.SGD(student_model.parameters(), lr=0.01)

# Train the student model
T = 4
for epoch in range(100):
    optimizer.zero_grad()
    output = student_model(data)
    teacher_output = teacher_model(data)
    loss = distillation_loss(output, teacher_output, labels, T)
    loss.backward()
    optimizer.step()
