from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from utils.utils import get_dataset, get_device, get_architecture, get_input_shape, get_num_classes
from knowledgematrix.models.alexnet import AlexNet
import torch

device = 'mps'#get_device()
dataname = 'cifar10'
pretrained = False

data = get_dataset(dataname, data_loader=False)[0]
num_classes = get_num_classes(dataname)

state_dict = torch.load(f'experiments/alexnet_cifar10/weights/epoch_70.pth', map_location=device)

model = get_architecture(
            input_shape = (3,224,224),
            num_classes = num_classes,
            architecture_index = -3,
            freeze_features = False
        )
model.to(device)
model.load_state_dict(state_dict)

if pretrained:
    model = AlexNet((3,224,224), 10, pretrained=True).to(device)

#print(model)

matrix_computer = KnowledgeMatrixComputer(model, batch_size=2, device=device) #max batch_size = 150528//8

d = torch.rand((1,3,224,224)).to(device)
#print("input shape", d.shape)e

#print("model.input_shape:", model.input_shape)
#print("d.shape:", d.shape)
#input()
pred = model(d)
#mat = matrix_computer.forward(d)