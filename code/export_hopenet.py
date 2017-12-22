from torch.autograd import Variable

import torch.onnx
import torchvision

import hopenet

# generate a random input
dummy_input = Variable(torch.randn(16, 3, 224, 224)).cuda(0)

# load hopenet model
model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck,
                        [3, 4, 6, 3], 66)

saved_state_dict = torch.load('../snapshots/hopenet_alpha1.pkl')
model.load_state_dict(saved_state_dict)
model.cuda(0)

# export network
torch.onnx.export(model, dummy_input, './output/hopenet_alpha1_onnx.pb',
                  verbose=True)
