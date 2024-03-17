import torch
import torch.nn as nn
import torch.nn.functional as F
from CustomLayers.Convolution import Convolution
from CustomLayers.Activation import RectifiedLinearUnit
from CustomLayers.MaxPool import MaxPool2d
from CustomLayers.Loss import CrossEntropyLoss
from CustomLayers.Dropout import Dropout
from CustomLayers.Flatten import Flatten
from CustomLayers.Dense import Dense
from CustomLayers.Activation import LogSoftmax

from CustomLayers.Loss import LogSoftmax

from torchsummary import summary
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity


# class ASLClassifier(nn.Module):
#     def __init__(self, num_classes=27):
#         super(ASLClassifier, self).__init__()
#         self.features = nn.Sequential(
#             Convolution(in_channels=3, out_channels=32, kernel_size=3, bias=False),
#             nn.BatchNorm2d(32),
#             # Convolution(in_channels=32, out_channels=32, kernel_size=3),
#             RectifiedLinearUnit(),
#             MaxPool2d(kernel_size=2, stride=2),
#             Dropout(0.2),
#
#             Convolution(in_channels=32, out_channels=64, kernel_size=3, bias=False),
#             nn.BatchNorm2d(64),
#             # Convolution(in_channels=64, out_channels=64, kernel_size=3),
#             RectifiedLinearUnit(),
#             MaxPool2d(kernel_size=2, stride=2),
#             Dropout(0.2),
#
#             Convolution(in_channels=64, out_channels=64, kernel_size=3, bias=False),
#             nn.BatchNorm2d(64),
#             # Convolution(in_channels=64, out_channels=64, kernel_size=3),
#             RectifiedLinearUnit(),
#             MaxPool2d(kernel_size=2, stride=2),
#             Dropout(0.5),
#             nn.AdaptiveAvgPool2d((7, 7))
#         )
#         self.classifier = nn.Sequential(
#             Flatten(),
#             Dense(64*7*7, 128),
#             RectifiedLinearUnit(),
#             Dropout(0.5),
#             Dense(128, num_classes),
#             LogSoftmax(dim=1)
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x
# #

class ASLClassifier(nn.Module):
    def __init__(self, num_classes=27, freeze_pretrained=True):
        super(ASLClassifier, self).__init__()

        # Load pretrained ResNet model without final fully connected layers
        # pretrained_resnet = models.resnet50(pretrained=True)
        # pretrained_resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        # self.features = nn.Sequential(*list(pretrained_resnet.children())[:-1])

        # Load pretrained VGG16 model without final fully connected layers
        pretrained_vgg = models.vgg16(weights='VGG16_Weights.DEFAULT')
        self.features = nn.Sequential(*list(pretrained_vgg.children())[:-1])

        # Freeze the pretrained layers
        if freeze_pretrained:
            for param in self.features.parameters():
                param.requires_grad = False

        # Add your own layers on top of the pretrained layers
        self.classifier_layers = nn.Sequential(
            # Dense(pretrained_vgg.classifier[-1].in_features, 512),
            Dense(7 * 7 * 512, 512),
            RectifiedLinearUnit(),
            Dropout(0.5),
            Dense(512, num_classes)
            # nn.LogSoftmax(dim=1)
        )

        # Flatten layer to flatten the tensor before passing it to fully connected layers
        # self.flatten = Flatten()
        # self.flatten = lambda x:  x.view(x.size(0),-1)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten feature map
        # x = self.flatten(x)
        x = self.classifier_layers(x)
        return x
#
#     def simple_summary(self):
#         total_parameters = 0
#         input_size = (3, 224, 224)
#         print(f"Model name: \"ASL Classifier\"")
#         print(f"Input size: {input_size}")
#         print(f"{'Layer (type)':<20}{'Input Shape':<20}"
#               f"{'Output Shape':<20}{'Param#':<10}\n{'-'*75}")
#         for name, layer in self.named_children():
#             input_tensor = torch.zeros(1, *input_size)
#             output = layer(input_tensor)
#             output_shape = tuple(output.size())[1:]
#             params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
#             total_parameters += params
#             print(f"{name:<20}{str(input_size):<20}{str(output_shape):<20}{params:<10}")
#             input_size = output_shape
#             print('-'*75)
#         print(f"Total number of parameters: {total_parameters}")



if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()

    # Define the device
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"The device used is: {device}")

    # Create the model
    model = ASLClassifier().cuda()

    # Sample input tensor (random initialization)
    input_tensor = torch.randn(1, 3, 224, 224).cuda()

    # Call the summary method
    # Won't work for Transfer Learning
    # print("\nThe basic summary is: \n")
    # model.simple_summary()
    # print("\n\nThe complete summary is: \n")
    # model.summary()

    print("\n\nThe model summary by torchsummary is: \n")
    summary(model, input_size=(3, 224, 224))

    with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            model(input_tensor)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
