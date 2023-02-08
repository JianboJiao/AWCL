import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class AWCL_model(nn.Module):
    ''' The ResNet feature extractor + projection head for AWCL '''

    def __init__(self, base_model, out_dim, pretrained=False):
        super(AWCL_model, self).__init__()

        self.resnet_dict = {"resnet18": models.resnet18(pretrained=pretrained),
                                "resnet50": models.resnet50(pretrained=pretrained)}


        if pretrained:
            print('\nImageNet pretrained parameters loaded.\n')
        else:
            print('\nRandom initialize model parameters.\n')
        
        resnet = self._get_basemodel(base_model)
        num_ftrs = resnet.fc.in_features

        self.features = nn.Sequential(*list(resnet.children())[:-1]) # discard the last fc layer

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        h1 = h.squeeze()  # feature before project g()=h1

        x = self.l1(h1)
        x = F.relu(x)
        x = self.l2(x)

        return h1, F.normalize(x, dim=-1)
