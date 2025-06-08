import torch.nn as nn
from torchvision import models

class MTL(nn.Module):
    def __init__(self, num_classes_school, num_classes_type):
        super(MTL, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet_feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.num_features = resnet.fc.in_features

        self.class_school_head = nn.Sequential(
            nn.Linear(self.num_features, num_classes_school)
        )
        self.class_type_head = nn.Sequential(
            nn.Linear(self.num_features, num_classes_type)
        )

    def forward(self, img):
        visual_emb = self.resnet_feature_extractor(img)
        visual_emb = visual_emb.view(visual_emb.size(0), -1)

        out_school = self.class_school_head(visual_emb)
        out_type = self.class_type_head(visual_emb)

        return out_school, out_type
