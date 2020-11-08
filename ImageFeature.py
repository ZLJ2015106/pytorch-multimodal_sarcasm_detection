import torch
import numpy as np
import matplotlib.pyplot as plt

import LoadData


class ExtractImageFeature(torch.nn.Module):
    def __init__(self):
        super(ExtractImageFeature, self).__init__()
        # 2048->1024
        self.Linear = torch.nn.Linear(2048, 1024)

    def forward(self, input):
        input=input.permute(1,0,2)
        output=list()
        for i in range(196):
            sub_output=torch.nn.functional.relu(self.Linear(input[i]))
            output.append(sub_output)
        output=torch.stack(output)
        mean=torch.mean(output,0)
        return mean,output

if __name__ == "__main__":
    test=ExtractImageFeature()
    for text_index,image_feature,attribute_index,group,id in LoadData.train_loader:
        result,seq=test(image_feature)
        # [2, 1024]
        print(result.shape)
        # [196, 2, 1024]
        print(seq.shape)
        break


