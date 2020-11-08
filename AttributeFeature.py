import torch
import numpy as np
import matplotlib.pyplot as plt

import LoadData


class ExtractAttributeFeature(torch.nn.Module):
    def __init__(self):
        super(ExtractAttributeFeature, self).__init__()
        embedding_weight=self.getEmbedding()
        self.embedding_size=embedding_weight.shape[1]
        self.embedding=torch.nn.Embedding.from_pretrained(embedding_weight)
        # embedding size 200->100
        self.Linear_1 = torch.nn.Linear(self.embedding_size, int(self.embedding_size/2))
        # embedding size 100->1
        self.Linear_2 = torch.nn.Linear(int(self.embedding_size/2),1)

    def forward(self, input):
        """
        e(a_i)
        """
        # -1 represent batch size
        self.embedded=self.embedding(input).view(-1, 5, self.embedding_size)
        """
        a_i=W_2*tanh(W_1*e(a_i)+b_1)+b_2
        """
        attn_weights = self.Linear_1(self.embedded.view(-1,self.embedding_size))
        # attn_weights = torch.nn.functional.tanh(attn_weights)
        attn_weights = torch.nn.functional.relu(attn_weights)
        attn_weights = self.Linear_2(attn_weights)
        """
        a=softmax(a)
        """
        attn_weights = torch.nn.functional.softmax(attn_weights.view(-1,5),dim=1)
        finalState = torch.bmm(attn_weights.unsqueeze(1), self.embedded).view(-1,200)
        return finalState,self.embedded

    def getEmbedding(self):
        return torch.from_numpy(np.loadtxt("multilabel_database_embedding/vector.txt", delimiter=' ', dtype='float32'))

if __name__ == "__main__":
    test=ExtractAttributeFeature()
    for text_index,image_feature,attribute_index,group,id in LoadData.train_loader:
        result,seq=test(attribute_index)
        # [2, 200]
        print(result.shape)
        # [2, 5, 200]
        print(seq.shape)
        break


