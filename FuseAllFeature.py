import torch
import LoadData
import TextFeature
import AttributeFeature
import ImageFeature

class RepresentationFusion(torch.nn.Module):
    def __init__(self,att1_feature_size,att2_feature_size,att3_feature_size):
        super(RepresentationFusion, self).__init__()
        self.linear1_1 = torch.nn.Linear(att1_feature_size+att1_feature_size, int((att1_feature_size+att1_feature_size)/2))
        self.linear1_2 = torch.nn.Linear(att1_feature_size+att2_feature_size, int((att1_feature_size+att2_feature_size)/2))
        self.linear1_3 = torch.nn.Linear(att1_feature_size+att3_feature_size, int((att1_feature_size+att3_feature_size)/2))
        self.linear2_1 = torch.nn.Linear(int((att1_feature_size+att1_feature_size)/2), 1)
        self.linear2_2 = torch.nn.Linear(int((att1_feature_size+att2_feature_size)/2), 1)
        self.linear2_3 = torch.nn.Linear(int((att1_feature_size+att3_feature_size)/2), 1)

    def forward(self, feature1,feature2,feature3,feature1_seq):
        output_list_1=list()
        output_list_2=list()
        output_list_3=list()
        length=feature1_seq.size(0)
        for i in range(length):
            output1=torch.tanh(self.linear1_1(torch.cat([feature1_seq[i],feature1],dim=1)))
            output2=torch.tanh(self.linear1_2(torch.cat([feature1_seq[i],feature2],dim=1)))
            output3=torch.tanh(self.linear1_3(torch.cat([feature1_seq[i],feature3],dim=1)))
            output_list_1.append(self.linear2_1(output1))
            output_list_2.append(self.linear2_2(output2))
            output_list_3.append(self.linear2_3(output3))
        weight_1=torch.nn.functional.softmax(torch.torch.stack(output_list_1),dim=0)
        weight_2=torch.nn.functional.softmax(torch.torch.stack(output_list_2),dim=0)
        weight_3=torch.nn.functional.softmax(torch.torch.stack(output_list_3),dim=0)
        output=torch.mean((weight_1+weight_2+weight_3)*feature1_seq/3,0)
        return output

class ModalityFusion(torch.nn.Module):
    def __init__(self):
        super(ModalityFusion, self).__init__()
        image_feature_size=1024#image_feature.size(1)
        text_feature_size=512#text_feature.size(1)
        attribute_feature_size=200#attribute_feature.size(1)
        self.image_attention=RepresentationFusion(image_feature_size,text_feature_size,attribute_feature_size)
        self.text_attention=RepresentationFusion(text_feature_size,image_feature_size,attribute_feature_size)
        self.attribute_attention=RepresentationFusion(attribute_feature_size,image_feature_size,text_feature_size)
        self.image_linear_1=torch.nn.Linear(image_feature_size,512)
        self.text_linear_1=torch.nn.Linear(text_feature_size,512)
        self.attribute_linear_1=torch.nn.Linear(attribute_feature_size,512)
        self.image_linear_2=torch.nn.Linear(512,1)
        self.text_linear_2=torch.nn.Linear(512,1)
        self.attribute_linear_2=torch.nn.Linear(512,1)
        self.image_linear_3=torch.nn.Linear(image_feature_size,512)
        self.text_linear_3=torch.nn.Linear(text_feature_size,512)
        self.attribute_linear_3=torch.nn.Linear(attribute_feature_size,512)
    def forward(self, image_feature,image_seq,text_feature,text_seq,attribute_feature,attribute_seq):
                                             # [2, 1024]     [2, 512]      [2, 200]         [196, 2, 1024]
        image_vector    =self.image_attention(image_feature,text_feature,attribute_feature,image_seq)
                                             # [2, 512]     [2, 1024]      [2, 200]       [75, 2, 512]
        text_vector     =self.text_attention(text_feature,image_feature,attribute_feature,text_seq)
                                                     #[2, 200]      [2, 1024]     [2, 512]       [5, 2, 200]
        attribute_vector=self.attribute_attention(attribute_feature,image_feature,text_feature,attribute_seq)
        image_hidden=torch.tanh(self.image_linear_1(image_vector))
        text_hidden=torch.tanh(self.text_linear_1(text_vector))
        attribute_hidden=torch.tanh(self.attribute_linear_1(attribute_vector))
        image_score=self.image_linear_2(image_hidden)
        text_score=self.text_linear_2(text_hidden)
        attribute_score=self.attribute_linear_2(attribute_hidden)
        score=torch.nn.functional.softmax(torch.stack([image_score,text_score,attribute_score]),dim=0)
        image_vector=torch.tanh(self.image_linear_3(image_vector))
        text_vector=torch.tanh(self.text_linear_3(text_vector))
        attribute_vector=torch.tanh(self.attribute_linear_3(attribute_vector))
        # final fuse
        output=score[0]*image_vector+score[1]*text_vector+score[2]*attribute_vector
        return output
if __name__ == "__main__":
    image=ImageFeature.ExtractImageFeature()
    text=TextFeature.ExtractTextFeature(LoadData.TEXT_LENGTH, LoadData.TEXT_HIDDEN)
    attribute=AttributeFeature.ExtractAttributeFeature()
    fuse=ModalityFusion()
    for text_index,image_feature,attribute_index,group,id in LoadData.train_loader:
        image_result,image_seq=image(image_feature)
        text_result,text_seq=text(text_index)
        attribute_result,attribute_seq=attribute(attribute_index)
        result=fuse(image_result,image_seq,text_result,text_seq.permute(1,0,2),attribute_result,attribute_seq.permute(1,0,2))
        print(result.shape)

        break