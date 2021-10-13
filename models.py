# The class containing the model
from multiprocessing import Array
import torch
from PIL import Image
import torchvision
from torchvision import transforms
import io
from math import floor
from typing import List
import traceback,sys,os
from flask import jsonify

class MobileNet:
    def __init__(self):
        # Source: https://github.com/Lasagne/Recipes/blob/master/examples/resnet50/imagenet_classes.txt
        with open('imagenet_classes.txt') as f:
            self.classes = [line.strip() for line in f.readlines()]

        # self.model = torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=True)
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        
        self.model = torchvision.models.resnet18(pretrained=True).to(self.device)
        self.model.eval()

    def transform_image(self, image_path):
        my_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                [0.485, 0.456, 0.406],
                                                [0.229, 0.224, 0.225])])
        image = Image.open(image_path)
        return my_transforms(image).unsqueeze(0)   
    
    def infer(self, image_path):   
        try:     
            image_tensors = [self.transform_image(image_path=image) for image in image_path]
            
            # move the input and model to GPU for speed if available        
            # self.model.to(self.device)
            tensor = torch.cat(image_tensors).to(self.device)

            with torch.no_grad():
                outputs = self.model(tensor)   

            outputs = [torch.nn.functional.softmax(output, dim=0) for output in outputs]
            results =[torch.max(output, 0) for output in outputs]

            print(len(results))

            return [((self.classes[int(result[1].item())]),str(floor(result[0].item() * 10000) / 100)) for result in results]
                        
        except :
            print(jsonify({'error': traceback.print_exception(*sys.exc_info())}))
    
    def get_prediction(self,image_path):
        tensor = self.transform_image(image_path=image_path)
        # self.model.to(self.device)
        outputs = self.model.forward(tensor)
        _, y_hat = outputs.max(1)
        predicted_idx = y_hat.item()
        return self.classes[predicted_idx]

if __name__ == "__main__":
    with open(r"cat.jpg", 'rb') as f:
        image_bytes = f.read()
    model = MobileNet()
    result = model.get_prediction("cat.jpg")
    print(result)
    batch_result = model.infer("cat.jpg",64)
    assert batch_result == [result] * 64        



