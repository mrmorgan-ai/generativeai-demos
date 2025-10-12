from transformers import ViltProcessor, ViltForQuestionAnswering
# import requests
from PIL import Image

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw) #type: ignore
# text = "How many cats are there?"

# 470MB
processor = ViltProcessor.from_pretrained('dandelin/vilt-b32-finetuned-vqa')
model = ViltForQuestionAnswering.from_pretrained('dandelin/vilt-b32-finetuned-vqa')


def model_pipeline(text: str, image: Image): #type: ignore
    encoding = processor(image, text, return_tensors="pt")
    
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    
    return model.config.id2label[idx] #type: ignore

    