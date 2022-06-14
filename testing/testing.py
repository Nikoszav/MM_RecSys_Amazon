from modules.language_encoder import get_language_encoder
from modules.language_encoder import BertTokenizer
from modules.poster_encoder import get_feature_extractor
from tqdm import tqdm
import torch

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from datasets import Movies

def testingBert(training_dataset): 
    # input = ["The dark knight rises This movie is about Batman. It takes place in Gotham.", "The dark knight rises This movie is about Batman. It takes place in Gotham."]
    # bert = BertTokenizer()
    # tokienizedInput, maskInput = BertTokenizer.tokenizeText(bert, input = input, maxLength = len(input[0]))
    # languageEncoder = get_language_encoder()
    # q_encoded = languageEncoder(tokienizedInput, maskInput)
    
    # print(tokienizedInput.shape, maskInput.shape)
    # # print(len(input.split()))

    # img = Image.open("../recommender_system_amazon/poster_downloads/99994.jpg")
   
   

    pbar = tqdm(train_dataset)
    b = 0
    for batch in pbar:
        b+=1
        # batch = [t.squeeze() for t in batch]
        # print(len(batch))
        image_tensor, movie_text_tokens, movie_text_mask = batch
        movie_text_tokens = movie_text_tokens.squeeze()
        movie_text_mask = movie_text_mask.squeeze()
        image_extractor = get_feature_extractor()
        image_new = image_extractor(image_tensor) # [batch_size, 512] -> Output of resnet
		
        languageEncoder = get_language_encoder()
        print(movie_text_tokens.shape, movie_text_mask.shape)
        movie_descr = languageEncoder(movie_text_tokens, movie_text_mask) # [batch_size, config.question_length, 768 (bert)]
        movie_descr = ""
        print(b, "         ", image_new.shape, "\n", movie_descr)
  
  
if __name__ == "__main__": 
    print("running")
    train_dataset = DataLoader(Movies(), batch_size=16, num_workers=0)
    # print(len(train_dataset))
    testingBert(train_dataset)