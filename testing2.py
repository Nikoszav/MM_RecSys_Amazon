from modules.language_encoder import get_language_encoder
from modules.language_encoder import BertTokenizer
from modules.poster_encoder import get_feature_extractor
from tqdm import tqdm
import torch

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from datasets_concat import Movies
# ratings = pd.read_pickle(config.ratings_file)
# data =  pd.read_pickle(config.data_file)

# user_count = ratings["userId"].nunique()
# movie_count = data["movieId"].nunique()


# # Let's peek into the loaded DataFrames
# print('Sneak peek into the ratings DataFrame:\n\n', ratings.head(), '\n')
# print(f'Number of users: {user_count:,}')
# print(f'Number of ratings: {ratings.shape[0]:,}')
# print(f'Number of rated movies: {ratings["movieId"].nunique():,}\n')
# print(f'Sneak peek into the movies DataFrame:\n\n', data.head(), '\n')
# print(f'Number of movies: {movie_count:,} - notice this is higher than the number of rated movies!')

# print("##########################################################################################")

# union = pd.merge(ratings, data, on='movieId', how='inner')
# # print (union)
# # Define our input and labels data X,Y
# X = union[['userId','movieId', 'overview', "path" ]]
# Y = union['rating'].astype(np.float32)


def testingBert(training_dataset): 

    pbar = tqdm(training_dataset)
    b = 0
    for batch in pbar:
        b+=1
        # batch = [t.squeeze() for t in batch]
        # print(len(batch))
        
        # movie_text_tokens = movie_text_tokens.squeeze()
        # movie_text_mask = movie_text_mask.squeeze()
        # image_extractor = get_feature_extractor()
        # image_new = image_extractor(image_tensor) # [batch_size, 512] -> Output of resnet
		
        # languageEncoder = get_language_encoder()
        # print(movie_text_tokens.shape, movie_text_mask.shape)
        # movie_descr = languageEncoder(movie_text_tokens, movie_text_mask) # [batch_size, config.question_length, 768 (bert)]
        # movie_descr = ""
        # print(b, "         ", image_new.shape, "\n", movie_descr)
        # print(len(batch))
        # break
        user_id, movie_id, movie_image_encoder, movie_rep = batch
        movie_rep = torch.squeeze(movie_rep)
        print(user_id, movie_id, movie_image_encoder.shape, movie_rep.shape)
        
        # resize = transforms.Resize(250)
        # norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  #not sure why this normalization
        # to_tensor = transforms.ToTensor()
        # poster_name = "../recommender_system_amazon/poster_downloads/2461.jpg"
        # image = Image.open(poster_name).convert('RGB')  #read and convert images to RGB
        
        # image_tensor = norm(to_tensor(resize(image))) 
        # print(image_tensor.shape)
        image_extractor = get_feature_extractor()
        image_new = image_extractor(movie_image_encoder)
        print(image_new.shape)
        break
       

if __name__ == "__main__": 
    print("running")
    train_dataset = DataLoader(Movies(), batch_size=64, num_workers=0)
    # print(len(train_dataset))
    testingBert(train_dataset)