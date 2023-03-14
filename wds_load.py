import webdataset as wds
import numpy as np
import cv2
from torch.utils.data import DataLoader
    
def get_pixels(img, proportion):
    mask = np.random.choice([0, 1], size=img.shape[:2], p=[1-proportion, proportion])

    return img * np.repeat(mask, 3, axis=1).reshape(img.shape)

def make_parser(proportion):

    def get_item(item):
        np_array = np.frombuffer(item['jpg'], np.uint8)
        cv2img = cv2.cvtColor(cv2.imdecode(np_array, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)  
        cv2img = (cv2img.astype(np.float32) / 127.5) - 1.0

        pixel_hint = get_pixels(cv2img, proportion)

        return dict(txt=item['txt'].decode("utf-8"), jpg=cv2img, pixel_hint=pixel_hint)

    return get_item

def load_laion(batch_size, train_url, test_url, proportion):
    parser = make_parser(proportion)
    train = wds.WebDataset(train_url).map(parser)
    test = wds.WebDataset(test_url).map(parser)

    train_dl = DataLoader(train, batch_size=batch_size, num_workers=0)
    test_dl = DataLoader(test, batch_size=batch_size, num_workers=0)

    return train_dl, test_dl

# load_laion100k(4, 'training/laion-100k/part-00000-5114fd87-297e-42b0-9d11-50f1df323dfa-c000.snappy.parquet', n_samples=100_000, train_size=99_000)
load_laion(
    4, 
    'training/laion-20-data/{00000..00008}.tar', 
    'training/laion-20-data/00009.tar', 
    proportion=0.2
)
