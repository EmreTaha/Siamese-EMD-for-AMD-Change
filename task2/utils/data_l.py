from torch.utils import data
from torchvision.io import read_image
import numpy as np
from monai.data import CacheDataset, DataLoader
from monai.data.utils import worker_init_fn

class Dataset_memm(data.Dataset):
    """
    A custom dataset class for handling MEMM dataset.

    Args:
        df (pandas.DataFrame): The input dataframe containing the dataset.
        dtype (numpy.dtype, optional): The data type of the images. Defaults to np.uint8.
    """
    def __init__(self, df, dtype=np.uint8, grayscale=False, image_column='image', challange_eval=False):
        self.df = df
        self.dtype = dtype
        self.grayscale = grayscale
        self.image_column = image_column
        self.challange_eval = challange_eval
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """
        Retrieves a single item from the dataset.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the image, label, path, patient ID, visit ID, eye ID, and case.
        """
        eyes_paths = self.df.iloc[index][self.image_column]
        pat_id = self.df.iloc[index]['id_patient']
        visit_id = self.df.iloc[index]['num_current_visit']
        eye_id = self.df.iloc[index]['side_eye']
        case = self.df.iloc[index]['case']
        if not self.challange_eval: conv_label = self.df.iloc[index]['label']
        else: conv_label = -1
        
        eyes = read_image(eyes_paths)
        if eyes.shape[0] == 3 and self.grayscale:
            eyes = eyes[:1]

        return_dict = {"image": eyes, "label": conv_label, "path": eyes_paths,'patID': pat_id, 
                       'visitID': visit_id, 'eyeID': eye_id, 'case': case}

        return return_dict
    

def create_cacheds_dl(dataset,transforms,cache_rate=None, batch_size=32, shuffle=True, num_workers=0, worker_fn=worker_init_fn, drop_last=False, progress=True, **kwargs):
    '''
    Create a Monai CacheDataset and a Monai DataLoader.

    Args:
        dataset (Dataset): The input dataset.
        transforms (Callable): The transforms to apply to the dataset.
        cache_rate (float, optional): The cache rate for the CacheDataset. Defaults to None.
        batch_size (int, optional): The batch size for the DataLoader. Defaults to 32.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        num_workers (int, optional): The number of worker processes for data loading. Defaults to 0.
        worker_fn (Callable, optional): The function to initialize each worker process. Defaults to worker_init_fn.
        drop_last (bool, optional): Whether to drop the last incomplete batch. Defaults to False.
        progress (bool, optional): Whether to show the progress bar. Defaults to True.
        **kwargs: Additional keyword arguments to be passed to the DataLoader.

    Returns:
        DataLoader: The created DataLoader.
    '''
    if cache_rate is None:
        cache_rate=0.0
    data_s = CacheDataset(dataset, transforms, cache_rate=cache_rate, progress=progress)
    data_l = DataLoader(data_s, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, worker_init_fn=worker_fn, drop_last=drop_last, **kwargs)
    return data_l