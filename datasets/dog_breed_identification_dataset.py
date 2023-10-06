import os
import typing

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DogBreedIdentificationDataset(Dataset):
    all_classes = [
        "affenpinscher",
        "afghan_hound",
        "african_hunting_dog",
        "airedale",
        "american_staffordshire_terrier",
        "appenzeller",
        "australian_terrier",
        "basenji",
        "basset",
        "beagle",
        "bedlington_terrier",
        "bernese_mountain_dog",
        "black-and-tan_coonhound",
        "blenheim_spaniel",
        "bloodhound",
        "bluetick",
        "border_collie",
        "border_terrier",
        "borzoi",
        "boston_bull",
        "bouvier_des_flandres",
        "boxer",
        "brabancon_griffon",
        "briard",
        "brittany_spaniel",
        "bull_mastiff",
        "cairn",
        "cardigan",
        "chesapeake_bay_retriever",
        "chihuahua",
        "chow",
        "clumber",
        "cocker_spaniel",
        "collie",
        "curly-coated_retriever",
        "dandie_dinmont",
        "dhole",
        "dingo",
        "doberman",
        "english_foxhound",
        "english_setter",
        "english_springer",
        "entlebucher",
        "eskimo_dog",
        "flat-coated_retriever",
        "french_bulldog",
        "german_shepherd",
        "german_short-haired_pointer",
        "giant_schnauzer",
        "golden_retriever",
        "gordon_setter",
        "great_dane",
        "great_pyrenees",
        "greater_swiss_mountain_dog",
        "groenendael",
        "ibizan_hound",
        "irish_setter",
        "irish_terrier",
        "irish_water_spaniel",
        "irish_wolfhound",
        "italian_greyhound",
        "japanese_spaniel",
        "keeshond",
        "kelpie",
        "kerry_blue_terrier",
        "komondor",
        "kuvasz",
        "labrador_retriever",
        "lakeland_terrier",
        "leonberg",
        "lhasa",
        "malamute",
        "malinois",
        "maltese_dog",
        "mexican_hairless",
        "miniature_pinscher",
        "miniature_poodle",
        "miniature_schnauzer",
        "newfoundland",
        "norfolk_terrier",
        "norwegian_elkhound",
        "norwich_terrier",
        "old_english_sheepdog",
        "otterhound",
        "papillon",
        "pekinese",
        "pembroke",
        "pomeranian",
        "pug",
        "redbone",
        "rhodesian_ridgeback",
        "rottweiler",
        "saint_bernard",
        "saluki",
        "samoyed",
        "schipperke",
        "scotch_terrier",
        "scottish_deerhound",
        "sealyham_terrier",
        "shetland_sheepdog",
        "shih-tzu",
        "siberian_husky",
        "silky_terrier",
        "soft-coated_wheaten_terrier",
        "staffordshire_bullterrier",
        "standard_poodle",
        "standard_schnauzer",
        "sussex_spaniel",
        "tibetan_mastiff",
        "tibetan_terrier",
        "toy_poodle",
        "toy_terrier",
        "vizsla",
        "walker_hound",
        "weimaraner",
        "welsh_springer_spaniel",
        "west_highland_white_terrier",
        "whippet",
        "wire-haired_fox_terrier",
        "yorkshire_terrier"
    ]

    def __init__(self, dataset_folder_path, image_transform=None):
        self.image_transform = image_transform
        self.dataset_folder_path = dataset_folder_path

        self.loaded_csv = pd.read_csv(os.path.join(self.dataset_folder_path, "labels.csv"))
        self.images_folder_path = os.path.join(self.dataset_folder_path, "train")

    def __len__(self) -> int:
        return len(self.loaded_csv)

    def __getitem__(self, item: int) -> typing.Tuple[typing.Union[np.ndarray, torch.Tensor], int]:
        """
        :param item: index of a pair of data
        :return: Tuple of loaded image and index of a class, to which the image belongs
        """
        image_name, class_name = self.loaded_csv.iloc[item]
        loaded_image = cv2.imread(
            os.path.join(self.images_folder_path, image_name + ".jpg")
        )

        if self.image_transform is not None:
            loaded_image = self.image_transform(image=loaded_image)["image"]

        class_index = DogBreedIdentificationDataset.all_classes.index(class_name)

        return loaded_image, class_index
