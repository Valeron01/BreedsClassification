from argparse import ArgumentParser

import cv2

import augmentations
from datasets.dog_breed_identification_dataset import DogBreedIdentificationDataset


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parsed_args = parser.parse_args()

    dataset = DogBreedIdentificationDataset(parsed_args.dataset_path, image_transform=augmentations.get_augmentations())

    for image, class_index in dataset:
        print(dataset.all_classes[class_index])
        cv2.imshow("Image", image.permute(1, 2, 0).numpy())
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
