# 导入必要的库
import torch
import torchvision
import numpy as np
import random
import os
from PIL import Image
import sys

# 定义一些常量
IMG_SIZE = 256
CNV_VALUE = 1
LAYER_LABEL_VALUE = 4

# 定义一些文件夹的路径
DIR_CNV_IMAGE = r"/"
DIR_CNV_LABEL = r"/"
DIR_LAYER_IMAGE = r"/"
DIR_LAYER_LABEL = r"/"
DIR_CL_IMAGE = r"/"
DIR_CL_LABEL = r"/"


def FLS(dir_cnv_image, dir_cnv_label, dir_layer_image, dir_layer_label, dir_cl_image, dir_cl_label):
    file_names_cnv_image = os.listdir(dir_cnv_image)
    file_names_cnv_label = os.listdir(dir_cnv_label)
    file_names_layer_image = os.listdir(dir_layer_image)
    file_names_layer_label = os.listdir(dir_layer_label)

    for i in range(len(file_names_cnv_image)):
        file_names_cnv_image = file_names_cnv_image[i]
        file_names_cnv_label = file_names_cnv_label[i]
        file_names_layer_image = file_names_layer_image[i]
        file_names_layer_label = file_names_layer_label[i]

        cnv_image = load_image(os.path.join(dir_cnv_image, file_names_cnv_image))
        cnv_label = load_image(os.path.join(dir_cnv_label, file_names_cnv_label))
        layer_image = load_image(os.path.join(dir_layer_image, file_names_layer_image))
        layer_label = load_image(os.path.join(dir_layer_label, file_names_layer_label))

        cnv_image, cnv_offest = extract_cnv(cnv_image, cnv_label)

        cnv_layer_image, layer_image_star_coord, star_coord, layer_image_coords = generate_cl_image(cnv_image,
                                                                                                    layer_image,
                                                                                                    layer_label,
                                                                                                    cnv_offest)

        cnv_layer_label = generate_cl_label(layer_label, cnv_offest, layer_image_star_coord, star_coord,
                                            layer_image_coords)

        cnv_layer_image = Image.fromarray(cnv_layer_image)
        cnv_layer_label = Image.fromarray(cnv_layer_label)

        cnv_layer_image.save(os.path.join(dir_cl_image, file_names_layer_image))
        cnv_layer_label.save(os.path.join(dir_cl_label, file_names_layer_label))


def load_image(file_path):
    image = Image.open(file_path)

    image = image.resize((IMG_SIZE, IMG_SIZE))

    image = image.convert('L')

    image = np.array(image).astype(np.uint8)

    return image


def extract_cnv(cnv_image, cnv_label):
    cnv_image[cnv_label == 0] = 0

    cnv_offest = np.count_nonzero(cnv_label, axis=0).tolist()

    cnv_offest = [x for x in cnv_offest if x != 0]

    scale_factor = torch.rand(1)
    scale_factor = scale_factor * 1.5 + 0.5

    scale_factor = torch.add(scale_factor, 1e-6)

    cnv_offest = torch.tensor(cnv_offest) * scale_factor
    cnv_offest = cnv_offest.int()
    return cnv_image, cnv_offest


def generate_cl_image(cnv_image, layer_image, layer_label, cnv_offest):
    layer_image_coords = np.argwhere(layer_label == LAYER_LABEL_VALUE)

    layer_image_coords = layer_image_coords[np.lexsort((-layer_image_coords[:, 0], layer_image_coords[:, 1]))]

    layer_image_coords = layer_image_coords[(layer_image_coords[:, 0] + len(cnv_offest) < IMG_SIZE) & (
            layer_image_coords[:, 1] + len(cnv_offest) < IMG_SIZE)]

    unique_cols, unique_indices = np.unique(layer_image_coords[:, 1], return_index=True)

    layer_image_coords = layer_image_coords[unique_indices]

    star_coord = np.random.choice(len(layer_image_coords))
    layer_image_star_coord = layer_image_coords[star_coord]

    while layer_image_star_coord[0] <= torch.max(cnv_offest):
        if torch.max(torch.tensor(layer_image_coords[0])) <= torch.max(cnv_offest):
            scale_factor = torch.rand(1)
            scale_factor = scale_factor + 0.5

            scale_factor = torch.add(scale_factor, 1e-6)

            cnv_offest = cnv_offest * scale_factor
            cnv_offest = cnv_offest.int()
        star_coord = np.random.choice(len(layer_image_coords))
        layer_image_star_coord = layer_image_coords[star_coord]

    layer_image = lift_layer(layer_image, cnv_offest, layer_image_star_coord, star_coord, layer_image_coords)

    cnv_layer_image = fill_layer(cnv_image, layer_image, cnv_offest, layer_image_star_coord)

    return cnv_layer_image, layer_image_star_coord, star_coord, layer_image_coords


def lift_layer(layer_image, cnv_offest, layer_image_star_coord, star_coord, layer_image_coords):
    for i in range(len(cnv_offest)):
        layer_image[:layer_image_star_coord[0], layer_image_star_coord[1] + i] = np.roll(
            layer_image[:layer_image_star_coord[0], layer_image_star_coord[1] + i],
            -cnv_offest[i])

    return layer_image


def fill_layer(cnv_image, layer_image, cnv_offest, layer_image_star_coord):
    cnv_image = torch.from_numpy(cnv_image)

    cnv_start_col = torch.nonzero(torch.sum(cnv_image, axis=0))[0]

    for i in range(len(cnv_offest)):

        layer_col = layer_image_star_coord[1] + i
        cnv_col = cnv_start_col + i

        row_indices = torch.nonzero(cnv_image[:, cnv_col] != 0).squeeze()

        values = cnv_image[row_indices, cnv_col]

        values = values.flatten()
        try:
            layer_image[layer_image_star_coord[0] - cnv_offest[i]:layer_image_star_coord[0],
            layer_col] = values[:cnv_offest[i]]
        except Exception as e:

            print(layer_image_star_coord[0] - cnv_offest[i])
            sys.exit()

    return layer_image


def generate_cl_label(layer_label, cnv_offest, layer_image_star_coord, star_coord, layer_image_coords):
    cnv_layer_label = lift_layer(layer_label, cnv_offest, layer_image_star_coord, star_coord, layer_image_coords)

    start_col = layer_image_star_coord[1]
    end_col = start_col + len(cnv_offest) + 1

    cnv_region = cnv_layer_label[:, start_col:end_col]

    cnv_region[cnv_region == LAYER_LABEL_VALUE] = CNV_VALUE

    cnv_layer_label[:, start_col:end_col] = cnv_region

    return cnv_layer_label
