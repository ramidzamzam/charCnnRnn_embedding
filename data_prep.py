# Rami Derawi 21 Nov 2020
# Create json data out of .png images and text files
# The result is folder contains the json (see below)
# and original, 64x64, 256x256 image and encoded versions
# Images /*.png
# [
#     {
#         id: "",
#         img_path: "",
#         img_64x64_path: "",
#         img_256x256_path: "",
#         encod_64x64_path: "",
#         encod_256x256_path: "",
#         embedded_txt: "",
#         text: ""
#     }
# ]
# !/usr/bin/python

import argparse
import glob
import ntpath
import time
import shutil
import json
import os
import cv2
import torch
import torchvision.transforms as transforms
import sys
from pathlib import Path

# Import models
path = os.path.dirname(os.path.abspath(__file__))
path = Path(path)
src_path = path.parent
print("Add models path to sys paths - ", src_path)
sys.path.append(str(src_path) + '/models/conv_autoencoder/')
from models import ConvAutoencoder

sys.path.append(str(src_path) + '/models/char_embedding/')
from char_cnn_rnn import CharCnnRnn, prepare_text


def main():
    parser = argparse.ArgumentParser(description='Fix images name and create json texts.')
    parser.add_argument('input', metavar='input', type=str,
                        help='Source folder full path.')
    parser.add_argument('output', metavar='output', type=str,
                        help='Destination folder full path.')
    parser.add_argument('encoder_path', metavar='encoder_path', type=str,
                        help='Full path of the encoder trained model')
    parser.add_argument('embedding_path', metavar='embedding_path', type=str,
                        help='Full path of the text embedding trained model')

    args = parser.parse_args()
    input_folder = args.input
    output_folder = args.output
    encoder_path = args.encoder_path
    embedding_path = args.embedding_path

    # Avoid duplication
    timestamp = str(time_msec())
    output_folder = output_folder + '/prep_data_' + timestamp
    # Create output dir
    ensure_folder(output_folder)
    ensure_folder(output_folder + '/original_images')
    ensure_folder(output_folder + '/enc_256x256_images')
    ensure_folder(output_folder + '/enc_64x64_images')
    ensure_folder(output_folder + '/64x64_images')
    ensure_folder(output_folder + '/256x256_images')
    ensure_folder(output_folder + '/embedded_text')

    print(" main - source: ", input_folder, "destination: ", output_folder)
    # Load encoder and embedding models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Load encoder model - ", encoder_path)
    encoder = ConvAutoencoder()
    encoder.load_state_dict(torch.load(encoder_path))
    encoder = encoder.to(device)
    print("Encoder model created! ")
    print("Load embedding model - ", embedding_path)
    embedding = CharCnnRnn()
    embedding.load_state_dict(torch.load(embedding_path))
    embedding = embedding.to(device)
    print("Embedding model created! ")
    # Pull image and text paths
    json_data = []
    for img in glob.glob(input_folder + '/*.png'):
        # Get image text file name
        text = text_file_name(img)
        # Read and save text
        with open(input_folder + '/' + text, 'r') as file:
            data = file.read()
        # Rename image and copy to destination folder
        json_img = {}
        # and create json data file
        img_id = str(time_msec())
        print("Process file - ", img_id)
        img_des = output_folder + '/original_images/' + img_id + '.png'
        json_img['id'] = img_id
        json_img['img_path'] = '/original_images/' + img_id + '.png'
        json_img['text'] = data
        # Copy original image
        shutil.copy(img, img_des)
        # Create scaled copies
        img_64x64, img_256x256 = scale_img(img_des, output_folder)
        json_img['img_64x64_path'] = img_64x64
        json_img['img_256x256_path'] = img_256x256
        # Encode images
        name_enc_img_64, enc_img_64 = encode_img(output_folder + img_64x64, encoder, device,
                                                 output_folder + '/enc_64x64_images')
        name_enc_img_256, enc_img_256 = encode_img(output_folder + img_256x256, encoder, device,
                                                   output_folder + '/enc_256x256_images')
        json_img['encod_64x64_path'] = '/enc_64x64_images/' + name_enc_img_64
        json_img['encod_256x256_path'] = '/enc_256x256_images/' + name_enc_img_256
        # Embedding text
        name_embedding_text = embedding_text(json_img['text'], img_id,
                                             output_folder, embedding, device)
        json_img['embedded_txt'] = '/embedded_text/' + name_embedding_text
        # Append new data
        json_data.append(json_img)
    # Save json
    with open(output_folder + '/images_data.json', 'w') as json_file:
        json_file.write(json.dumps(json_data))


def text_file_name(img):
    # .../Screen Shot file_name.png
    name = ntpath.basename(img)
    # Remove Screen Shot and .png
    name = name.replace('Screen Shot ', '')
    name = name.replace('.png', '.txt')
    return name


def encode_img(img, encoder, device, output_folder):
    torch.no_grad()
    encoder.eval()
    transform = transforms.ToTensor()
    image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    image = transform(image)
    image = image.float()
    image = image.to(device)
    enc_img = encoder(image.unsqueeze(0), encoder_mode=True)
    name = ntpath.basename(img).split('.')[0]
    name = 'enc_' + name + '.pt'
    torch.save(enc_img, output_folder + '/' + name)
    return name, enc_img


def embedding_text(text, img_id, output_folder, embedding, device):
    torch.no_grad()
    embedding.eval()
    text = prepare_text(text)
    text.to(device)
    embedded_txt = embedding(text.unsqueeze(0))
    # Get image id from name (enc_64x64_id.pt)
    name = 'embedded_txt_' + img_id + '.pt'
    torch.save(embedded_txt, output_folder + '/embedded_text/' + name)
    return name


def scale_img(img, output_folder):
    # Create 256x256 and 64x64 gray copies
    img_name = ntpath.basename(img)
    image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    # resize image to 64X64 and save image
    dsize = (64, 64)
    sized_img = cv2.resize(image, dsize, interpolation=cv2.INTER_AREA)
    sized_img = cv2.cvtColor(sized_img, cv2.COLOR_BGR2GRAY)
    img_64x64 = '/64x64_images/64x64_' + img_name
    cv2.imwrite(output_folder + img_64x64, sized_img)
    # resize image to 256x256 and save image
    dsize = (256, 256)
    sized_img = cv2.resize(image, dsize, interpolation=cv2.INTER_AREA)
    sized_img = cv2.cvtColor(sized_img, cv2.COLOR_BGR2GRAY)
    img_256x256 = '/256x256_images/256x256_' + img_name
    cv2.imwrite(output_folder + img_256x256, sized_img)
    return img_64x64, img_256x256


def time_msec():
    return int(round(time.time() * 1000))


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


if __name__ == '__main__':
    print("<<<<<<<<<<<<<<< Stared >>>>>>>>>>>>>>>>>")
    main()
