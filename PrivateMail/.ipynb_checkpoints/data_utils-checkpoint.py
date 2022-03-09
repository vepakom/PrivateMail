{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "processing car dataset\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "process cropped data for car dataset: 100%|██████████| 16185/16185 [05:24<00:00, 49.94it/s] \n"
     ]
    }
   ],
   "source": [
    "import argparse\n",
    "import os\n",
    "\n",
    "import torch\n",
    "from PIL import Image\n",
    "from scipy.io import loadmat\n",
    "from tqdm import tqdm\n",
    "\n",
    "\n",
    "def read_txt(path, data_num):\n",
    "    data = {}\n",
    "    for line in open(path, 'r', encoding='utf-8'):\n",
    "        if data_num == 2:\n",
    "            data_1, data_2 = line.split()\n",
    "        else:\n",
    "            data_1, data_2, data_3, data_4, data_5 = line.split()\n",
    "            data_2 = [data_2, data_3, data_4, data_5]\n",
    "        data[data_1] = data_2\n",
    "    return data\n",
    "\n",
    "\n",
    "def process_car_data(data_path, data_type):\n",
    "    if not os.path.exists('{}/{}'.format(data_path, data_type)):\n",
    "        os.mkdir('{}/{}'.format(data_path, data_type))\n",
    "    train_images, test_images = {}, {}\n",
    "    annotations = loadmat('{}/cars_annos.mat'.format(data_path))['annotations'][0]\n",
    "    for img in tqdm(annotations, desc='process {} data for car dataset'.format(data_type)):\n",
    "        img_name, img_label = str(img[0][0]), str(img[5][0][0])\n",
    "        if data_type == 'uncropped':\n",
    "            img = Image.open('{}/{}'.format(data_path, img_name)).convert('RGB')\n",
    "        else:\n",
    "            x1, y1, x2, y2 = int(img[1][0][0]), int(img[2][0][0]), int(img[3][0][0]), int(img[4][0][0])\n",
    "            img = Image.open('{}/{}'.format(data_path, img_name)).convert('RGB').crop((x1, y1, x2, y2))\n",
    "        save_name = '{}/{}/{}'.format(data_path, data_type, os.path.basename(img_name))\n",
    "        img.save(save_name)\n",
    "        if int(img_label) < 99:\n",
    "            if img_label in train_images:\n",
    "                train_images[img_label].append(save_name)\n",
    "            else:\n",
    "                train_images[img_label] = [save_name]\n",
    "        else:\n",
    "            if img_label in test_images:\n",
    "                test_images[img_label].append(save_name)\n",
    "            else:\n",
    "                test_images[img_label] = [save_name]\n",
    "    torch.save({'train': train_images, 'test': test_images}, '{}/{}_data_dicts.pth'.format(data_path, data_type))\n",
    "\n",
    "\n",
    "def process_cub_data(data_path, data_type):\n",
    "    if not os.path.exists('{}/{}'.format(data_path, data_type)):\n",
    "        os.mkdir('{}/{}'.format(data_path, data_type))\n",
    "    images = read_txt('{}/images.txt'.format(data_path), 2)\n",
    "    labels = read_txt('{}/image_class_labels.txt'.format(data_path), 2)\n",
    "    bounding_boxes = read_txt('{}/bounding_boxes.txt'.format(data_path), 5)\n",
    "    train_images, test_images = {}, {}\n",
    "    for img_id, img_name in tqdm(images.items(), desc='process {} data for cub dataset'.format(data_type)):\n",
    "        if data_type == 'uncropped':\n",
    "            img = Image.open('{}/images/{}'.format(data_path, img_name)).convert('RGB')\n",
    "        else:\n",
    "            x1, y1 = int(float(bounding_boxes[img_id][0])), int(float(bounding_boxes[img_id][1]))\n",
    "            x2, y2 = x1 + int(float(bounding_boxes[img_id][2])), y1 + int(float(bounding_boxes[img_id][3]))\n",
    "            img = Image.open('{}/images/{}'.format(data_path, img_name)).convert('RGB').crop((x1, y1, x2, y2))\n",
    "        save_name = '{}/{}/{}'.format(data_path, data_type, os.path.basename(img_name))\n",
    "        img.save(save_name)\n",
    "        if int(labels[img_id]) < 101:\n",
    "            if labels[img_id] in train_images:\n",
    "                train_images[labels[img_id]].append(save_name)\n",
    "            else:\n",
    "                train_images[labels[img_id]] = [save_name]\n",
    "        else:\n",
    "            if labels[img_id] in test_images:\n",
    "                test_images[labels[img_id]].append(save_name)\n",
    "            else:\n",
    "                test_images[labels[img_id]] = [save_name]\n",
    "    torch.save({'train': train_images, 'test': test_images}, '{}/{}_data_dicts.pth'.format(data_path, data_type))\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    parser = argparse.ArgumentParser(description='Process datasets')\n",
    "    parser.add_argument('--data_path', default='/data', type=str, help='datasets path')\n",
    "\n",
    "    opt = parser.parse_args()\n",
    "\n",
    "    print('processing car dataset')\n",
    "    process_car_data('{}/car'.format(opt.data_path), 'cropped')\n",
    "    print('processing cub dataset')\n",
    "    process_cub_data('{}/cub'.format(opt.data_path), 'cropped')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
