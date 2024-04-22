import os
import glob
import shutil
import random
import logging
import argparse
import numpy as np

from pathlib import Path
from tqdm import trange, tqdm
from datetime import timedelta
from PIL import Image, ImageDraw, ImageFont

import dv_processing as dv
import dv_toolkit as kit

import xml.etree.ElementTree as ET 


def process_data(args, file):
    # define folder creator
    def create_folder(path):
        if path.exists(): 
            shutil.rmtree(path)
        path.mkdir(parents=True)

        return path

    # initialize folder
    clip_path  = create_folder(file / 'clips')
    image_path = create_folder(file / 'images')

    # load offline data
    reader = kit.io.MonoCameraReader(f"{file}/denoised_record.aedat4")
    data, resolution = reader.loadData(), reader.getResolution("events")

    # register a slicer
    if args.clip:
        slicer = kit.MonoCameraSlicer()

    # register accumulator
    if args.image:
        accumulator = dv.Accumulator(resolution)
        accumulator.setMinPotential(-np.inf)
        accumulator.setMaxPotential(+np.inf)
        accumulator.setEventContribution(1.0)
        accumulator.setIgnorePolarity(False)
        accumulator.setDecayFunction(dv.Accumulator.Decay.NONE)

    # initialize
    clip_index = 0
    if not data.frames().isEmpty():
        latest_frame = data.frames().at(0).image
    else:
        latest_frame = np.zeros(resolution[::-1]).astype(np.uint8)

    def subprocess(data):
        # define nonlocal variable
        nonlocal clip_index, latest_frame

        # set output file name
        output_name = f"{clip_index:05d}"
        
        # update for next subprocess
        clip_index = clip_index + 1
        latest_frame = data.frames().at(0).image if not data.frames().isEmpty() else latest_frame
        
        if args.clip:
            writer = kit.io.MonoCameraWriter(f"{clip_path}/{output_name}.aedat4", resolution)
            writer.writeData(data)

        if args.image:
            # generate count image
            accumulator.clear()
            accumulator.accept(data.events().toEventStore())
            count = accumulator.getPotentialSurface()

            # # overlap to image
            # image = cv2.cvtColor(latest_frame.copy(), cv2.COLOR_GRAY2RGB)
            # image[count != 0] = 0
            # image[count > 0, 2] = 255
            # image[count < 0, 1] = 255

            # cv2.imwrite(f"{image_path}/{output_name}.png", image)

    # do every 33ms (cannot modify!)
    slicer = kit.MonoCameraSlicer()
    slicer.doEveryTimeInterval("events", timedelta(milliseconds=33), subprocess)
    slicer.accept(data)


def parse_to_new_xml(elements, partition, path):
    new_elem = ET.Element(partition)

    # create labels
    label_names  = ['fire', 'person', 'other']
    label_colors = ['#33ddff', '#ff6037', '#b83df5']
    labels_elem  = ET.SubElement(new_elem, 'labels')
    for name, color in zip(label_names, label_colors):
        label_elem = ET.SubElement(labels_elem, 'label', attrib={'name': name, 'color': color})

    # create annotations
    annotations_elem = ET.SubElement(new_elem, 'annotations')
    for id, image_elem in enumerate(elements):
        image_elem.set('id', f"{id}")
        annotations_elem.append(image_elem)

    # save to file
    path.mkdir(parents=True, exist_ok=True)
    shuffled_tree = ET.ElementTree(new_elem)
    shuffled_tree.write(path / f'{partition}.xml')


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description="paramters")
    parser.add_argument('--clip', action="store_false")
    parser.add_argument('--image', action="store_true")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()

    # initialize file lists and output directories
    cwd = os.path.dirname(__file__)
    input_path = Path(os.path.dirname(__file__))

    # register a logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s [%(levelname)s]: %(message)s',
                        handlers=[logging.FileHandler('output.log'), logging.StreamHandler()])
    logging.info(f"Now loading {input_path}")
    logging.info(f"{args}")

    # recursively obtain file list
    element_list = []
    for file in tqdm(sorted(input_path.glob("./aedats/*"))):
        # clip data and generate images
        process_data(args, file)

        # parse annotations
        tree = ET.parse(file / 'annotations.xml')
        for image in tree.getroot().findall('.//image'):
            
            # reset name
            original_name = image.get('name').split('/')
            modified_name = f"{original_name[0]}/clips/{original_name[1][9:14]}.aedat4"
            image.set('name', modified_name)

            # update element
            element_list.append(image)

    # random shuffle and split
    random.seed(args.seed)
    random.shuffle(element_list)
    ratio_index = int(0.8 * len(element_list))
    train_set   = sorted(element_list[:ratio_index], key=lambda x:x.get('name'))
    test_set    = sorted(element_list[ratio_index:], key=lambda x:x.get('name'))
    
    # write to new files
    # TODO 文件输出路径不对
    parse_to_new_xml(train_set, 'train', input_path / 'annotations')
    parse_to_new_xml(test_set,  'test',  input_path / 'annotations')
