from xml.etree import ElementTree
from tqdm import tqdm
import os
import cv2
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r"D:\Program Files\Tesseract-OCR\tesseract.exe"

# Data directory
xml_dir = './datasets/annotations'
images_dir = './datasets/images'

# YOLO struct directory
images_data_dir = './datasets/data'
images_labeled_dir = './datasets/data/img'

if not os.path.isdir(images_data_dir):
    os.mkdir(images_data_dir)
if not os.path.isdir(images_labeled_dir):
    os.mkdir(images_labeled_dir)

xml_files = [os.path.join(xml_dir, xml_file).replace("\\", "/") for xml_file in os.listdir(xml_dir)]

image_num = 0

images_train = []

with tqdm(bar_format='{l_bar}{bar}{n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
    for xml_file in xml_files:
        tree = ElementTree.parse(xml_file)
        root = tree.getroot()

        title = root.get('title')
        pages = root.find('pages')

        images = os.path.join(images_dir, title).replace("\\", "/")

        pbar.total = len(pages)
        pbar.set_description("Labeling {0}".format(title), refresh=False)
        pbar.reset()

        for page in pages:
            index_img = page.get('index').zfill(3)

            width = page.get('width')
            height = page.get('height')

            pbar.update()

            texts = page.findall('text')
            if not texts:
                continue

            clean_image = os.path.join(images_dir, title, index_img + '.jpg').replace("\\", "/")
            cv_image = cv2.imread(clean_image)

            labeled_image = os.path.join(images_labeled_dir, str(image_num) + '.jpg').replace("\\", "/")
            images_train.append(os.path.relpath(labeled_image, './datasets') + '\n')

            bounding_boxes = open(os.path.join(images_labeled_dir, str(image_num) + '.txt').replace("\\", "/"), 'w')

            for each_text in texts:
                # Draw rectangle around the text
                # +/- 10 because some letters are output of the rectangle
                x_min = int(each_text.get('xmin')) - 10
                x_max = int(each_text.get('xmax')) + 10
                y_min = int(each_text.get('ymin')) - 10
                y_max = int(each_text.get('ymax')) + 10

                # Convert to yolo format
                dw = 1. / float(width)
                dh = 1. / float(height)
                x = (x_min + x_max) / 2.0
                y = (y_min + y_max) / 2.0
                w = x_max - x_min
                h = y_max - y_min
                x = x * dw
                w = w * dw
                y = y * dh
                h = h * dh

                text = each_text.text
                if text is not None:
                    bounding_boxes.write("0 {0} {1} {2} {3}\n".format(x, y, w, h))

                    # TEST PYTESSERACT
                    # RESULT : FAILED
                    # image_roi = cv_image[y_min:y_max, x_min:x_max]
                    # configuration = "-l jpn_vert --oem 1 --psm 5"
                    # pytext = pytesseract.image_to_string(image_roi, config=configuration).replace("\n", "").replace(" ", "")
                    # if text == pytext:
                    #     print("It's equal {0} and {1}".format(text, pytext))
                    # else:
                    #     print("It's NOT equal {0} and {1}".format(text, pytext))
                    # assert text == pytext, "Difference between {0} and {1}".format(text, pytext)

                    # For debugging
                    # cv2.rectangle(cv_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            image_num = image_num + 1
            bounding_boxes.close()
            cv2.imwrite(labeled_image, cv_image)

# Store path images in a file (yolo convention)
with open(os.path.join(images_data_dir, 'train.txt').replace("\\", "/"), 'w') as train_file:
    train_file.writelines(images_train)
    train_file.close()
