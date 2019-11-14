from xml.etree import ElementTree as ET
import os
import cv2

xml_dir = './datasets/annotations'
images_dir = './datasets/images'
images_labeled_dir = './datasets/images-labeled'

os.mkdir(images_labeled_dir)

xml_files = [os.path.join(xml_dir, xml_file).replace("\\", "/") for xml_file in os.listdir(xml_dir)]

for xml_file in xml_files:
    tree = ET.parse(xml_file)
    root = tree.getroot()

    title = root.get('title')
    pages = root.find('pages')

    images = os.path.join(images_dir, title).replace("\\", "/")

    os.mkdir(os.path.join(images_labeled_dir, title).replace("\\", "/"))

    for page in pages:
        index = page.get('index').zfill(3)
        width = page.get('width')
        height = page.get('height')
        texts = page.findall('text')

        clean_image = os.path.join(images_dir, title, index + '.jpg').replace("\\", "/")
        labeled_image = os.path.join(images_labeled_dir, title, index + '.jpg').replace("\\", "/")

        cv_image = cv2.imread(clean_image)

        for each_text in texts:
            id = each_text.get('id') #Not sure if necessary

            # Draw rectangle around the text 
            # +/- 10 because some letters are output of the rectangle
            xmin = int(each_text.get('xmin')) - 10
            xmax = int(each_text.get('xmax')) + 10
            ymin = int(each_text.get('ymin')) - 10
            ymax = int(each_text.get('ymax')) + 10
            text = each_text.text
            if text is not None:
                cv2.rectangle(cv_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.imwrite(labeled_image, cv_image)
    print("{0} labeled".format(title))