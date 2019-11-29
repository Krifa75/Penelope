from xml.etree import ElementTree as ET
import os
import cv2

xml_dir = './datasets/annotations'
images_dir = './datasets/images'
images_labeled_dir = './datasets/images-labeled'

os.mkdir(images_labeled_dir)

xml_files = [os.path.join(xml_dir, xml_file).replace("\\", "/") for xml_file in os.listdir(xml_dir)]

index = 0

for xml_file in xml_files:
    tree = ET.parse(xml_file)
    root = tree.getroot()

    title = root.get('title')
    pages = root.find('pages')

    images = os.path.join(images_dir, title).replace("\\", "/")

    os.mkdir(os.path.join(images_labeled_dir, title).replace("\\", "/"))

    for page in pages:
        index_img = page.get('index').zfill(3)
        width = page.get('width')
        height = page.get('height')
        texts = page.findall('text')

        index_str = str(index)
        clean_image = os.path.join(images_dir, title, index_img + '.jpg').replace("\\", "/")

        labeled_image = os.path.join(images_labeled_dir, index_str + '.jpg').replace("\\", "/")
        bboxes = open(os.path.join(images_labeled_dir, index_str + '.txt').replace("\\", "/"), 'w')

        cv_image = cv2.imread(clean_image)

        for each_text in texts:
            id = each_text.get('id')  # Not sure if necessary

            # Draw rectangle around the text 
            # +/- 10 because some letters are output of the rectangle
            xmin = int(each_text.get('xmin')) - 10
            xmax = int(each_text.get('xmax')) + 10
            ymin = int(each_text.get('ymin')) - 10
            ymax = int(each_text.get('ymax')) + 10

            # Convert to yolo format
            dw = 1. / float(width)
            dh = 1. / float(height)
            x = (xmin + xmax) / 2.0
            y = (ymin + ymax) / 2.0
            w = xmax - xmin
            h = ymax - ymin
            x = x * dw
            w = w * dw
            y = y * dh
            h = h * dh
            text = each_text.text
            if text is not None:
                bboxes.write("0 {0} {1} {2} {3}\n".format(x, y, w, h))
                # for debugging
                # cv2.rectangle(cv_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        bboxes.close()
        index = index + 1
        cv2.imwrite(labeled_image, cv_image)
    print("{0} labeled".format(title))

print("finis")
