from xml.etree import ElementTree
from xml.dom import minidom
from tqdm import tqdm
# from PIL import Image
from shutil import copyfile
# Note : Code took from https://github.com/ssaru/convert2Yolo
# It converts voc annotation to yolo format
from Format import VOC, YOLO
import os

xmls_dir = './datasets/annotations'
images_dir = './datasets/images'

images_voc_dir = './datasets/voc-annotations'

data_yolo = './data'
data_yolo_labels = './data/imgs'

if not os.path.isdir(images_voc_dir):
    os.mkdir(images_voc_dir)
if not os.path.isdir(data_yolo):
    os.mkdir(data_yolo)
if not os.path.isdir(data_yolo_labels):
    os.mkdir(data_yolo_labels)


def get_xml_files():
    return [os.path.join(xmls_dir, xml_file).replace("\\", "/") for xml_file in os.listdir(xmls_dir)]


def create_yolo_data():
    data_path = os.path.join(data_yolo, 'penelope.data').replace("\\", "/")
    train_path = os.path.join(data_yolo, 'train.txt').replace("\\", "/")
    backup_path = os.path.join(data_yolo, 'backup/').replace("\\", "/")

    name_path = os.path.join(data_yolo, 'penelope.name').replace("\\", "/")
    with open(name_path, 'w') as f:
        f.write('text')

    with open(data_path, 'w') as f:
        f.write('classes=1\n')
        f.write('train={0}\nvalid={0}\n'.format(os.path.abspath(train_path)))
        f.write('names={0}\n'.format(os.path.abspath(name_path)))
        if not os.path.isdir(backup_path):
            os.mkdir(backup_path)
        f.write('backup={0}\n'.format(os.path.abspath(backup_path)))


def generate_voc_annotations():
    image_num = 0
    images_train = []
    xml_files = get_xml_files()
    voc_format = VOC()
    yolo_format = YOLO(os.path.abspath(os.path.join(data_yolo, 'penelope.name').replace("\\", "/")))
    with tqdm(bar_format='{l_bar}{bar}{n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        for xml_file in xml_files:
            root = ElementTree.parse(xml_file).getroot()

            title = root.get('title')
            pages = root.find('pages')

            pbar.total = len(xml_files)
            pbar.set_description("Labeling {0}".format(title), refresh=True)
            # pbar.reset()
            pbar.update()

            for page in pages:
                index_img = page.get('index').zfill(3)

                width = page.get('width')
                height = page.get('height')

                texts = page.findall('text')

                # Some page doesn't have texts
                if not texts:
                    continue

                image_path = os.path.join(images_dir, title, index_img + '.jpg').replace("\\", "/")
                new_image_path = os.path.join(data_yolo_labels, str(image_num) + '.jpg').replace("\\", "/")
                copyfile(image_path, new_image_path)

                images_train.append(new_image_path + '\n')

                voc_file = os.path.join(images_voc_dir, str(image_num) + '.xml').replace("\\", "/")
                voc_root = ElementTree.Element("annotations")
                ElementTree.SubElement(voc_root, "filename").text = "{}.jpg".format(image_num)
                ElementTree.SubElement(voc_root, "folder").text = new_image_path

                source = ElementTree.SubElement(voc_root, "source")
                ElementTree.SubElement(source, "database").text = "Unknown"

                size = ElementTree.SubElement(voc_root, "size")
                ElementTree.SubElement(size, "width").text = str(width)
                ElementTree.SubElement(size, "height").text = str(height)
                ElementTree.SubElement(size, "depth").text = "3"

                ElementTree.SubElement(voc_root, "segmented").text = str(0)

                for each_text in texts:
                    # Draw rectangle around the text
                    # +/- 10 because some letters are output of the rectangle
                    x_min = int(each_text.get('xmin')) - 10
                    x_max = int(each_text.get('xmax')) + 10
                    y_min = int(each_text.get('ymin')) - 10
                    y_max = int(each_text.get('ymax')) + 10

                    text = each_text.text
                    if text is not None:
                        obj = ElementTree.SubElement(voc_root, "object")
                        ElementTree.SubElement(obj, "name").text = "text"
                        ElementTree.SubElement(obj, "pose").text = "Unspecified"
                        ElementTree.SubElement(obj, "truncated").text = str(0)
                        ElementTree.SubElement(obj, "difficult").text = str(0)
                        bbox = ElementTree.SubElement(obj, "bndbox")
                        ElementTree.SubElement(bbox, "xmin").text = str(x_min)
                        ElementTree.SubElement(bbox, "ymin").text = str(y_min)
                        ElementTree.SubElement(bbox, "xmax").text = str(x_max)
                        ElementTree.SubElement(bbox, "ymax").text = str(y_max)

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
                        # cv2.rectangle(cv_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

                image_num = image_num + 1
                xml_str = minidom.parseString(ElementTree.tostring(voc_root)).toprettyxml(indent="   ")
                with open(voc_file, "w") as f:
                    f.write(xml_str)

    with open(os.path.join(data_yolo, 'train.txt').replace("\\", "/"), 'w') as f:
        f.writelines(images_train)

    flag, data = voc_format.parse(images_voc_dir)
    if flag:
        flag, data = yolo_format.generate(data)
        if flag:
            flag, data = yolo_format.save(data, data_yolo_labels, data_yolo_labels, '.jpg', data_yolo_labels)
            if not flag:
                print("Saving Result : {}, msg : {}".format(flag, data))
        else:
            print("YOLO Generating Result : {}, msg : {}".format(flag, data))


if __name__ == '__main__':
    create_yolo_data()
    generate_voc_annotations()
