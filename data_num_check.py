import xml.dom.minidom
import os

root_path = 'E:/final/2D_label_parser-master/'
annotation_path = root_path + 'target_labels/CAM_ALL_TXT/'
img_path = 'E:/final/v1.0-mini/samples/CAM_ALL/'

annotation_list = os.listdir(annotation_path)
img_list = os.listdir(img_path)

if len(img_list) != len(annotation_list):
    print("num not match")
    if len(img_list) < len(annotation_list):
        print("labels' num > images'")
        error_xml = []
        for _ in annotation_list:
            xml_name = _.split('.')[0]
            img_name = xml_name + '.jpg'
            if img_name not in img_list:
                error_xml.append(_)
                os.remove(os.path.join(annotation_path, _))
        print("error xml:", error_xml)
    else:
        print("images' num > labels'")
        error_img = []
        for _ in img_list:
            img_name = _.split('.')[0]
            xml_name = img_name + '.txt'
            if xml_name not in annotation_list:
                error_img.append(_)
                os.remove(os.path.join(img_path, _))
        print("lack label's image:", error_img)
