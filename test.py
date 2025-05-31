from ultralytics import YOLO

# 加载训练好的模型  load the model we have trained
#model = YOLO("D:/yolov8_project/scripts/runs/detect/yolov8_pedestrian/weights/best.pt")

# 测试模型 test model
#results = model.predict(
#    source="D:/yolov8_project/datasets/images/val",
#    save=True,
#    conf=0.5
#)
#model = YOLO("runs/detect/yolov8_pedestrian_v2/weights/best.pt")
#results = model.predict(source="D:/yolov8_project/datasets/images/val", conf=0.3, save=True)

from ultralytics import YOLO

if __name__ == '__main__':
    # 加载模型
    model = YOLO("D:/yolov8_project/scripts/runs/detect/train3/weights/best.pt")

    # 对正常天气子集进行评估
    normal_test_results = model.val(data='D:/yolov8_project/datasets/normal_weather_data.yaml')

    # 获取正常天气测试集的评估指标
    normal_mean_results = normal_test_results.mean_results()
    print("Normal weather test accuracy :", normal_mean_results[2])  # mAP50

    # 对正常天气子集进行评估
    #bad_test_results = model.val(data='D:/yolov8_project/datasets/bad_weather_data.yaml')

    # 获取正常天气测试集的评估指标
    #bad_mean_results = bad_test_results.mean_results()
    #print("Bad weather test accuracy :", bad_mean_results[2])  # mAP50