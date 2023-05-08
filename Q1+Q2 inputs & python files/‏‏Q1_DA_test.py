'''Q1'''
#import relevent libraries
import json
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import warnings

class BoundingBox:
    def __init__(self, x_center=0, y_center=0, width=0, height=0, iou=0):
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height
        self.iou = iou


def read_lean_map_of_bboxes(input_json_file_path: str):
    with open(input_json_file_path, 'r') as json_stream:
        raw_object = json.load(json_stream)
    return {k: [BoundingBox(**item) for item in v] for k, v in raw_object.items()}

'''Q1 - Section d'''
#This function gets a dictionary built from keys that are the frames and values that are
#the list of objects for each frame,
#at the same time getting the iou_threshold for testing,
#the function retains the highest height value of an object that passes 
#iou_threshold and at the same time the lowest value of the height of an object
#that passes iou_threshold
def get_minimum_and_maximum_height(boxes, iou):
    min_height = np.math.inf
    max_height = -np.math.inf
    for key, value in boxes.items():
        for box in value:
            if box.iou>iou:
                if box.height>max_height:
                    max_height = box.height
                if box.height<min_height:
                    min_height = box.height
    return min_height, max_height

'''Q1 - Section c'''
#This function gets a dictionary built from keys that are 
#the frames and values that are the list of objects for each frame,
#at the same time receives the iou threshold for testing,
#The function lists all the iou values that exceed the threshold
#The function generates a histogram graph based on a defined number of bins,
#along with the previously defined iou list
#it should be noted that to the graph added the average point of iou
def create_historgram(boxes, iou_threshold):
    iou_list = []
    bins = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
    for key, value in boxes.items():
        for box in value:
            if box.iou>iou_threshold:
                iou_list.append(box.iou)
    sns.set()
    warnings.filterwarnings('ignore')
    sns.histplot(x=iou_list, bins=bins, kde=True)
    plt.title('Count of objects occurrences iou by ranges')
    plt.xlabel('iou range')
    plt.ylabel('Occurrences')
    plt.axvline(x=np.average(np.array(iou_list)), color='red')
    plt.text((np.average(np.array(iou_list))-0.02), 1500, 'Medien', rotation=90)
    plt.tight_layout()
    plt.show()
    
'''Q1 - Section b'''
#This function gets a dictionary built from keys that are the frames and values 
#that are the list of objects for each frame,
#At the same time receives the threshold for testing,
#The function lists all the iou values that pass the threshold and calculates
#them as an average.
#The function returns the total average of all iou values that pass the threshold 
def calculate_average_iou(boxes, iou):
    pass_iou_list=[]
    for key, value in boxes.items():
        for box in value:
            if box.iou>iou:
                pass_iou_list.append(box.iou)  
    avg_pass_iou = np.average(np.array(pass_iou_list))
    return avg_pass_iou

'''Q1 - Section a'''
# this function determine the (x, y)-coordinates of the intersection object - by a calculation of the 
#center x , y to the corners of the object 
# compute the area of intersection object
# compute the area of both the prediction and ground-truth objects
# compute the intersection over union by taking the intersection
# area and dividing it by the sum of prediction + ground-truth
# areas - the interesection area
# return the intersection over union value
def calculate_iou_for_2_boxes(box1, box2):
    
    box1_x = ((box1.x_center - ((box1.width)/2)), box1.x_center + ((box1.width)/2))
    box1_y = ((box1.y_center - ((box1.height)/2)), box1.y_center + ((box1.height)/2))
    box2_x = ((box2.x_center - ((box2.width)/2)), box2.x_center + ((box2.width)/2))
    box2_y = ((box2.y_center - ((box2.height)/2)), box2.y_center + ((box2.height)/2))
    x1 = max(box1_x[0], box2_x[0])
    y1 = max(box1_y[0], box2_y[0])
    x2 = min(box1_x[1], box2_x[1])
    y2 = min(box1_y[1], box2_y[1])
    
    interArea = max(0, (x2 - x1 + 1)) * max(0, (y2 - y1 + 1))
    box1Area = (box1_x[1] - box1_x[0] + 1) * (box1_y[1] - box1_y[0] + 1)
    box2Area = (box2_x[1] - box2_x[0] + 1) * (box2_y[1] - box2_y[0] + 1)
    iou = interArea / float(box1Area + box2Area - interArea)
    return iou
    
	


if __name__ == '__main__':
    path_detection_boxes_json = "Q1_gt.json"
    path_groundtruth_boxes_json = "Q1_system_output.json"
    iou_threshold = 0.5
    
    detection_boxes = read_lean_map_of_bboxes(path_detection_boxes_json)
    ground_truth_boxes = read_lean_map_of_bboxes(path_groundtruth_boxes_json)
    
    '''Q1 - Section a'''
    #because there are keys in the prediction dictionary that do not exist in the truth dictionary,
    #we will use try & except to go through all the objects in the existing frames in both groups,
    #also, a frame that exists in the prediction dictionary and not in the truth dictionary its iou value
    #will be considered zero
    for name, detection_bounding_box_list in detection_boxes.items():
        try:
            ground_truth_bounding_box_list = ground_truth_boxes[name]
            for det_box in detection_bounding_box_list:
                for gt_bbox in ground_truth_bounding_box_list:
                    iou = calculate_iou_for_2_boxes(det_box, gt_bbox)
                    # saving the highest iou for a detection bounding box
                    if iou > getattr(det_box, 'iou', 0):
                        det_box.iou = iou
        except:
            for det_box in detection_bounding_box_list:
                det_box.iou = 0
    
    '''Q1 - Section b'''            
    # calculate average iou for the boxes that pass > iou_threshold
    average_iou = calculate_average_iou(detection_boxes, iou_threshold)
    print(f"The average_iou is: {average_iou}")
    
    '''Q1 - Section c''' 
    # create histogram for the boxes that pass iou_threshold.
    # x_axis: iou, y_axis: occurrences
    create_historgram(detection_boxes, iou_threshold)
    
    '''Q1 - Section d''' 
    # find the minimum and the maximum height for the boxes that pass > iou_threshold
    min_height, max_height = get_minimum_and_maximum_height(detection_boxes, iou_threshold)
    print(f"The min_height is: {min_height} and the max_height is: {max_height}")



