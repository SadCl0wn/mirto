
# given an image
img = Image.open('ML-Project/images/armas (2).jpg').convert("RGB")

#
# perform yolo prediction
#
# yolo should return 'yolo_prediction' which is a list of boxes (and a box is a list of coordinates)

# list of boxes around weapons to add on image at the end
boxes = []
labels = []

# for each box around people found by yolo
for i in range(len(yolo_prediction[0]['boxes'])):
    yolo_score = round(yolo_prediction[0]['scores'][i].item(), 2)
    
    # if the confidence score is greater than a threshold
    if yolo_score > 0.8:
        box = prediction[0]['boxes'][i]
        # extend the area around person (20 pixel)
        xmin = int(box[0]) - 20
        ymin = int(box[1]) - 20
        xmax = int(box[2]) + 20
        ymax = int(box[3]) + 20
        
        # crop the initial image according to this new coordinates
        potential_area = img.crop(xmin, ymin, xmax, ymax);
        transform = get_transform(train=False)
        potential_area, _ = transform(potential_area, _)
        
        # put the Faster R-CNN in evaluation mode and detect the image
        model.eval()
        with torch.no_grad():
            prediction = model([potential_area.to(device)])
    
        # for each box found by Faster R-CNN in the current crop of the image
        for i in range(len(prediction[0]['boxes'])):
            score = round(prediction[0]['scores'][i].item(), 2)
            
            # if the confidence score is greater than a threshold
            if score > 0.7:
                label = WEAPON_CATEGORY_NAMES[int(prediction[0]['labels'][i].item())]
                box = prediction[0]['boxes'][i]
                txt = '{} {}'.format(label, score)
                
                # update the box values so that the new coordinates identify the object in the original image
                box[0] += xmin
                box[1] += ymin
                box[2] += xmax
                box[3] += ymax
                
                # add the box and the text label to the lists
                boxes.append(box)
                labels.append(txt)

# read the original image
img = cv2.imread('ML-Project/images/armas (2).jpg') # Read image with cv2

# add boxes found to the image
for i, box in enumerate(boxes):
    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=(0, 255, 0), thickness=2)
    cv2.putText(img, labels[i], (int(box[0]), int(box[1])-5),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,220,0),thickness=1)

# show the image with boxes around weapons
cv2_imshow(img)

