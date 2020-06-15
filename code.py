import cv2

# Checking Version of openCV
print(cv2.__version__)

# Importing cascades
cascade_src1 = 'Audi_Q7_cascade.xml'
"""cascade_src2 = 'Audi_r8_cascade.xml'
cascade_src3 = 'Audi_A6_cascade.xml'
cascade_src4 = 'Bmw_m5_cascade.xml'
cascade_src5 = 'Bmw_i8_cascade.xml'
cascade_src6 = 'Bmw_x7_cascade.xml'
cascade_src7 = 'Merc_Sclass_cascade.xml'
cascade_src8 = 'Merc_Cclass_cascade.xml'
cascade_src9 = 'Merc_Gwagon_cascade.xml'
cascade_src10 = 'Bus_cascade.xml'
cascade_src11 = 'Truck_cascade.xml'
cascade_src12 = 'ford_ecosport.xml'
cascade_src13 = 'ford_endeavour.xml'
cascade_src14 = 'ford_mustang.xml'
"""

# Importing Video
video_src1 = 'Q7_video.mp4'


# Making Classifier Using Cascades
cap = cv2.VideoCapture(video_src1)
src_cascade1 = cv2.CascadeClassifier(cascade_src1)
"""src_cascade2 = cv2.CascadeClassifier(cascade_src2)
src_cascade3 = cv2.CascadeClassifier(cascade_src3)
src_cascade4 = cv2.CascadeClassifier(cascade_src4)
src_cascade5 = cv2.CascadeClassifier(cascade_src5)
src_cascade6 = cv2.CascadeClassifier(cascade_src6)
src_cascade7 = cv2.CascadeClassifier(cascade_src7)
src_cascade8 = cv2.CascadeClassifier(cascade_src8)
src_cascade9 = cv2.CascadeClassifier(cascade_src9)
src_cascade10 = cv2.CascadeClassifier(cascade_src10)
src_cascade11 = cv2.CascadeClassifier(cascade_src11)
src_cascade12 = cv2.CascadeClassifier(cascade_src12)
src_cascade13 = cv2.CascadeClassifier(cascade_src13)
src_cascade14 = cv2.CascadeClassifier(cascade_src14)"""


# Reading Video Frame by Frame
while True:
    ret, img = cap.read()
    if (type(img) == type(None)):
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    car1 = src_cascade1.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in car1:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 165, 255), 5)
        cv2.putText(img, 'Audi(Q7, SUV)', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 1)
    
    
    """car2 = src_cascade2.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in car2:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 60, 255), 5)
        cv2.putText(img, 'Audi(r8, Sports Car)', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 1)
  
    
    car3 = src_cascade3.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in car3:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 165, 200), 5)
        cv2.putText(img, 'Audi(A6, Sedan)', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 1)

    
    car4 = src_cascade4.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in car4:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 165, 155), 5)
        cv2.putText(img, 'Bmw(m5, Sedan)', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 1)

    
    car5 = src_cascade5.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in car5:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 100, 255), 5)
        cv2.putText(img, 'Bmw(i8, Sports Car)', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 1)

    
    car6 = src_cascade6.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in car6:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 50, 125), 5)
        cv2.putText(img, 'Bmw(x7, SUV)', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 1)

    
    car7 = src_cascade7.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in car7:
        cv2.rectangle(img, (x, y), (x + w, y + h), (165, 0, 255), 5)
        cv2.putText(img, 'Merc(S Class, Sedan)', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 1)

    
    car8 = src_cascade8.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in car8:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 5)
        cv2.putText(img, 'Merc(C Class, Sedan)', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 1)

    
    car9 = src_cascade9.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in car9:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 5)
        cv2.putText(img, 'Merc(G Wagon, SUV)', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 1)

    
    bus = src_cascade10.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in bus:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 155), 5)
        cv2.putText(img, 'Bus', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 1)

    
    truck = src_cascade11.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in truck:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 5)
        cv2.putText(img, 'Truck', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 1)
        
    car10 = src_cascade12.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in car10:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 165, 255), 5)
        cv2.putText(img, 'Ford(EcoSport, MUV)', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 1) 

    car11 = src_cascade13.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in car11:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 165, 255), 5)
        cv2.putText(img, 'Ford(Endeavour, SUV)', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 1)

    car12 = src_cascade14.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in car12:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 165, 255), 5)
        cv2.putText(img, 'Ford(Mustang, Sports)', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 1)""" 
        
        
        
    screen_res = 1920, 1080
    scale_width = screen_res[0] / img.shape[1]
    scale_height = screen_res[1] / img.shape[0]
    scale = min(scale_width, scale_height)

    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)

    cv2.namedWindow('Resized Window', cv2.WINDOW_NORMAL)

    cv2.resizeWindow('Resized Window', window_width, window_height)

    cv2.imshow('Resized Window', img)

    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()