import pandas as pd
from PIL import Image 
  


path='C:\\Users\\DELL\\Downloads\\jessicali9530_stanford-dogs-dataset.csv'

dataset=pd.read_csv(path)
data=dataset[dataset.breed=='african-hunting-dog']
for i in (data.index):
    name=data['annotation_path'][i].split('/')[-1]
    
    img = Image.open(f'C:\\Users\\DELL\\Desktop\\Python\\my computer vision\YOLO\\dog_object_detection\\data\\images\\train\\{name}.jpg') 
    width = img.width 
    height = img.height 

    x1=data['xmin'][i]/width
    x2=data['xmax'][i]/width

    y1=data['ymin'][i]/height
    y2=data['ymax'][i]/height

    w=(x2-x1)
    h=(y2-y1)
    x=x1+(w/2)
    y=y1+(h/2)

    with open(f'{name}.txt', 'a') as f:  
        f.write(f'0 {x} {y} {w} {h}')
        f.close()

