import os
import base64
import exifread
import requests

count = 0
correct_count = 0
for filename in os.listdir('./'):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        picture = open(filename, 'rb')
        pngb64 = base64.b64encode(picture.read())
        pngtxt = str(pngb64).replace("'", '')[1:]
        tags = exifread.process_file(picture)
        payload = {
                "image": pngtxt,
                "exif": tags,
                }
        response = requests.post("https://us-central1-smartest-df9af.cloudfunctions.net/predict_test", json = payload)
        #print(response.request)
        #print(filename.replace('.png', '').replace('.jpg', '') + ": " + response.text.strip('\n') + ",")

        pred = int(response.text.strip('\n'))
        count += 1
        
        if 'negative' in filename: 
                label = 0
        elif 'positive' in filename:
                label = 1
        elif 'invalid' in filename:
                label = 2
        else:
                print('image name error!')
        
        if pred == label:
                print('GROUND TRUTH: ' + str(label) + '   PREDICTION: ' + str(pred) + '   CORRECT!')
                correct_count += 1
        else:
                print('GROUND TRUTH: ' + str(label) + '   PREDICTION: ' + str(pred) + '   WRONG!')
print('ACCURACY: ', correct_count / count)
        
