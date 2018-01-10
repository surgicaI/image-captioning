import json

data = []

with open('raw/captions') as handle:
    for line in handle:
        id, caption = line.split('\t')
        id = int(id.split('_')[-1].split('.')[0])
        caption = caption.replace('<start>','').replace('<end>','').strip().strip('.').strip()
        imgInfo = {}
        imgInfo['image_id'] = id
        imgInfo['caption'] = caption
        data.append(imgInfo)

data = json.dumps(data)
with open('results/captions.json', 'r+') as handle:
    handle.write(data)