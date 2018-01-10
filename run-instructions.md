## To run the code
1. Download the dataset

```
pip install -r requirements.txt
chmod +x download.sh
./download.sh
```

2. Building vocab and resize images
```
python build_vocab.py   
python resize.py
```

3. To train the model on prince cluster, submit run.sh
```
sbatch run.sh
```

4. To generate captions for images in a directory 'image_dir'

```
python sample.py --image='image_dir'
```
Or on prince cluster submit test.sh
```
sbatch test.sh
```

