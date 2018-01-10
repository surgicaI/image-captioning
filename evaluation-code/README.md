To get the BLEU score and CIDEr score:
1. Copy the captions generated on the validation set using sample.py or test.sh in the main code.
2. Paste that captions file in raw directory and rename that file to captions.
3. Then run 'python3 caption2json.py'
4. This will create captions.json file in results directory.
5. Then run python3 eval.py
6. This will print BLEU and CIDEr scores.
