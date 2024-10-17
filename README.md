# iLookGood
Open-Source DL transfer learning on the DeepFashion Dataset for multi-label classification and generative outfit creation

## Files Included
### Attr_Pred.py
CustomDataset for the DeepFashion dataset complete with file structure parsing (only for multi-attribute classification out of 1000 attributes)
### dataloader.py
Utilize CustomDataset to save pt files for data (need to expedite this process, possible different format for quicker reading)
### transfer_train.py
Training and Validations loops + saving model and losses
### predict.py
Load model and predict on ONE photo at a time, positive attribute predictions will be outputted in console

## Best Models Downloadable Here
https://drive.google.com/drive/folders/1QrR73iES8IzjCGcLCmvu7T4mNdcmvzkt?usp=sharing
