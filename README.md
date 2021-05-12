# Data Augmentation using GANs for Multi-Label Classification of X-Ray Images

In order to reproduce the results, follow these steps:
	1. Download the CheX-Ray14 dataset from https://nihcc.app.box.com/v/ChestXray-NIHCC/file/220660789610 (There is a batch download python script that downloads all dataset files at once) 
	2. Unzip all files to create images directory and perform one-hot encoding on .csv lists provided
	3. There are train validation list and testing list files that the dataset provides. For the classifier, it is enough to put all the dataset in one directory and specify directory path in main.py. For the generator (CUT), a train and test directories must be created. Use xargs to move all the file names mentioned in train validation list and testing list to their train and test directories respectively.
	4. In order to produce the experiments, run the command specified in our submitted report (After setting up the envrionment). You must specify the dataroot that contains both train and test directories. 
	5. After training the generator, you may generate images now through testing (python test.py, more info in paper)
	6. After generating images for a target disease, inject these images into train.csv of the classifier. You may prepend them with Fake and add them to the directory created in Step 2. 
	7. Install the dependecies of the classifer (f1.yml), and Train the classifier on all classes or one target class. After training, change mode in main.py to test and run the script again to get the experiment metrics
	Note: More detailed steps are provided in the paper.
