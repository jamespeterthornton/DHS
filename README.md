# DHS
Passenger Screening Kaggle Competition, hosted by the Department of Homeland Security

19th place solution. 520 entrants. $1.5 million prize pool.

1. Brief Overview of Solution

The core idea of this pipeline is that for each threat zone (e.g. right forearm) we are cropping images of a subject from various angles, concatenating them into a "threat plate", and training a CNN to classify those threat plates. We begin with images of a subject from 16 angles, taken by an airport body scanner. The first step is to normalize the data, as subjects adopt roughly the same pose but that leaves a lot of room for variation. To do so, I cropped the image around an activation threshold and resized this crop to be the full image size. This helped with getting each threat zone into a consistent location. Then, I specify which images to use for each threat zone and specify a large rectangle to be used as the crop. The last big preprocessing step is to exploit to symmetry of the human body to double the size of the dataset. With the right forearm, for example, images of the left forearm will be the same - all that is necessary is to switch and horizonatlly flip certain images. We began with 17 threat zones, but due to this symmetry I only needed to train 9 networks for a complete segmentation approach. 

The model is an ensemble in which each network is a Keras applications ResNet50, pretrained on Imagenet. I left all layers trainable and added an FC-1024 layer on top, followed by another dense layer for the final predictions with sigmoidal activation. I used five fold cross validation to create the ensemble. The first four folds were used to train the ensemble, and the fifth fold was unique - a holdout fold in which half the data was not seen by any network. This was used to test various ensemble techniques.

Finally, the statistical properties of the log loss function (used to score this competition) are such that I knew overconfidence would be heavily penalized. Thus, I clipped each prediction to the [0.015, 0.985] interval, which made a great improvement to my stage 2 score.

2. Review 

Model creation and most of the pipeline is contained in src/tz_plate.py. The other two files are src/logging_callback.py, and src/zone_crops.py - a logging callback and a list of coordinates for image cropping, which I factored out. All other files in src show various prototypes that I experimented with before settling on the final pipeline. 

3. Replication

I'm assuming no one will actually try to replicate these results because we were instructed to delete all of the data at the conclusion of the competition. If you would like to replicate, though, there are some instructions at the top of tz_plate and you will also need to create the expected file tree.

Feel free to message me any questions!
