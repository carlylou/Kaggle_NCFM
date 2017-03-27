#--coding:utf-8--#
#!/usr/bin/env python

# This py script for ensemble three trained models together to do the classification task.
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def create_submission(root_path, test_image_list, predictions):
	print('Begin to write submission file ..')
	f_submit = open(os.path.join(root_path, 'submit_ensemble_avg.csv'), 'w')
	f_submit.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')
	for i, image_name in enumerate(test_image_list):
		pred = ['%.6f' % p for p in predictions[i, :]]
		if i % 100 == 0:
			print('{} / {}'.format(i, nbr_test_samples))
		f_submit.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))
	f_submit.close()
	print('Submission file successfully generated!')

def merge_several_folds_mean(data, nfolds):
	a = np.array(data[0])
	for i in range(1, nfolds):
		a += np.array(data[i])
	a /= nfolds
	return a.tolist()

img_width = 299
img_height = 299
batch_size = 32
nbr_test_samples = 1000
nbr_augmentation = 3

FishNames = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

root_path = '/Users/pengpai/Desktop/python/DeepLearning/Kaggle/NCFM'
# test data:
test_data_dir = os.path.join(root_path, 'data/test_stg1/')
# test data generator for prediction
test_datagen = ImageDataGenerator(rescale=1./255)
# model name:
model_name = ['inception', 'resnet', 'vgg16']
y_result = []

for m_name in model_name:
	# model weights path
	weights_path = os.path.join(root_path, m_name, 'weights.h5')
	print('Loading model and weights from training process ...')
	model = load_model(weights_path)
	for idx in range(nbr_augmentation):
	    print('{}th augmentation for testing ...'.format(idx))
	    random_seed = np.random.random_integers(0, 100000)

	    test_generator = test_datagen.flow_from_directory(
	            test_data_dir,
	            target_size=(img_width, img_height),
	            batch_size=batch_size,
	            shuffle = False, # Important !!!
	            seed = random_seed,
	            classes = None,
	            class_mode = None)
	    test_image_list = test_generator.filenames
		print('Begin to predict for testing data ...')
		predictions = model.predict_generator(test_generator, nbr_test_samples)
		y_result.append(predictions)

# merge the predictions produced by all models
test_res = merge_several_folds_mean(y_result, len(model_name)*nbr_augmentation)
# create submission:
create_submission(root_path, test_image_list, test_res)
