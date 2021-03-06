#--coding:utf-8--#
#!/usr/bin/env python

# This py script for ensemble three trained models together to do the classification task.
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np

def create_submission(root_path, test_image_list, predictions):
	print('Begin to write submission file ..')
	f_submit = open(os.path.join(root_path, 'ensem_submit.csv'), 'w')
	f_submit.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')
	for i, image_name in enumerate(test_image_list):
		pred = ['%.6f' % p for p in predictions[i, :]]
		if i % 100 == 0:
			print('{} / {}'.format(i, nbr_test_samples))
		f_submit.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))
	f_submit.close()
	print('Submission file successfully generated!')



def merge_several_folds_mean(data, nfolds):
	arr_pred = data[0]
	for i in range(1, nfolds):
		arr_pred += data[i]
	arr_pred /= nfolds
	return arr_pred

img_width = 299
img_height = 299
batch_size = 32
nbr_test_samples = 1000

FishNames = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

root_path = '/home/cyang/Kaggle_NCFM-master'

# test data:
test_data_dir = os.path.join(root_path, 'data/test_stg1/')
# test data generator for prediction
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle = False, # Important !!!
        classes = None,
        class_mode = None)
test_image_list = test_generator.filenames

model_name = ['InceptionV3_weights.h5', 'ResNet50_weights.h5', 'VGG16_weights.h5']
y_result = []

for m_name in model_name:
	# model weights path
	weights_path = os.path.join(root_path, m_name)
	print('Loading model and weights from training process ...')
	model = load_model(weights_path)
	print('Begin to predict for testing data ...')
	predictions = model.predict_generator(test_generator, nbr_test_samples)
	y_result.append(predictions)

# merge the predictions produced by all models
test_res = merge_several_folds_mean(y_result, len(model_name))
# create submission:
create_submission(root_path, test_image_list, test_res)


