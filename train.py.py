

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import backend as K
from keras.layers import Conv2D , BatchNormalization
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau , ModelCheckpoint , LearningRateScheduler

base_model = InceptionV3(weights='imagenet', include_top=False , input_shape=(299, 299, 3))

x = base_model.output
x = Dropout(0.5)(x)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
predictions = Dense(17, activation='softmax')(x)


model = Model(inputs=base_model.input, outputs=predictions)


lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=0.0001, patience=1, verbose=1)

filepath="/content/drive/My Drive/A/transferlearning_weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')


model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])


data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)


train_generator = data_generator.flow_from_directory(
        '/content/drive/My Drive/A/train',
        target_size=(299, 299),
        batch_size=32,
        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
        '/content/drive/My Drive/A/val',
        target_size=(299, 299),
        batch_size=16,
        class_mode='categorical')

history1 = model.fit_generator(
           train_generator,
           steps_per_epoch=50,
           validation_data = validation_generator,
           epochs=20,
           validation_steps=50,
           callbacks=[lr_reduce,checkpoint])

model.load_weights("/content/drive/My Drive/A/transferlearning_weights.hdf5")

import matplotlib.pyplot as plt

plt.plot(history1.history['acc'])
plt.plot(history1.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig ( "A.png" )

# summarize history for loss
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig ( "L.png" )


