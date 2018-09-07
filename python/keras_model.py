import keras
from keras.models import Model
from keras.layers import Input, Conv1D, BatchNormalization, GlobalMaxPooling1D, Activation
from utils import get_train_test


n_classes = 10


inputs = Input(shape=(None,1))
x = inputs

for _ in range(12):
    x = Conv1D(64, 3, strides=2, padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)

x = Conv1D(n_classes, 3, strides=2)(x)
x = GlobalMaxPooling1D()(x)
y = Activation('softmax')(x)

model = Model(inputs=inputs, outputs=y)


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

train, test = get_train_test(1, True)
model.fit_generator(train.batch_gen(16), steps_per_epoch=10000)  # starts training



