from sklearn.utils import resample

import pandas as pd



from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv1D, MaxPooling1D, BatchNormalization, Input

from tensorflow.keras.models import Model
from tensorflow.python.keras.utils.np_utils import to_categorical

mitbih_train_df = pd.read_csv("data/mitbih_train.csv", header= None)
mitbih_test_df = pd.read_csv("data/mitbih_test.csv", header= None)
equilibre = mitbih_train_df[187].value_counts()

print(equilibre)

df_1 = mitbih_train_df[mitbih_train_df[187] == 1]
df_2 = mitbih_train_df[mitbih_train_df[187] == 2]
df_3 = mitbih_train_df[mitbih_train_df[187] == 3]
df_4 = mitbih_train_df[mitbih_train_df[187] == 4]
df_0 = (mitbih_train_df[mitbih_train_df[187] == 0]).sample(n=20000, replace=True)

df_1_upsample = resample(df_1, replace=True, n_samples=20000)
df_2_upsample = resample(df_2, replace=True, n_samples=20000)
df_3_upsample = resample(df_3, replace=True, n_samples=20000)
df_4_upsample = resample(df_4, replace=True, n_samples=20000)

mitbih_train_df = pd.concat([df_0, df_1_upsample, df_2_upsample, df_3_upsample, df_4_upsample])

equilibre = mitbih_train_df[187].value_counts()

target_train=mitbih_train_df[187]
target_test=mitbih_test_df[187]
y_train=to_categorical(target_train)
y_test=to_categorical(target_test)

X_train=mitbih_train_df.iloc[:,:186].values
X_test=mitbih_test_df.iloc[:,:186].values
#for i in range(len(X_train)):
#    X_train[i,:186]= add_gaussian_noise(X_train[i,:186])
X_train = X_train.reshape(len(X_train), X_train.shape[1],1)
X_test = X_test.reshape(len(X_test), X_test.shape[1],1)
# print(X_train.shape)
# print(y_train.shape)

verbose, epochs, batch_size = 0, 10, 32

im_shape=(X_train.shape[1],1)
print("lassan------------")
im_shape = (X_train.shape[1], 1)
inputs_cnn = Input(shape=(im_shape), name='inputs_cnn')
conv1_1 = Conv1D(64, (6), activation='relu', input_shape=im_shape)(inputs_cnn)
conv1_1 = BatchNormalization()(conv1_1)
pool1 = MaxPooling1D(pool_size=(3), strides=(2), padding="same")(conv1_1)
conv2_1 = Conv1D(64, (3), activation='relu', input_shape=im_shape)(pool1)
conv2_1 = BatchNormalization()(conv2_1)
pool2 = MaxPooling1D(pool_size=(2), strides=(2), padding="same")(conv2_1)
conv3_1 = Conv1D(64, (3), activation='relu', input_shape=im_shape)(pool2)
conv3_1 = BatchNormalization()(conv3_1)
pool3 = MaxPooling1D(pool_size=(2), strides=(2), padding="same")(conv3_1)
flatten = Flatten()(pool3)
dense_end1 = Dense(64, activation='relu')(flatten)
dense_end2 = Dense(32, activation='relu')(dense_end1)
main_output = Dense(5, activation='softmax', name='main_output')(dense_end2)

model = Model(inputs=inputs_cnn, outputs=main_output)
print(model.summary())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train,  epochs=10, batch_size=32, validation_split=0.2, verbose=2)
# scores = model.evaluate(X_train, y_train, verbose=0)
# y_pred = model.predict(X_test)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")