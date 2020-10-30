from tensorflow.python.keras.models import model_from_json
import pandas as pd
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.utils.np_utils import to_categorical
import numpy as np

mitbih_test_df = pd.read_csv("data/mitbih_test.csv", header= None)
target_test=mitbih_test_df[187]
y_test=to_categorical(target_test)
X_test=mitbih_test_df.iloc[:,:186].values
X_test = X_test.reshape(len(X_test), X_test.shape[1],1)


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


scores = loaded_model.evaluate((X_test), y_test, verbose= 2)
ypred = loaded_model.predict(X_test)


print("Accuracy: %.2f%%" % (scores[1]*100))
np.savetxt("pred.csv", ypred, delimiter=",")