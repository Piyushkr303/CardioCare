
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers

import warnings
warnings.simplefilter('ignore')


# Loading the train & test data -
train = pd.read_csv(r'C:\Users\piyus\OneDrive\Desktop\Heart-Disease-Pred-main\Heart-Disease-Pred-main\train2.csv')
test = pd.read_csv(r'C:\Users\piyus\OneDrive\Desktop\Heart-Disease-Pred-main\Heart-Disease-Pred-main\test2.csv')


# Splitting the data into independent & dependent variables -
X_train, y_train =  base.splitter(train, y_var='DISEASE')
X_test, y_test =  base.splitter(test, y_var='DISEASE')


# Standardizing the data -
X_scaled, X_train_scaled, X_test_scaled = base.standardizer(X_train, X_test)


model = keras.Sequential([
                          keras.layers.Dense(units=128, input_shape=(13,), activation='relu', kernel_regularizer=regularizers.l2(2.0)),
                          keras.layers.BatchNormalization(axis=1),
                          keras.layers.Dense(units=64, activation='relu', kernel_regularizer=regularizers.l2(3.0)),
                          keras.layers.BatchNormalization(axis=1),
                          keras.layers.Dense(units=32, activation='relu', kernel_regularizer=regularizers.l2(3.0)),
                          keras.layers.BatchNormalization(axis=1),
                          keras.layers.Dense(units=16, activation='relu', kernel_regularizer=regularizers.l2(3.0)),
                          keras.layers.BatchNormalization(axis=1),
                          keras.layers.Dense(units=1, activation='sigmoid', kernel_regularizer=regularizers.l2(3.0))
                         ])


adam=keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])


print(model.summary())


es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=1e-3, patience=10, mode='max', verbose=0)
mc = tf.keras.callbacks.ModelCheckpoint(filepath='model.h5', save_best_only=True, save_weights_only=True)


hist = model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), 
                 epochs=100, batch_size=32, callbacks=[es, mc], verbose=1)


_, train_acc = model.evaluate(X_train_scaled, y_train, batch_size=32, verbose=0)
_, test_acc = model.evaluate(X_test_scaled, y_test, batch_size=32, verbose=0)

print('Train Accuracy: {:.3f}'.format(train_acc))
print('Test Accuracy: {:.3f}'.format(test_acc))


y_pred_proba = model.predict(X_test_scaled, batch_size=32, verbose=0)

threshold = 0.60
y_pred_class = np.where(y_pred_proba > threshold, 1, 0)


from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


plt.figure(figsize=(10, 5))
plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='test')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(hist.history['accuracy'], label='train')
plt.plot(hist.history['val_accuracy'], label='test')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


from sklearn.metrics import roc_auc_score, roc_curve

logit_roc_auc = roc_auc_score(y_test, y_pred_proba)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_class)

plt.figure()
plt.plot(fpr, tpr, label='(area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


from sklearn.metrics import precision_recall_curve
    
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

threshold_boundary = thresholds.shape[0]

# plot precision
plt.plot(thresholds, precisions[0:threshold_boundary], label='precision')
# plot recall
plt.plot(thresholds, recalls[0:threshold_boundary], label='recalls')

start, end = plt.xlim()
plt.xticks(np.round(np.arange(start, end, 0.1), 2))

plt.xlabel('Threshold Value'); plt.ylabel('Precision and Recall Value')
plt.legend(); plt.grid()
plt.show()


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_class)

plt.figure(figsize = (10, 7))
sns.heatmap(cm, annot=True, fmt='d', cbar=False, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted label')
plt.ylabel('Actual label')
plt.show()


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_class))


# Saving the model -
model_json = model.to_json()
with open(r'C:\Users\piyus\OneDrive\Desktop\Heart-Disease-Pred-main\Heart-Disease-Pred-main\ann.py', 'w') as json_file:
    json_file.write(model_json)
    
# Serialize weights to HDF5 -
model.save_weights(r'C:\Users\piyus\OneDrive\Desktop\Heart-Disease-Pred-main\Heart-Disease-Pred-main\ann.py')