import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
import streamlit as st
import h5py 

import warnings
warnings.simplefilter('ignore')


# Loading the train & test data -
train = pd.read_csv('train2.csv')
test = pd.read_csv('test2.csv')


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


# from keras.utils.vis_utils import plot_model
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


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


model.save('model.h5')


attrib_info = """
#### Fields:
    -age
    -sex
    -cp
    -trestbps
    -chol
    -fbs
    -restecg
    -thalach
    -exang
    -oldpeak
    -slope
"""

@st.cache_resource()#allow_output_mutation=True)


def load_model(model_file):
    h5_file = h5py.File('model.h5', 'r')
    model = tf.keras.models.load_model(h5_file)
    return model

def ann_app():
    st.subheader("ANN Model Section")
    loaded_model = load_model('model.h5')
    
    col1,col2=st.columns(2)
    
    with col1:
        AGE = st.number_input("AGE", step=1)
        RESTING_BP = st.number_input("RESTING BP", step=1)
        SERUM_CHOLESTROL = st.number_input("SERUM CHOLESTROL", step=1)
        TRI_GLYCERIDE = st.number_input("TRI GLYCERIDE", step=1)
        LDL = st.number_input("LDL", step=1)
        HDL = st.number_input("HDL", step=1)
        FBS = st.number_input("FBS", step=1)
           
    with col2:
        GENDER = st.selectbox('GENDER', [0, 1])
        CHEST_PAIN = st.selectbox('CHEST PAIN', [0, 1])
        RESTING_ECG = st.selectbox('ESTING ECG', [0, 1])
        TMT = st.selectbox('TMT', [0, 1])
        ECHO = st.number_input("ECHO", step=1)
        MAX_HEART_RATE = st.number_input("MAX HEART RATE", step=1)
        
        encoded_results = [AGE, GENDER, CHEST_PAIN, RESTING_BP, SERUM_CHOLESTROL, TRI_GLYCERIDE, LDL, HDL, FBS, RESTING_ECG, MAX_HEART_RATE, ECHO, TMT]
    
    with st.expander('Predicted'):
        sample = np.array(encoded_results).reshape(1, -1)
        prediction = loaded_model.predict(sample)
        rounded_prediction = np.around(prediction[0], decimals=2)  # Round the value to 2 decimal places
        st.success(rounded_prediction[0])
        
    st.write("""
             **Now, let's delve deeper into the intricacies of the evaluation metrics employed to gauge the remarkable performance exhibited by the model in question.**
             
1. ROC-AUC Curve: The Receiver Operating Characteristic (ROC) curve plots the true positive rate against the false positive rate at various classification thresholds. The Area Under the Curve (AUC) quantifies the classifier's ability to distinguish between positive and negative instances. A higher AUC indicates better discrimination, implying that the model can effectively differentiate between individuals with heart disease and those without.

2. Model Loss Curve: The model loss curve illustrates the training and validation loss over the course of training epochs. The loss function measures the discrepancy between the predicted and actual values. Monitoring the loss curve helps assess the model's convergence and whether it's overfitting or underfitting the data. Ideally, we aim for the training and validation loss to decrease steadily until convergence.

3. Model Accuracy Curve: The model accuracy curve tracks the training and validation accuracy as the model learns from the data. Accuracy represents the proportion of correctly classified instances out of the total. By observing the accuracy curve, we can determine if the model is progressively improving or if there are signs of overfitting or underfitting.

4. Precision-Recall Plot: The Precision-Recall plot showcases the trade-off between precision and recall at various classification thresholds. Precision measures the proportion of true positive predictions out of all positive predictions, while recall calculates the proportion of true positive predictions out of all actual positive instances. The plot helps evaluate the model

5. Confusion Matrix summarizes the model's classification results, depicting true positives, true negatives, false positives, and false negatives, offering a comprehensive view of the model's performance across different classes. Collectively, these evaluation matrices aid in understanding and fine-tuning the ANN model for heart disease prediction.
""")


    def plot():
     
     with st.expander('ROC-AUC Curve'):
        st.subheader("ROC-AUC Curve")
        from sklearn.metrics import roc_auc_score, roc_curve

        logit_roc_auc = roc_auc_score(y_test, y_pred_proba)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_class)

        fig1=plt.figure(figsize=(10, 5))
        plt.plot(fpr, tpr, label='(area = %0.2f)' % logit_roc_auc)
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
        st.pyplot(fig1)
        
     with st.expander('Model Loss Curve'):
        st.subheader("Model Loss Curve")
        fig2=plt.figure(figsize=(10, 5))
        plt.plot(hist.history['loss'], label='train')
        plt.plot(hist.history['val_loss'], label='test')
        plt.title('Model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()
        st.pyplot(fig2)
        
     with st.expander('Model Accuracy Curve'):
        st.subheader("Model Accuracy Curve")
        fig3=plt.figure(figsize=(10, 5))
        plt.plot(hist.history['accuracy'], label='train')
        plt.plot(hist.history['val_accuracy'], label='test')
        plt.title('Model Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()
        st.pyplot(fig3)
        
     with st.expander('Precision-Recall Plot'):
        st.subheader("Precision-Recall Plot")
       
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

        threshold_boundary = thresholds.shape[0]
        threshold_boundary = min(len(thresholds), len(precisions), len(recalls))
        thresholds = thresholds[:threshold_boundary]
        precisions = precisions[:threshold_boundary]
        recalls = recalls[:threshold_boundary]

        # plot precision
        fig4 = plt.figure(figsize=(10, 5))
        plt.plot(thresholds, precisions, label='Precision')
    
        # plot recall
        plt.plot(thresholds, recalls, label='Recall')
        
        start, end = plt.xlim()
        plt.xticks(np.round(np.arange(start, end, 0.1), 2))
        plt.xlabel('Threshold Value')
        plt.ylabel('Precision and Recall Value')
        plt.legend()
        plt.grid()
        
        # Display the plot in Streamlit
        st.pyplot(fig4)
    
        
     with st.expander('Confusion Matrix'):
        st.subheader("Confusion Matrix")
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred_class)
        fig5 = plt.figure(figsize = (10, 5))
        sns.heatmap(cm, annot=True, fmt='d', cbar=False, cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted label')
        plt.ylabel('Actual label')
        plt.show()
        st.pyplot(fig5)
        
    plot()
        
