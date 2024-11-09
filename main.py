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
mc = tf.keras.callbacks.ModelCheckpoint(filepath='model.weights.h5', monitor='val_accuracy', save_best_only=True, save_weights_only=True)

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


@st.cache_resource()#allow_output_mutation=True)
def load_model(model_file):
    model = tf.keras.models.load_model('model.h5')
    return model

def ann_app():
    # Add custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 30px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 30px;
            color: #1E88E5;
        }
        .stExpander {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        .metric-header {
            color: #1E88E5;
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .prediction-box {
            padding: 20px;
            border-radius: 10px;
            background-color: #f0f8ff;
            margin: 20px 0;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    # Main header
    st.markdown('<p class="main-header">Heart Disease Prediction Model</p>', unsafe_allow_html=True)
    
    # Load model
    loaded_model = load_model('model.h5')
    
    # Create tabs for input and analysis
    tab1, tab2 = st.tabs(["📝 Input Parameters", "📊 Model Analysis"])
    
    with tab1:
        # Create two columns with better spacing
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("### Patient Metrics")
            AGE = st.number_input("Age", min_value=0, max_value=120, step=1, 
                                help="Enter patient's age")
            RESTING_BP = st.number_input("Resting Blood Pressure", min_value=0, max_value=300, step=1,
                                       help="Enter resting blood pressure in mm Hg")
            SERUM_CHOLESTROL = st.number_input("Serum Cholesterol", min_value=0, max_value=1000, step=1,
                                             help="Enter serum cholesterol level in mg/dl")
            TRI_GLYCERIDE = st.number_input("Triglycerides", min_value=0, max_value=1000, step=1,
                                          help="Enter triglyceride level in mg/dl")
            LDL = st.number_input("LDL", min_value=0, max_value=300, step=1,
                                help="Enter LDL cholesterol level in mg/dl")
            HDL = st.number_input("HDL", min_value=0, max_value=100, step=1,
                                help="Enter HDL cholesterol level in mg/dl")
            FBS = st.number_input("Fasting Blood Sugar", min_value=0, max_value=500, step=1,
                                help="Enter fasting blood sugar level in mg/dl")
           
        with col2:
            st.markdown("### Clinical Parameters")
            GENDER = st.selectbox('Gender', 
                                options=[0, 1], 
                                format_func=lambda x: "Female" if x == 0 else "Male",
                                help="Select patient's gender")
            
            CHEST_PAIN = st.selectbox('Chest Pain', 
                                    options=[0, 1],
                                    format_func=lambda x: "No" if x == 0 else "Yes",
                                    help="Presence of chest pain")
            
            RESTING_ECG = st.selectbox('Resting ECG', 
                                     options=[0, 1],
                                     format_func=lambda x: "Normal" if x == 0 else "Abnormal",
                                     help="Resting electrocardiographic results")
            
            TMT = st.selectbox('TMT (Treadmill Test)', 
                             options=[0, 1],
                             format_func=lambda x: "Normal" if x == 0 else "Abnormal",
                             help="Treadmill test results")
            
            ECHO = st.number_input("Echo", min_value=0, max_value=100, step=1,
                                 help="Echocardiogram value")
            
            MAX_HEART_RATE = st.number_input("Maximum Heart Rate", 
                                           min_value=0, max_value=250, step=1,
                                           help="Maximum heart rate achieved")
        
        # Collect all inputs
        encoded_results = [AGE, GENDER, CHEST_PAIN, RESTING_BP, SERUM_CHOLESTROL, 
                         TRI_GLYCERIDE, LDL, HDL, FBS, RESTING_ECG, MAX_HEART_RATE, 
                         ECHO, TMT]
        
        # Add a predict button
        if st.button('Predict', type='primary', use_container_width=True):
            # Show a spinner while predicting
            with st.spinner('Analyzing...'):
                sample = np.array(encoded_results).reshape(1, -1)
                prediction = loaded_model.predict(sample)
                rounded_prediction = np.around(prediction[0], decimals=2)
            
            # Display prediction in a nice box
            st.markdown(
                f"""
                <div class="prediction-box">
                    <h2>Prediction Result</h2>
                    <h1 style="font-size: 48px; color: #1E88E5;">{rounded_prediction[0]:.2%}</h1>
                    <p>Probability of Heart Disease</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
    
    with tab2:
        st.markdown("""
        ### Model Evaluation Metrics
        
        Explore the various metrics used to evaluate the model's performance:
        """)
        
        metrics = st.radio(
            "Select Metric to View:",
            ["ROC-AUC Curve", "Model Loss", "Model Accuracy", "Precision-Recall", "Confusion Matrix"],
            horizontal=True
        )
        
        # Create expandable sections for each plot
        if metrics == "ROC-AUC Curve":
            with st.expander("📈 ROC-AUC Curve", expanded=True):
                st.markdown("#### Receiver Operating Characteristic (ROC) Curve")
                logit_roc_auc = roc_auc_score(y_test, y_pred_proba)
                fpr, tpr, thresholds = roc_curve(y_test, y_pred_class)
                
                fig = plt.figure(figsize=(10, 6))
                plt.plot(fpr, tpr, label=f'AUC = {logit_roc_auc:.2f}')
                plt.plot([0, 1], [0, 1], 'r--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc="lower right")
                plt.grid(True)
                st.pyplot(fig)
                
        elif metrics == "Model Loss":
            with st.expander("📉 Model Loss Curve", expanded=True):
                st.markdown("#### Training and Validation Loss")
                fig = plt.figure(figsize=(10, 6))
                plt.plot(hist.history['loss'], label='Training Loss')
                plt.plot(hist.history['val_loss'], label='Validation Loss')
                plt.title('Model Loss Over Time')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend(loc='upper right')
                plt.grid(True)
                st.pyplot(fig)
                
        elif metrics == "Model Accuracy":
            with st.expander("📊 Model Accuracy Curve", expanded=True):
                st.markdown("#### Training and Validation Accuracy")
                fig = plt.figure(figsize=(10, 6))
                plt.plot(hist.history['accuracy'], label='Training Accuracy')
                plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
                plt.title('Model Accuracy Over Time')
                plt.xlabel('Epochs')
                plt.ylabel('Accuracy')
                plt.legend(loc='lower right')
                plt.grid(True)
                st.pyplot(fig)
                
        elif metrics == "Precision-Recall":
          with st.expander('Precision-Recall Plot'):
            st.subheader("Precision-Recall Plot")
            try:
              precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
            
              # Ensure all arrays have the same length
              min_length = min(len(thresholds), len(precisions) - 1, len(recalls) - 1)
              thresholds = thresholds[:min_length]
              precisions = precisions[:min_length]
              recalls = recalls[:min_length]

              fig4 = plt.figure(figsize=(10, 5))
              plt.plot(thresholds, precisions[:min_length], label='Precision')
              plt.plot(thresholds, recalls[:min_length], label='Recall')
            
              plt.xlabel('Threshold Value')
              plt.ylabel('Precision and Recall Value')
              plt.legend()
              plt.grid()
              st.pyplot(fig4)
              plt.close(fig4)
            except Exception as e:
              st.error(f"Error in precision-recall plotting: {str(e)}")
                
        else:  # Confusion Matrix
            with st.expander("🔢 Confusion Matrix", expanded=True):
                st.markdown("#### Model Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred_class)
                fig = plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                st.pyplot(fig)

        # Add explanation of metrics
        with st.expander("📚 Understanding the Metrics"):
            st.markdown("""
            #### Detailed Explanation of Evaluation Metrics
            
            1. **ROC-AUC Curve**
            - Plots true positive rate vs false positive rate
            - Higher AUC indicates better model discrimination
            - Perfect classifier would have AUC = 1.0
            
            2. **Model Loss Curve**
            - Shows how well the model is learning over time
            - Decreasing loss indicates improving model performance
            - Gap between training and validation loss helps identify overfitting
            
            3. **Model Accuracy Curve**
            - Tracks prediction accuracy during training
            - Higher accuracy indicates better model performance
            - Helps monitor potential overfitting
            
            4. **Precision-Recall Plot**
            - Shows trade-off between precision and recall
            - Helps in choosing optimal threshold for classification
            - Important for imbalanced datasets
            
            5. **Confusion Matrix**
            - Shows true positives, false positives, true negatives, and false negatives
            - Helps understand model's classification performance
            - Useful for identifying specific types of errors
            """)
