import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, classification_report
import streamlit as st
import joblib
import warnings
warnings.simplefilter('ignore')

# This assumes you have the base.py module with splitter and standardizer functions
# I'll reimplement these if they don't exist
def splitter(data, y_var):
    """Split data into X and y components"""
    X = data.drop(y_var, axis=1)
    y = data[y_var]
    return X, y

def standardizer(X_train, X_test):
    """Standardize features using training data statistics"""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_test_scaled

# Loading the train & test data
train = pd.read_csv('train2.csv')
test = pd.read_csv('test2.csv')

# Splitting the data into independent & dependent variables
X_train, y_train = splitter(train, y_var='DISEASE') 
X_test, y_test = splitter(test, y_var='DISEASE')

# Standardizing the data
scaler, X_train_scaled, X_test_scaled = standardizer(X_train, X_test)

# Define individual classifiers
log_clf = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
svc_clf = SVC(kernel='rbf', C=1, probability=True, random_state=42)

# Create the voting classifier
voting_clf = VotingClassifier(
    estimators=[
        ('lr', log_clf),
        ('rf', rf_clf),
        ('svc', svc_clf)
    ],
    voting='soft'  # Use 'soft' voting to get probability outputs
)

# Train the model
print("Training the voting classifier...")
voting_clf.fit(X_train_scaled, y_train)

# Evaluate on train and test data
train_acc = voting_clf.score(X_train_scaled, y_train)
test_acc = voting_clf.score(X_test_scaled, y_test)

print('Train Accuracy: {:.3f}'.format(train_acc))
print('Test Accuracy: {:.3f}'.format(test_acc))

# Make predictions
y_pred_proba = voting_clf.predict_proba(X_test_scaled)[:, 1]

threshold = 0.60
y_pred_class = np.where(y_pred_proba > threshold, 1, 0)

# ROC curve and AUC score
logit_roc_auc = roc_auc_score(y_test, y_pred_proba)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(10, 5))
plt.plot(fpr, tpr, label='AUC = %0.2f' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('roc_curve.png')  # Save for Streamlit
plt.close()

# Precision-Recall curve
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

threshold_boundary = thresholds.shape[0]
plt.figure(figsize=(10, 5))
plt.plot(thresholds, precisions[0:threshold_boundary], label='Precision')
plt.plot(thresholds, recalls[0:threshold_boundary], label='Recall')
plt.xlabel('Threshold Value')
plt.ylabel('Precision and Recall Value')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.savefig('precision_recall.png')  # Save for Streamlit
plt.close()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_class)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cbar=False, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted label')
plt.ylabel('Actual label')
plt.savefig('confusion_matrix.png')  # Save for Streamlit
plt.close()

# Classification Report
report = classification_report(y_test, y_pred_class)
print(report)

# Save the model and scaler
joblib.dump(voting_clf, 'voting_classifier_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Streamlit App
def voting_clf_app():
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
    
    @st.cache_resource
    def load_model_and_scaler():
        model = joblib.load('voting_classifier_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    
    # Load model and scaler
    try:
        loaded_model, loaded_scaler = load_model_and_scaler()
        model_loaded = True
    except:
        st.warning("Model files not found. Please run the training code first.")
        model_loaded = False
    
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
            if model_loaded:
                # Show a spinner while predicting
                with st.spinner('Analyzing...'):
                    sample = np.array(encoded_results).reshape(1, -1)
                    # Scale the input using the same scaler used during training
                    sample_scaled = loaded_scaler.transform(sample)
                    prediction = loaded_model.predict_proba(sample_scaled)[0, 1]
                
                # Display prediction in a nice box
                st.markdown(
                    f"""
                    <div class="prediction-box">
                        <h2>Prediction Result</h2>
                        <h1 style="font-size: 48px; color: #1E88E5;">{prediction:.2%}</h1>
                        <p>Probability of Heart Disease</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            else:
                st.error("Please train the model first")
    
    with tab2:
        if model_loaded:
            st.markdown("""
            ### Model Evaluation Metrics
            
            Explore the various metrics used to evaluate the model's performance:
            """)
            
            metrics = st.radio(
                "Select Metric to View:",
                ["ROC-AUC Curve", "Precision-Recall", "Confusion Matrix", "Classification Report"],
                horizontal=True
            )
            
            # Show metrics
            if metrics == "ROC-AUC Curve":
                with st.expander("📈 ROC-AUC Curve", expanded=True):
                    st.markdown("#### Receiver Operating Characteristic (ROC) Curve")
                    st.image('roc_curve.png')
                    st.markdown(f"**AUC Score:** {logit_roc_auc:.4f}")
                    
            elif metrics == "Precision-Recall":
                with st.expander("📊 Precision-Recall Curve", expanded=True):
                    st.markdown("#### Precision-Recall Curve")
                    st.image('precision_recall.png')
                    
            elif metrics == "Confusion Matrix":
                with st.expander("🔢 Confusion Matrix", expanded=True):
                    st.markdown("#### Model Confusion Matrix")
                    st.image('confusion_matrix.png')
                    
            elif metrics == "Classification Report":
                with st.expander("📝 Classification Report", expanded=True):
                    st.markdown("#### Detailed Classification Metrics")
                    st.code(report)
            
            # Add explanation of metrics
            with st.expander("📚 Understanding the Metrics"):
                st.markdown("""
                #### Detailed Explanation of Evaluation Metrics
                
                1. **ROC-AUC Curve**
                - Plots true positive rate vs false positive rate
                - Higher AUC indicates better model discrimination
                - Perfect classifier would have AUC = 1.0
                
                2. **Precision-Recall Plot**
                - Shows trade-off between precision and recall
                - Helps in choosing optimal threshold for classification
                - Important for imbalanced datasets
                
                3. **Confusion Matrix**
                - Shows true positives, false positives, true negatives, and false negatives
                - Helps understand model's classification performance
                - Useful for identifying specific types of errors
                
                4. **Classification Report**
                - Provides precision, recall, f1-score, and support for each class
                - Gives a comprehensive overview of model performance
                - Helps identify class imbalance issues
                """)
        else:
            st.warning("Please train the model first to view model evaluation metrics.")

if __name__ == "__main__":
    voting_clf_app()
