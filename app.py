import streamlit as st
from main import ann_app

def main():
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-title {
            font-size: 34px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(to right, #1E88E5, #1565C0);
            color: white;
            border-radius: 10px;
        }
        .section-header {
            font-size: 24px;
            font-weight: bold;
            color: #1E88E5;
            margin: 20px 0;
            padding-bottom: 10px;
            border-bottom: 2px solid #1E88E5;
        }
        .sidebar-header {
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 20px;
            text-align: center;
        }
        .about-box {
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 10px;
            border-left: 4px solid #1E88E5;
            margin: 10px 0;
        }
        .future-scope-item {
            padding: 15px;
            background-color: #f0f8ff;
            border-radius: 8px;
            margin: 10px 0;
        }
        .creator-info {
            text-align: center;
            padding: 20px;
            background-color: #e3f2fd;
            border-radius: 10px;
            margin-top: 30px;
        }
        .highlight-text {
            color: #1E88E5;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    # Main title with enhanced styling
    st.markdown('<p class="main-title">CardioCare: A Heart Disease Prediction using Artificial Neural Networks (ANN)</p>', 
                unsafe_allow_html=True)

    # Enhanced sidebar
    with st.sidebar:
        st.markdown('<p class="sidebar-header">Navigation Menu</p>', unsafe_allow_html=True)
        menu = ['Home', 'Model', 'Metrics', 'About']
        choice = st.selectbox('', menu)

    if choice == 'Home':
        st.markdown('<p class="section-header">Introduction</p>', unsafe_allow_html=True)
        
        # Create tabs for different sections of information
        tabs = st.tabs(["Overview", "Dataset Features"])
        
        with tabs[0]:
            st.markdown("""
            Heart disease prediction using Artificial Neural Networks (ANN) is a cutting-edge approach in medical diagnosis 
            and research. This application demonstrates how machine learning can analyze patient health data to predict 
            the likelihood of heart disease.
            """)
            
            # Add an info box
            st.info("📊 This model analyzes multiple health parameters to provide risk assessment for heart disease.")
            
            # First part of the original text
            st.write("""
            ANN is a sophisticated machine learning technique that can identify complex patterns and relationships within medical data 
            to make accurate predictions or classifications. In the context of heart disease prediction, our ANN model analyzes 
            various patient health indicators to assess the risk of developing heart disease.
            """)

        with tabs[1]:
            # Create an organized feature list with better formatting
            features = {
                "Patient Demographics": {
                    "AGE": "Patient's age - a crucial factor in heart disease risk",
                    "GENDER": "Patient's gender - impacts risk factors and disease manifestation"
                },
                "Vital Signs": {
                    "RESTING_BP": "Resting blood pressure - key indicator of cardiovascular health",
                    "MAX_HEART_RATE": "Maximum heart rate achieved - important stress indicator"
                },
                "Blood Tests": {
                    "SERUM_CHOLESTEROL": "Total cholesterol levels in blood",
                    "TRI_GLYCERIDE": "Triglyceride levels - fat type linked to heart disease",
                    "LDL": "Low-density lipoprotein - 'bad' cholesterol",
                    "HDL": "High-density lipoprotein - 'good' cholesterol",
                    "FBS": "Fasting blood sugar - diabetes indicator"
                },
                "Diagnostic Tests": {
                    "CHEST_PAIN": "Type and severity of chest pain",
                    "RESTING_ECG": "Resting electrocardiogram results",
                    "ECHO": "Echocardiogram findings",
                    "TMT": "Treadmill Test results"
                }
            }

            for category, items in features.items():
                st.markdown(f"**{category}**")
                for feature, description in items.items():
                    st.markdown(f"- **{feature}**: {description}")
                st.write("")

    elif choice == 'Model':
        ann_app()
        
    elif choice == "Metrics":
        st.markdown('<p class="section-header">Model Metrics and Performance</p>', unsafe_allow_html=True)
        with open('exp.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=800)
        f.close()
   
    else:  # About section
        st.markdown('<p class="section-header">About the Project</p>', unsafe_allow_html=True)
        
        # Project significance
        with st.expander("🎯 Project Significance", expanded=True):
            st.markdown("""
            <div class="about-box">
            This project represents a significant contribution to healthcare technology by:
            - 🏥 Addressing a leading global health concern
            - 🔍 Enabling early detection of heart disease
            - 💻 Leveraging advanced machine learning techniques
            - 🤝 Bridging healthcare and technology
            - 📈 Providing data-driven insights for medical professionals
            </div>
            """, unsafe_allow_html=True)

        # Future scope with better organization
        st.markdown('<p class="section-header">Future Scope</p>', unsafe_allow_html=True)
        
        future_scope_items = [
            ("Model Enhancement", "Refining the predictive model with advanced ANN architectures and optimization techniques"),
            ("Healthcare Integration", "Seamless integration with existing healthcare systems and EHR"),
            ("Mobile Solutions", "Development of mobile applications and wearable device integration"),
            ("Advanced Analytics", "Incorporation of genetic data and advanced imaging analysis"),
            ("Clinical Support", "Enhanced decision support tools for healthcare professionals"),
            ("Outcome Prediction", "Long-term outcome predictions and risk assessment"),
            ("Data Integration", "Integration with multiple data sources for comprehensive analysis")
        ]

        cols = st.columns(2)
        for i, (title, description) in enumerate(future_scope_items):
            with cols[i % 2]:
                st.markdown(f"""
                <div class="future-scope-item">
                    <h4>{title}</h4>
                    <p>{description}</p>
                </div>
                """, unsafe_allow_html=True)

        # Creator information
        st.markdown("""
        <div class="creator-info">
            <h3>Created by</h3>
            <h2 class="highlight-text">Piyush Kumar</h2>
            <p>KNIT Sultanpur</p>
        </div>
        """, unsafe_allow_html=True)

main()
