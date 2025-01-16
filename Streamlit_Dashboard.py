import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Insurance Premium Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 20px;
        background-color: #1f77b4;
        color: white;
        border-radius: 10px;
        padding: 15px;
        font-weight: bold;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #145c8e;
    }
    .prediction-box {
        padding: 30px;
        border-radius: 15px;
        background: linear-gradient(135deg, #f6f8fb 0%, #e9f1f7 100%);
        margin: 20px 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .section-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    .stNumberInput, .stSelectbox {
        margin-bottom: 15px;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stMarkdown {
        color: #34495e;
    }
    .health-tip {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #666;
        cursor: help;
    }
    </style>
""", unsafe_allow_html=True)

# Load the saved model and scaler
@st.cache_resource
def load_model_and_scaler():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, scaler

model, scaler = load_model_and_scaler()

# Header section with description
st.title("🏥 Insurance Premium Predictor")
st.markdown("""
    <div style='background-color: #f8fafc; padding: 20px; border-radius: 10px; margin-bottom: 30px;'>
        <p style='font-size: 1.1em; color: #444;'>
            Get an instant estimate of your insurance premium based on your health profile. 
            Fill in your details below to receive a personalized prediction.
        </p>
    </div>
""", unsafe_allow_html=True)

# Create tabs for different sections
tabs = st.tabs(["📋 Basic Information", "🏥 Medical History", "📊 Prediction"])

with tabs[0]:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Details")
        age = st.number_input("Age", 
                            min_value=1, max_value=100, value=30,
                            help="Enter your current age in years")
        height = st.number_input("Height (cm)", 
                               min_value=100, max_value=250, value=170,
                               help="Enter your height in centimeters")
        weight = st.number_input("Weight (kg)", 
                               min_value=30, max_value=200, value=70,
                               help="Enter your weight in kilograms")
        
        # BMI Calculator
        bmi = weight / ((height/100) ** 2)
        st.markdown(f"""
            <div class='health-tip'>
                <strong>Your BMI:</strong> {bmi:.1f}<br>
                <small>BMI Categories:<br>
                • Underweight: < 18.5<br>
                • Normal: 18.5 - 24.9<br>
                • Overweight: 25 - 29.9<br>
                • Obese: ≥ 30</small>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tabs[1]:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Medical Conditions")
        diabetes = st.selectbox("Diabetes", ["No", "Yes"],
                              help="Select 'Yes' if you have been diagnosed with diabetes")
        blood_pressure = st.selectbox("Blood Pressure Problems", ["No", "Yes"],
                                    help="Select 'Yes' if you have blood pressure issues")
        transplants = st.selectbox("Any Transplants", ["No", "Yes"],
                                 help="Select 'Yes' if you have had any organ transplants")
        chronic_diseases = st.selectbox("Any Chronic Diseases", ["No", "Yes"],
                                      help="Select 'Yes' if you have any chronic conditions")
    
    with col2:
        st.subheader("Additional Health Info")
        allergies = st.selectbox("Known Allergies", ["No", "Yes"],
                               help="Select 'Yes' if you have any known allergies")
        cancer_history = st.selectbox("History of Cancer in Family", ["No", "Yes"],
                                    help="Select 'Yes' if there is a history of cancer in your immediate family")
        major_surgeries = st.number_input("Number of Major Surgeries", 
                                        min_value=0, max_value=10, value=0,
                                        help="Enter the number of major surgeries you have undergone")
    st.markdown("</div>", unsafe_allow_html=True)

with tabs[2]:
    # Function to preprocess input data
    def preprocess_input(data):
        binary_columns = ['diabetes', 'blood_pressure', 'transplants', 
                         'chronic_diseases', 'allergies', 'cancer_history']
        for col in binary_columns:
            data[col] = 1 if data[col] == "Yes" else 0
        
        df = pd.DataFrame([data])
        column_order = ['Age', 'Diabetes', 'BloodPressureProblems', 'AnyTransplants',
                        'AnyChronicDiseases', 'Height', 'Weight', 'KnownAllergies',
                        'HistoryOfCancerInFamily', 'NumberOfMajorSurgeries']
        df.columns = column_order
        
        scaled_features = scaler.transform(df)
        return pd.DataFrame(scaled_features, columns=column_order)

    # Predict button
    if st.button("Calculate Premium Estimate", use_container_width=True):
        with st.spinner("Calculating your premium..."):
            input_data = {
                'age': age, 'height': height, 'weight': weight,
                'diabetes': diabetes, 'blood_pressure': blood_pressure,
                'transplants': transplants, 'chronic_diseases': chronic_diseases,
                'allergies': allergies, 'cancer_history': cancer_history,
                'major_surgeries': major_surgeries
            }
            
            processed_input = preprocess_input(input_data)
            prediction = model.predict(processed_input)[0]
            
            # Display prediction
            st.markdown("<h3 style='text-align: center;'>Your Premium Estimate</h3>", unsafe_allow_html=True)
            
            left_col, pred_col, right_col = st.columns([1, 2, 1])
            
            with pred_col:
                st.markdown(
                    f"""
                    <div class="prediction-box">
                        <h1 style='color: #1f77b4; font-size: 2.5em;'>₹{prediction:,.2f}</h1>
                        <p style='color: #666; font-size: 1.2em;'>Estimated Annual Premium</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # Risk assessment gauge
            risk_score = min(100, max(0, (prediction / 50000) * 100))
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = risk_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Risk Assessment", 'font': {'size': 24}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': "#1f77b4"},
                    'steps': [
                        {'range': [0, 33], 'color': "#e8f4f8"},
                        {'range': [33, 66], 'color': "#bde0ec"},
                        {'range': [66, 100], 'color': "#92cce0"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': risk_score
                    }
                }
            ))
            
            fig.update_layout(
                height=300,
                font={'size': 16},
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Factors affecting premium
            st.markdown("""
                <div class='section-card'>
                    <h3>Factors Influencing Your Premium</h3>
                    <p>Your estimated premium is calculated based on several key factors:</p>
                    <ul style='list-style-type: none; padding-left: 0;'>
                        <li style='margin: 10px 0; padding: 10px; background-color: #f8fafc; border-radius: 5px;'>
                            🧑 <strong>Age and Physical Characteristics:</strong> Including your age, height, and BMI
                        </li>
                        <li style='margin: 10px 0; padding: 10px; background-color: #f8fafc; border-radius: 5px;'>
                            🏥 <strong>Medical History:</strong> Current conditions and past procedures
                        </li>
                        <li style='margin: 10px 0; padding: 10px; background-color: #f8fafc; border-radius: 5px;'>
                            👪 <strong>Family Health History:</strong> Including genetic predispositions
                        </li>
                        <li style='margin: 10px 0; padding: 10px; background-color: #f8fafc; border-radius: 5px;'>
                            🔬 <strong>Current Health Status:</strong> Including any chronic conditions or allergies
                        </li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; background-color: #f8fafc; padding: 20px; border-radius: 10px;'>
        <p style='font-size: 0.9em;'>📊 This prediction model uses advanced machine learning algorithms trained on historical insurance data.</p>
        <p style='font-size: 0.9em;'>⚠️ This tool provides estimates only. Please consult with an insurance provider for accurate quotes.</p>
        <p style='font-size: 0.8em; color: #888;'>© 2025 Insurance Premium Predictor</p>
    </div>
""", unsafe_allow_html=True)