# 📌 Health Insurance Cost Prediction - Machine Learning Project

## 🏆 Problem Statement
Insurance companies need to predict health insurance costs accurately to set premiums appropriately. Traditional methods rely on actuarial tables and historical averages, which often fail to consider individual differences. By utilizing machine learning, insurers can achieve more precise cost estimations tailored to individual profiles, leading to better pricing strategies and risk management.

---

## 🚀 Project Workflow
### 1️⃣ Data Visualization with Tableau Dashboards
Created four Tableau dashboards to analyze key insights:
- **📊 Summary Statistics Dashboard**: Provides an overview of the dataset, including total records, average age, and health condition-wise counts.
- **💰 Premium Pricing Dashboard**: Displays premium-related information like average prices, price distribution by health condition, and age groups.
- **⚠️ Risk Factor Analysis**: Visualizes the impact of health conditions on premium prices (higher risk = higher price).
- **👥 Demographics Insights**: Analyzes the effect of height, weight, BMI, and age on premium prices.

### 2️⃣ Exploratory Data Analysis (EDA) & Hypothesis Testing
- **Data Preprocessing**: Loaded dataset, checked for missing values, and performed basic analysis.
- **Outlier Detection**: Identified and addressed outliers in weight and premium price.
- **Statistical Analysis**: Conducted univariate and multivariate analyses.
- **Hypothesis Testing**:
    - **Example Hypothesis:**
        - **H₀**: Mean premium price is the same for individuals with and without a family history of cancer.
        - **H₁**: Mean premium price differs between the two groups.

#### 🔍 Key Insights from EDA
**Significant factors affecting premium prices:**
✅ Age  
✅ Any Transplants  
✅ Chronic Diseases  
✅ Weight  
✅ Family History of Cancer  
✅ Number of Major Surgeries  

**Minimal impact on premium prices:**
❌ Known Allergies  
❌ Blood Pressure Problems  
❌ Diabetes  

### 3️⃣ Machine Learning Model Training
- **Models Trained:**
    - 🔹 Linear Regression
    - 🔹 Ridge Regression (Tuned)
    - 🔹 Random Forest (Tuned)
    - 🔹 Gradient Boosting (Tuned)
    - 🔹 XGBoost (Tuned)
    - 🔹 Support Vector Regression (SVR) (Tuned)

- **Hyperparameter Tuning:** GridSearchCV was used for optimal parameter selection.

#### 📈 Model Performance Comparison
| Model                          | R² Score | RMSE   | MAE   |
|--------------------------------|---------|--------|--------|
| Linear Regression              | 0.713   | 3495.95 | 2586.23 |
| Ridge Regression (Tuned)       | 0.712   | 3506.14 | 2596.71 |
| Random Forest (Tuned)          | 0.902   | 2047.96 | 978.90 |
| Gradient Boosting (Tuned)      | 0.874   | 2318.99 | 1452.02 |
| XGBoost (Tuned)                | 0.855   | 2482.56 | 1326.84 |
| SVR (Tuned)                    | 0.496   | 4637.22 | 3494.91 |

📌 **Best Model:** Random Forest was the top performer and was saved as a pickle file for deployment.

#### 🔑 Top 5 Features Influencing Premium Prices
1️⃣ Age  
2️⃣ Any Transplants  
3️⃣ Weight  
4️⃣ Any Chronic Disease  
5️⃣ Number of Major Surgeries  

### 4️⃣ Deployment using Streamlit
- **📟 Built an interactive Streamlit Dashboard** with three sections:
  - **📝 Basic Information**: Users input age, height, and weight. BMI is automatically calculated, indicating whether the person is underweight, normal, or obese.
  - **🏥 Medical History**: Users provide details about their medical conditions (e.g., diabetes, blood pressure, chronic diseases).
  - **📊 Premium Prediction**: Users receive an estimated annual premium price based on their inputs.

- **🚀 Deployment:** Hosted on **Streamlit Community Cloud** for easy accessibility.

---
## 🎯 Conclusion
This project successfully demonstrates how machine learning can enhance insurance premium predictions by leveraging individual health data. The **Random Forest model** proved to be the best, providing accurate cost estimations while identifying key influencing factors. The **Streamlit dashboard** ensures an intuitive and user-friendly experience for premium estimation.

💡 **Future Improvements:** Further model optimization, feature engineering, and integration with real-world insurance datasets can enhance prediction accuracy even more. 🔥

---
**🔗 Connect & Explore More** 🚀  
📩 Email: dhruvsharma356@gmail.com   
🔗 LinkedIn: https://www.linkedin.com/in/dhruv-sharma-182828225/

---

