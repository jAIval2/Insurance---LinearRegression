import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import streamlit as st

# Load data (replace with your CSV file path)
df = pd.read_csv('C:/Users/sonal/Desktop/HAckx/insurance.csv')

# Preprocess data
df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})
df["sex"] = df["sex"].map({"male": 1, "female": 0})

# Split features and target variable
X = df.drop("charges", axis=1)
y = df["charges"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Define data for prediction (global variable)
data = {
    "age": [],
    "sex": [],
    "bmi": [],
    "children": [],
    "smoker": [],
}


# Function to make predictions
def predict_charges(new_data):
    global predicted_cost  # Make predicted_cost global
    new_df = pd.DataFrame(new_data, index=[new_data["age"]])  # Use age as index
    predicted_cost = model.predict(new_df)[0]
    return predicted_cost


# Streamlit app
st.set_page_config(page_title="Insurance Cost Prediction")

# Homepage
navigation_choice = st.sidebar.radio("Navigation", ["Home", "About Us", "Contact Us", "Analysis"])
if navigation_choice:
    selected_page = navigation_choice

if selected_page == "Home":
    st.header("Insurance Cost Prediction")
    st.subheader("Enter your details to predict insurance charges:")

    # Input fields for prediction
    data["age"] = st.number_input("Age", min_value=0)
    data["sex"] = st.selectbox("Sex", ["Male", "Female"])
    data["bmi"] = st.number_input("BMI")
    data["children"] = st.number_input("Number of Children")
    data["smoker"] = st.selectbox("Smoker", ["Yes", "No"])

    # Convert categorical features
    data["sex"] = 1 if data["sex"] == "Male" else 0
    data["smoker"] = 1 if data["smoker"] == "Yes" else 0

    # Submit button
    if st.button("Predict Charges"):
        predicted_cost = predict_charges(data)  # Call function to make prediction
        st.success(f"Predicted Insurance Cost: ₹{predicted_cost:.2f}")

        # Assuming you have retrieved the predicted cost from the Home page (using the global variable)
        if predicted_cost is not None:  # Check if prediction was made
            if predicted_cost < 5000:
                analysis_text = """Based on your predicted insurance cost, a **low-cost provider** might be suitable. These plans typically offer basic coverage to keep premiums low. Consider factors like deductibles (the amount you pay out of pocket before insurance kicks in) and co-pays (fixed fees for certain services) when making your decision."""
                recommended_policy = "Budget-Friendly Provider"  # Example policy name
                policy_details = """* **Monthly Premium:** Lower than average
                        * **Coverage:** Basic hospital and physician services
                        * **Deductible:** Typically higher
                        * **Co-pay:** May apply for doctor visits and prescriptions
                        * **Network:** Limited network of healthcare providers"""

            elif predicted_cost < 10000:
                analysis_text = """Your predicted cost suggests a **mid-range plan** might be a good fit. These plans offer a balance between affordability and coverage."""
                recommended_policy = "Balanced Coverage Provider"  # Example policy name
                policy_details = """* **Monthly Premium:** Moderate
                        * **Coverage:** Broader range of hospital, physician, and specialist services
                        * **Deductible:** Reasonable deductible amount
                        * **Co-pay:** May apply for doctor visits and prescriptions, potentially lower than low-cost plans
                        * **Network:** Wider network of healthcare providers compared to low-cost plans"""

            else:
                analysis_text = """A **comprehensive insurance plan** with broader coverage might be more appropriate given your predicted cost. Consider consulting with an insurance professional to explore options that best suit your needs."""
                recommended_policy = "Comprehensive Coverage Provider"  # Example policy name
                policy_details = """* **Monthly Premium:** Higher than low- or mid-range plans
                        * **Coverage:** Comprehensive coverage for hospital stays, physician visits, specialists, prescription drugs, and preventive care
                        * **Deductible:** Lower deductible amount compared to lower-cost plans
                        * **Co-pay:** May apply for some services, potentially lower than lower-cost plans
                        * **Network:** Widest network of healthcare providers, including specialists and hospitals"""

            # Display analysis text and policy details in rectangle box
            with st.container():
                st.write(analysis_text)
                st.subheader(f"Recommended Policy: {recommended_policy}")
                st.write(policy_details)
                st.empty()  # Add vertical space

elif selected_page == "About Us":
    st.header("About Us")

    # Headline and Introduction
    st.subheader("Empowering you with informed insurance decisions")
    st.write(
        "This insurance cost prediction app helps you estimate potential healthcare insurance costs based on various factors. By providing basic personal details, you can receive a predicted cost, along with a recommended policy analysis to guide your research.")
    st.write(
        "We understand navigating the world of insurance can be overwhelming. This app aims to simplify the process by offering a user-friendly tool for gaining insights into potential costs.")

    # Team or Developer Information (replace with your details)
    st.subheader("Developed by:")
    st.write("jaival,harsh,vedashree/litbombay")
    st.write("A passionate data enthusiast striving to make insurance cost estimation more accessible.")

    # App Functionality and Benefits
    st.subheader("How it Works:")
    st.write("1. Enter your age, sex, BMI, number of children, and smoker status.")
    st.write("2. Click the 'Predict Charges' button.")
    st.write(
        "3. Receive your predicted insurance cost along with a recommended policy analysis based on the predicted range.")
    st.write(
        "This analysis provides a starting point by suggesting policy types that might be suitable based on the predicted cost. It's crucial to consult with a licensed insurance professional for personalized advice tailored to your specific needs and risk profile.")


elif selected_page == "Contact Us":
    st.header("Contact Us")

    # Contact Information

    st.subheader("We'd love to hear from you!")
    st.write(
        "If you have any questions, feedback, or suggestions, feel free to reach out using the following methods:")

    # Email
    st.write("**Email:** LIT@litbombay.com")

    # Social Media (optional)
    # Replace placeholders with your actual social media handles (if applicable)
    st.write("**Social Media:**")
    st.write("- Twitter: [@LITBOMBAY]")
    st.write("- LinkedIn: [WWW.LITBOMBAY.COM]")

else:
    st.write("Please predict your insurance cost on the Home page to see the analysis.")

