import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier, Pool
import base64

def main():
    st.title("Student Dropout Prediction System")
    st.write("""
    ### Upload your student data to predict their dropout risk
    This system predicts whether a student is likely to continue studying (0) or drop out (1).
    """)

    # File upload widget
    uploaded_file = st.file_uploader("final_data.csv", type="csv")

    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Show the uploaded data
            st.write("### Preview of uploaded data:")
            st.write(df.head())

            # Required columns
            required_columns = [
                'Gender', 'ChildStatus', 'DistanceToSchool', 'BirthOrder',
                'FinancialStatus', 'Residence', 'Transport', 'LightingEnergy'
            ]

            # Check if all required columns are present
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                st.write("Please ensure your CSV file contains all the following columns:")
                st.write(required_columns)
                return

            # Load the saved model
            model = CatBoostClassifier()
            model.load_model('catboost_model.bin')

            # Identify categorical features
            categorical_features = df.select_dtypes(include=['object']).columns.tolist()

            # Create Pool for prediction
            prediction_pool = Pool(data=df[required_columns], 
                                cat_features=categorical_features)

            # Make predictions
            predictions = model.predict(prediction_pool)
            
            # Add predictions to the dataframe
            df['Prediction'] = predictions
            df['Status'] = df['Prediction'].map({0: 'STUDYING', 1: 'DROPPED OUT'})

            # Display results
            st.write("### Prediction Results:")
            st.write(df)

            # Calculate statistics
            total_students = len(df)
            studying = (predictions == 0).sum()
            dropped = (predictions == 1).sum()

            # Display statistics
            st.write("### Summary Statistics:")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Students", total_students)
            with col2:
                st.metric("Studying", studying)
            with col3:
                st.metric("Dropout Risk", dropped)

            # Download results
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please ensure your CSV file is properly formatted and contains all required columns.")

    # Display input format instructions
    st.write("""
    ### Required Input Format:
    Your CSV file should contain the following columns:
    - **Gender**: MALE or FEMALE
    - **ChildStatus**: Orphan, Both parents, or One parent
    - **DistanceToSchool**: Numeric value (in kilometers)
    - **BirthOrder**: Firstborn, Secondborn, Thirdborn, etc.
    - **FinancialStatus**: Poverty, Medium, or Rich
    - **Residence**: House, Apartment, etc.
    - **Transport**: Walking, Car, Public Transit, etc.
    - **LightingEnergy**: Electricity, Solar, etc.
    """)

if __name__ == '__main__':
    main()