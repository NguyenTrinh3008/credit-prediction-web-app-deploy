import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open('E:\AI4E-Project\credit_classifier_model.sav', 'rb'))


# creating a function for Prediction

def credit_prediction(input_data):

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data = input_data_as_numpy_array.reshape(1,-1)

    adaboost_predictions_new = loaded_model.predict(input_data)

    # Print the predictions
    print(adaboost_predictions_new)

    if (adaboost_predictions_new[0] == 0):
        return'The person is good loan'
    else:
        return'The person is bad loan'
  
def main():    
    # giving a title
    st.title('Credit Prediction Web App')
    
    
    # getting the input data from the user
    duration = st.text_input('Duration')
    credit_amount = st.text_input('Credit Amount')
    installment_rate = st.text_input('Installment Rate')
    residence_since = st.text_input('Residence Since')
    age = st.text_input('Age')
    number_of_existcr = st.text_input('Number of Existing Credits')
    number_of_dependents = st.text_input('Number of Dependents')
    telephon = st.text_input('Telephone')
    foreign = st.text_input('Foreign')

    account_bal_neg_bal = st.checkbox('Account Balance - Negative Balance')
    account_bal_no_acc = st.checkbox('Account Balance - No Account')
    account_bal_positive_bal = st.checkbox('Account Balance - Positive Balance')

    payment_status_A30 = st.checkbox('Payment Status - A30')
    payment_status_A31 = st.checkbox('Payment Status - A31')
    payment_status_A32 = st.checkbox('Payment Status - A32')
    payment_status_A33 = st.checkbox('Payment Status - A33')
    payment_status_A34 = st.checkbox('Payment Status - A34')

    purpose_A40 = st.checkbox('Purpose - A40')
    purpose_A41 = st.checkbox('Purpose - A41')
    purpose_A410 = st.checkbox('Purpose - A410')
    purpose_A42 = st.checkbox('Purpose - A42')
    purpose_A43 = st.checkbox('Purpose - A43')
    purpose_A44 = st.checkbox('Purpose - A44')
    purpose_A45 = st.checkbox('Purpose - A45')
    purpose_A46 = st.checkbox('Purpose - A46')
    purpose_A48 = st.checkbox('Purpose - A48')
    purpose_A49 = st.checkbox('Purpose - A49')

    savings_bond_value_A61 = st.checkbox('Savings Bond Value - A61')
    savings_bond_value_A62 = st.checkbox('Savings Bond Value - A62')
    savings_bond_value_A63 = st.checkbox('Savings Bond Value - A63')
    savings_bond_value_A64 = st.checkbox('Savings Bond Value - A64')
    savings_bond_value_A65 = st.checkbox('Savings Bond Value - A65')

    employed_since_A71 = st.checkbox('Employed Since - A71')
    employed_since_A72 = st.checkbox('Employed Since - A72')
    employed_since_A73 = st.checkbox('Employed Since - A73')
    employed_since_A74 = st.checkbox('Employed Since - A74')
    employed_since_A75 = st.checkbox('Employed Since - A75')

    sex_marital_A91 = st.checkbox('Sex/Marital Status - A91')
    sex_marital_A92 = st.checkbox('Sex/Marital Status - A92')
    sex_marital_A93 = st.checkbox('Sex/Marital Status - A93')
    sex_marital_A94 = st.checkbox('Sex/Marital Status - A94')

    guarantor_A101 = st.checkbox('Guarantor - A101')
    guarantor_A102 = st.checkbox('Guarantor - A102')
    guarantor_A103 = st.checkbox('Guarantor - A103')

    most_valuable_asset_car = st.checkbox('Most Valuable Asset - Car')
    most_valuable_asset_life_insurance = st.checkbox('Most Valuable Asset - Life Insurance')
    most_valuable_asset_none = st.checkbox('Most Valuable Asset - None')
    most_valuable_asset_real_estate = st.checkbox('Most Valuable Asset - Real Estate')

    concurrent_credits_A141 = st.checkbox('Concurrent Credits - A141')
    concurrent_credits_A142 = st.checkbox('Concurrent Credits - A142')
    concurrent_credits_A143 = st.checkbox('Concurrent Credits - A143')

    type_of_housing_A151 = st.checkbox('Type of Housing - A151')
    type_of_housing_A152 = st.checkbox('Type of Housing - A152')
    type_of_housing_A153 = st.checkbox('Type of Housing - A153')

    job_highly_skilled = st.checkbox('Job - Highly Skilled')
    job_skilled = st.checkbox('Job - Skilled')
    job_unskilled = st.checkbox('Job - Unskilled')

    
    
    # code for Prediction
    diagnosis = ''
    
    
    # creating a button for Predictio

    # Button for prediction
    if st.button('Credit Test Result'):
        try:
        # Convert inputs to appropriate numeric types
            input_features = [
                float(duration), float(credit_amount), float(installment_rate), float(residence_since), float(age),
                int(number_of_existcr), int(number_of_dependents), int(telephon), int(foreign),
                int(account_bal_neg_bal), int(account_bal_no_acc), int(account_bal_positive_bal),
                int(payment_status_A30), int(payment_status_A31), int(payment_status_A32), int(payment_status_A33), int(payment_status_A34),
                int(purpose_A40), int(purpose_A41), int(purpose_A410), int(purpose_A42), int(purpose_A43), int(purpose_A44),
                int(purpose_A45), int(purpose_A46), int(purpose_A48), int(purpose_A49),
                int(savings_bond_value_A61), int(savings_bond_value_A62), int(savings_bond_value_A63), int(savings_bond_value_A64), int(savings_bond_value_A65),
                int(employed_since_A71), int(employed_since_A72), int(employed_since_A73), int(employed_since_A74), int(employed_since_A75),
                int(sex_marital_A91), int(sex_marital_A92), int(sex_marital_A93), int(sex_marital_A94),
                int(guarantor_A101), int(guarantor_A102), int(guarantor_A103),
                int(most_valuable_asset_car), int(most_valuable_asset_life_insurance), int(most_valuable_asset_none), int(most_valuable_asset_real_estate),
                int(concurrent_credits_A141), int(concurrent_credits_A142), int(concurrent_credits_A143),
                int(type_of_housing_A151), int(type_of_housing_A152), int(type_of_housing_A153),
                int(job_highly_skilled), int(job_skilled), int(job_unskilled)
            ]
        
            # Prediction function
            diagnosis = credit_prediction(input_features)
            st.success(diagnosis)
        except ValueError as e:
            st.error(f"Error in input conversion: {e}")


if __name__ == '__main__':
    main()