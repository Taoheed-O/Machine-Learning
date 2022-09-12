import pickle
import numpy as np
import pandas as pd
import streamlit as st


# loading in the model to predict on the data
scaler_in = open('scaler.pickle', 'rb')
scaler = pickle.load(scaler_in)

# loading Logistic Regression model
pickle_lr = open("model_lr.pickle", "rb")
classifier_lr = pickle.load(pickle_lr)

# Loading Decision Tree model
pickle_dt = open("model_dt.pickle", "rb")
classifier_dt = pickle.load(pickle_dt)

# Image
st.image("https://pngimg.com/uploads/titanic/titanic_PNG36.png")
def welcome():
	return 'welcome all'

# defining the function which will make the prediction{Logistic regression}using the user inputs
def predict_lr(sex, age, n_siblings_spouses, parch, fare, Class, alone):
    features = ['sex_female', 'n_siblings_spouses_8', 'n_siblings_spouses_1',
    'parch_6', 'n_siblings_spouses_4', 'parch_0', 'parch_5', 'n_siblings_spouses_0', 'parch_3',
    'sex_male', 'Class_First', 'parch_2', 'alone_y', 'n_siblings_spouses_5', 'n_siblings_spouses_2',
    'n_siblings_spouses_3', 'Class_Second', 'parch_1', 'alone_n', 'Class_Third', 'parch_4']
    labels = ['sex', 'age', 'n_siblings_spouses', 'parch', 'fare', 'Class', 'alone']
    feature_names = [sex, age, n_siblings_spouses, parch, fare, Class, alone]
    features_df = pd.DataFrame([feature_names], columns=labels)
    categorical_features = ['sex', 'n_siblings_spouses', 'parch', 'Class', 'alone']
    numeric_features = ['age', 'fare']
    features_df[numeric_features] = scaler.transform(features_df[numeric_features])
    features_df = pd.get_dummies(features_df,columns=categorical_features)
    #setting aside and making up for the whole categorical features from our first model
    c_engineering_features = set(features_df.columns) - set(numeric_features)
    missing_features = list(set(features) - c_engineering_features)
    for feature in missing_features:
        #add zeroes
        features_df[feature] = [0]*len(features_df)
    result = classifier_lr.predict(features_df)
    return result

# defining the function which will make the prediction{Decision Tree}using the user inputs
def predict_dt(sex, age, n_siblings_spouses, parch, fare, Class, alone):
    features = ['sex_female', 'n_siblings_spouses_8', 'n_siblings_spouses_1',
    'parch_6', 'n_siblings_spouses_4', 'parch_0', 'parch_5', 'n_siblings_spouses_0', 'parch_3',
    'sex_male', 'Class_First', 'parch_2', 'alone_y', 'n_siblings_spouses_5', 'n_siblings_spouses_2',
    'n_siblings_spouses_3', 'Class_Second', 'parch_1', 'alone_n', 'Class_Third', 'parch_4']
    labels = ['sex', 'age', 'n_siblings_spouses', 'parch', 'fare', 'Class', 'alone']
    feature_names = [sex, age, n_siblings_spouses, parch, fare, Class, alone]
    features_df = pd.DataFrame([feature_names], columns=labels)
    categorical_features = ['sex', 'n_siblings_spouses', 'parch', 'Class', 'alone']
    numeric_features = ['age', 'fare']
    features_df[numeric_features] = scaler.transform(features_df[numeric_features])
    features_df = pd.get_dummies(features_df,columns=categorical_features)
    #setting aside and making up for the whole categorical features from our first model
    c_engineering_features = set(features_df.columns) - set(numeric_features)
    missing_features = list(set(features) - c_engineering_features)
    for feature in missing_features:
        #add zeroes
        features_df[feature] = [0]*len(features_df)
    result = classifier_dt.predict(features_df)
    return result


#The parameters and their input formats.
# Gender
st.text("Male / Female")
status = st.radio("Select gender", ('male', 'female'))

#conditionals for status
if status == 'male':
	sex = 'male'
else:
	sex = 'female'

# Age
try:
	age = st.number_input("What is the age ?")
except:
	st.text("Enter the age of the passenger.")

# Spouses and siblings
st.text("Number of spouses & siblings. Max: 10")
n_siblings_spouses = st.slider("Select the number of siblings or spouses", 1,10)

# Parch
st.text("Parch number [1 - 6] ")
parch = st.slider("Select parch number", 1, 6)

# Fare
st.text("Fare")
try:
	fare = st.number_input("Thousand Dollars($)")
except:
	st.text("Enter a value for the fare")

# Class 
st.text("First/Second/Third class")
Class = st.radio("Select Class", ('First', 'Second', ' Third')) 

# Alone
alone = st.radio("Is the passenger alone ?", ('yes', 'no'))
#conditionals for alone status
if alone == 'male':
	alone = 'male'
else:
	alone = 'female'



# this is the main function in which is defined on the webpage
def main():
	# giving the webpage a title
	st.title("Machine Learning [ Classification ]")
	
	# the font and background color, the padding and the text to be displayed
	html_temp = """
	<div style ="background-color:black;padding:13px">
	<h1 style ="color:white;text-align:center;">Titanic Survivors Prediction App</h1>
	</div>
	"""
	
	# this line allows us to display the front end aspects we have
	# defined in the above code
	st.markdown(html_temp, unsafe_allow_html = True)
	
	#List of available models 
	options = st.radio("Available Models:", ["Logistic Regression", "Decision Tree"])
	result =""

	# the below line ensures that when the button called 'Predict' is clicked,
	# the prediction function defined above is called to make the prediction
	# and store it in the variable result
	if options == "Logistic Regression":
		st.success("You picked {}".format(options))

		if st.button('Predict'):
			result = predict_lr(sex, age, n_siblings_spouses, parch, fare, Class, alone)
			if result[0] == 0:
				st.success('Not a Survivor')
			else:
				st.error('A Survivor')
	else:
		st.success("You picked {}".format(options))

		if st.button('Predict'):
			result = predict_dt(sex, age, n_siblings_spouses, parch, fare, Class, alone)
			if result[0] == 0:
				st.success('Not a Survivor')
			else:
				st.error('A Survivor')
	
# Links and Final Touches
	html_git = """
	<h3>Checkout my GitHub</h3>
	<div style ="background-color:black;padding:13px">
	<h1 style ="color:white;text-align:center;"><a href="https://github.com/Taoheed-O"> My GitHub link</h1>
	</div>
	"""
	html_linkedIn = """
	<h3>Connect with me on LinkedIn</h3>
	<div style ="background-color:black;padding:13px">
	<h1 style ="color:white;text-align:center;"><a href="https://www.linkedin.com/in/taoheed-oyeniyi"> My LinkedIn</h1>
	</div>
	"""
	
	# this line allows us to display the front end aspects we have
	# defined in the above code
	st.markdown(html_git, unsafe_allow_html = True)
	st.markdown(html_linkedIn, unsafe_allow_html = True)

			
        
	
if __name__=='__main__':
	main()
