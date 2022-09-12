import joblib
import pandas as pd
import streamlit as st


# loading in the model to predict on the data
scaler = joblib.load(r'scaler.pickle')

# loading Logistic Regression model
classifier_lr = joblib.load(r'model_lr.pickle')

# Loading Decision Tree model
classifier_dt = joblib.load(r'model_dt.pickle')


# the font and background color, the padding and the text to be displayed
html_temp = """
	<div style ="background-color:black;padding:13px">
	<h1 style ="color:white;text-align:center;">Titanic Survivors Prediction App</h1>
	</div>
	"""
# this line allows us to display the front end aspects we have
# defined in the above code
st.markdown(html_temp, unsafe_allow_html = True)
# Image
st.image("https://pngimg.com/uploads/titanic/titanic_PNG36.png")

# giving the webpage a title
st.title("Machine Learning [ Classification ]")

# WElcome Function
def welcome():
	return 'welcome all'

# Features and labels
features = ['sex_female', 'n_siblings_spouses_8', 'n_siblings_spouses_1',
    'parch_6', 'n_siblings_spouses_4', 'parch_0', 'parch_5', 'n_siblings_spouses_0', 'parch_3',
    'sex_male', 'Class_First', 'parch_2', 'alone_y', 'n_siblings_spouses_5', 'n_siblings_spouses_2',
    'n_siblings_spouses_3', 'Class_Second', 'parch_1', 'alone_n', 'Class_Third', 'parch_4']
labels = ['sex', 'age', 'n_siblings_spouses', 'parch', 'fare', 'Class', 'alone']

# defining the function which will make the prediction{Logistic regression}using the user inputs
def predict_lr(sex, age, n_siblings_spouses, parch, fare, Class, alone):
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
st.write("Male / Female")
sex = st.radio("Select gender", ('male', 'female'))

# Age
age = st.number_input("What is the age ?")

# Spouses and siblings
st.write("Number of spouses & siblings.")
n_siblings_spouses = st.slider("Select the number of siblings or spouses", 0,5)

# Parch
st.write("Parch number ")
parch = st.slider("Select parch number", 0, 6)

# Fare
st.write("Fare")
fare = st.number_input("Thousand Dollars($)")

# Class 
st.write("First/Second/Third")
Class = st.radio("Select Class", ('First', 'Second', 'Third')) 

# Alone
passenger_status = st.radio("Is the passenger alone ?", ('yes', 'no'))
#conditionals for alone status
if (passenger_status) == 'yes':
	alone = 'y'
else:
	alone = 'n'



# this is the main function in which is defined on the webpage
def main():
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
				st.error('Not a Survivor')
			else:
				st.success('A Survivor')
	else:
		st.success("You picked {}".format(options))

		if st.button('Predict'):
			result = predict_dt(sex, age, n_siblings_spouses, parch, fare, Class, alone)
			if result[0] == 0:
				st.error('Not a Survivor')
			else:
				st.success('A Survivor')
	
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
