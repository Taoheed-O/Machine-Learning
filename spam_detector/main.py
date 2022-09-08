import pickle
import streamlit as st


# loading in the model to predict on the data

vectorizer_in = open('vectorizer.pkl', 'rb')
vectorizer = pickle.load(vectorizer_in)
pickle_in = open("model_log.pkl", "rb")
classifier = pickle.load(pickle_in)

# Image
st.image("https://media.istockphoto.com/photos/phishing-scam-email-identity-alert-3d-rendering-picture-id1046171248")
def welcome():
	return 'welcome all'

# defining the function which will make the prediction using
# the data(text) which the user inputs
def prediction(text):
    vector_text = vectorizer.transform([text]).toarray()
    prediction = classifier.predict(vector_text)
    print(prediction)
    return(prediction)

# this is the main function in which is defined on the webpage
def main():
	# giving the webpage a title
	st.title("Spam E-mail Detector")
	
	# the font and background color, the padding and the text to be displayed
	html_temp = """
	<div style ="background-color:black;padding:13px">
	<h1 style ="color:white;text-align:center;">Spam Detector App</h1>
	</div>
	"""
	
	# this line allows us to display the front end aspects we have
	# defined in the above code
	st.markdown(html_temp, unsafe_allow_html = True)
	
	#List of available models 
	options = st.radio("Available Models:", ["Logistic Regression", "Multinomial Naive Bayes","Decision Tree"])
	result =""

	# the below line ensures that when the button called 'Predict' is clicked,
	# the prediction function defined above is called to make the prediction
	# and store it in the variable result
	if options == "Logistic Regression":
		st.success("You picked {}".format(options))
		# the following lines create text boxes in which the user can enter
		# the data required to make the prediction
		text = st.text_input("Review:", "Type your review here")
	
		if st.button('Predict'):
			result = prediction(text)
			if result == 0:
				st.error('This is not a spam mail/sms.'.format(result))
			else:
				st.success('This is a spam mail/sms.'.format(result))
	else:
		st.warning('This model is under development and not available for predicting yet.'.format(result))
		pass
	
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
