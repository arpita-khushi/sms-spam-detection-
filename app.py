import streamlit as st
import pickle
model=pickle.load(open('spam.pkl','rb'))
cv=pickle.load(open('vectorizer.pkl','rb'))
st.title("EMAIL SPAM CLASSIFICATIONS")
st.write("This is a ml project")
user_input=st.text_area("Enter an email to classift", height=150)

if st.button("Classify"):
    if user_input:
        data=[user_input]
        vectorized_data=cv.transform(data).toarray()
        result=model.predict(vectorized_data)
        if result[0]==0:
            st.write("Email is not spam")
        else:
            st.write("Email is spam")
    else:
        st.write("Please type email to classify")
