
# Importing Dependencies
import os
import glob
import streamlit as st
from easyocr import Reader
import pickle

def main():
  """Main function, Default calls to home page"""
  st.title("Language Detector Application")
  menu = ["Home","About"]
  choice = st.sidebar.selectbox("Menu",menu)

  if choice == "Home":
    st.subheader("Detect what language is in a text or on an image.")
    st.markdown("<p style='font-size:12px;'></p>\n<b>Select an option to begin.</b>", unsafe_allow_html=True)

    options = st.radio(
    "",
    ('Text', 'Image'))

    # Image to text model
    reader_lang = Reader(['en', 'fr'])
    # Language recognizer model
    model = pickle.load(open('LD.model','rb'))
    CV = pickle.load(open('vectorizer.pickle','rb'))
    encoder = pickle.load(open('encoder.pickle','rb'))

    # function for predicting language
    def predict(text):
        x = CV.transform([text]).toarray()
        lang = model.predict(x)
        lang = encoder.inverse_transform(lang)
    
        st.write('The language in the above text is', lang[0])

    if options == 'Text':
        text = st.text_area('Text to analyze','''''')

        if st.button(label="Detect"):
            predict(text)
        
    else:
        # Upload Image
        uploaded_file = st.file_uploader("Upload a file", type=['.png','.jpg'])
        if uploaded_file is not None:
            # To read file as bytes:
            with open(os.path.join("/home/davidoden/Documents/Language-Detector-App/Images/",uploaded_file.name),"wb") as f: 
                f.write(uploaded_file.getbuffer())
            
            #image path
            path = "/home/davidoden/Documents/Language-Detector-App/Images/"+uploaded_file.name
             # Read the data
            text = reader_lang.readtext(path, detail = 0, paragraph=True)
            
            if st.button(label="Detect"):
                predict(text)
            


if __name__ == '__main__':
    main()
