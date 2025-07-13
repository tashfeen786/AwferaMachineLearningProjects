import streamlit as st 
import numpy as np 
import pandas as pd 
# Adding the title of app
st.title("My First Streamlit App with python ")
# Adding the text
st.write("Hello, World!")

# Add user input
number = st.slider('Pick a number', 0, 100)

# print the text of number 
st.write(f'you selected: {number}')
# Adding button
if st.button('Greeting'):
    st.write('Hi! hello there')
else:
    st.write('Goodbye')

#Add Radio Button 
genra = st.radio('What is your favorite genre', ('Action', 'Comedy', 'Drama'))
 
# Add a drop down list 
option = st.selectbo(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'mobile phone')
)

#add your whatsapp number 
st.sidebar.text_input('Enter your whatsapp number: ')

# add a file uploader
upload_file = st.sidebar.file_uploader('Choose a csv file ', type = "cssv")
# create a line plot 
# Plotting 
data = pd.DataFrame(
    {
        'first coulmn': list(range(1,11)),
        'second column': list(rang(number, number+10))
    }
)
st.line_chart(data)
