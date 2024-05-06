# Core Package
import streamlit as st 

# Load EDA Packages
import pandas as pd 

# Load Data Visualization Package
#import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('Agg')
import seaborn as sns 
import plotly.express as px 


# Load Dataset
#@st.cache_data
def load_data(data):
    df = pd.read_csv(data)
    return df


def run_eda_app():
    st.subheader("Exploratory Data Analysis")
    #df = pd.read_csv("C:/Users/arya.hisma/Downloads/Small Project/07. Machine Learning/data/diabetes_data_upload.csv")
    df = load_data("data/diabetes_data_upload.csv")
    # st.write(df)
    df_encode = load_data("data/diabetes_data_upload_clean.csv")
    freq_df = load_data("data/freqdist_of_age_data.csv")
    
    submenu = ["Descriptive", "Plots"]
    choice_submenu = st.sidebar.selectbox("Submenu", submenu)
    
    
    if choice_submenu == "Descriptive":
        st.subheader("descriptive Analysis")
        st.dataframe(df)
        
        
        with st.expander("Descriptive Summary"):
            st.dataframe(df_encode.describe())
            
            
        # Layout
        col1, col2, col3 = st.columns(3)
        
            
        with col1:
            with st.expander("Class Distribution"):
                st.dataframe(df['class'].value_counts())
        
            
        with col2:
            with st.expander("Gender Distribution"):
                st.dataframe(df['Gender'].value_counts())
        
                
        with col3:
            with st.expander("Data Types"):
                st.dataframe(df.dtypes)
    
    
    elif choice_submenu == "Plots":
        st.subheader("Plot")
        
        
        # Layout
        c1, c2 = st.columns([2,1])
        
        with c1:
            # Gender Distribution
            with st.expander("Dist Plot of Gender"):
                # Using Seaborn
                #fig = plt.figure()
                #sns.countplot(df['Gender'])
                #st.pyplot(fig)
                
                gen_df = df['Gender'].value_counts().to_frame()
                gen_df = gen_df.reset_index()
                gen_df.columns = ["Gender Type", "Counts"]
                st.dataframe(gen_df)
                
                p1 = px.pie(gen_df, names='Gender Type', values='Counts')
                st.plotly_chart(p1, use_container_width=True)
                
                
        # For Class Distribution
            with st.expander("Dist Plot of Class"):
                #fig = plt.figure()
                #sns.countplot(df['class'])
                #st.pyplot(fig, use_container_width=True)
                
                p2 = px.box(df, x='class')
                st.plotly_chart(p2, use_container_width=True)
                
                
        with c2:
            with st.expander("Gender Distribusion"):
                st.dataframe(gen_df, use_container_width=True)
                 
            with st.expander("Class Distribution"):
                st.dataframe(df['class'].value_counts())
                
                
        # Freq Distribution
        with st.expander("Frequency Dist of Age"):
            st.dataframe(freq_df)
            p3 = px.bar(freq_df, x="Age", y="count")
            st.plotly_chart(p3, use_container_width=True)


        # Outlier Detection
        with st.expander("Outlier Detection Plot"):
            #fig = plt.figure()
            #sns.boxplot(df['Age'])
            #st.pyplot(fig, use_container_width=True)
            
            p4 = px.box(df, x='Age', color='Gender')
            st.plotly_chart(p4, use_container_width=True)


        # Correlation
        with st.expander("Correlation Plot"):
            corr_matrix = df_encode.corr()
            #fig = plt.figure(figsize=(20, 10))
            #sns.heatmap(corr_matrix, annot=True)
            #st.pyplot(fig, use_container_width=True)
            
            p5 = px.imshow(corr_matrix)
            st.plotly_chart(p5, use_container_width=True)
        
    
    else:
        pass

