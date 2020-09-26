import streamlit as st
import pickle

model_logis=pickle.load(open('logis_model.pkl','rb'))
model_dtree=pickle.load(open('dtree_model.pkl','rb'))
model_forrest=pickle.load(open('forrest_model.pkl','rb'))
model_svc=pickle.load(open('svc_model.pkl','rb'))
model_knc=pickle.load(open('knc_model.pkl','rb'))


def classify(prediction):
    if prediction == 0:
        return 'BWNF'
    elif prediction == 1:
        return 'BWF'
    elif prediction == 2:
        return 'Headlamps'
    elif prediction == 3:
        return 'VWF'
    elif prediction == 4:
        return 'Containers'
    else: 
        return 'Tableware'
    
def main():
    st.title("Glass Classifier")
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Glass Classification</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    activities=['Logistic Regression','Decision Tree','Random Forrest','Support Vector','K-Neighbours']
    option=st.sidebar.selectbox('Which model would you like to use?',activities)
    st.subheader(option)

    ri=st.slider('Select % of Ri', 0.0, 100.0)
    na=st.slider('Select % of Na', 0.0, 100.0)
    mg=st.slider('Select % of Mg', 0.0, 100.0)
    al=st.slider('Select % of Al', 0.0, 100.0)
    si=st.slider('Select % of Si', 0.0, 100.0)
    k =st.slider('Select % of K', 0.0, 100.0)
    ca=st.slider('Select % of Ca', 0.0, 100.0)
    ba=st.slider('Select % of Ba', 0.0, 100.0)
    fe=st.slider('Select % of Fe', 0.0, 100.0)
    
    inputs=[[ri,na,mg,al,si,k,ca,ba,fe]]
    
    if st.button('Classify'):
        if option=='Logistic Regression':
            st.success(classify(model_logis.predict(inputs)[0]))
        elif option=='Decision Tree':
            st.success(classify(model_dtree.predict(inputs)[0]))
        elif option=='Random Forrest':
            st.success(classify(model_forrest.predict(inputs)[0]))
        elif option=='Support Vector':
            st.success(classify(model_svc.predict(inputs)[0]))
        else:
            st.success(classify(model_knc.predict(inputs)[0]))

if __name__=='__main__':
    main()
