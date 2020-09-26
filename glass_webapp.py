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

    ri=st.number_input('Select % of Ri', min_value=0.0, max_value=100.0,step=0.1)
    na=st.number_input('Select % of Na', min_value=0.0, max_value=100.0,step=0.1)
    mg=st.number_input('Select % of Mg', min_value=0.0, max_value=100.0,step=0.1)
    al=st.number_input('Select % of Al', min_value=0.0, max_value=100.0,step=0.1)
    si=st.number_input('Select % of Si', min_value=0.0, max_value=100.0,step=0.1)
    k =st.number_input('Select % of K',  min_value=0.0, max_value=100.0,step=0.1)
    ca=st.number_input('Select % of Ca', min_value=0.0, max_value=100.0,step=0.1)
    ba=st.number_input('Select % of Ba', min_value=0.0, max_value=100.0,step=0.1)
    fe=st.number_input('Select % of Fe', min_value=0.0, max_value=100.0,step=0.1)
    
    inputs=[[ri,na,mg,al,si,k,ca,ba,fe]]
    summation = ri+na+mg+al+si+k+ca+ba+fe
    
    if st.button('Classify'):
        if summation == 100:
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
        else:
            st.warning('Values must add up to 100 ! Try again.')

if __name__=='__main__':
    main()
