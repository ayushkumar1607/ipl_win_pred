import streamlit as st
import pickle
import pandas as pd

st.title("IPL WIN PREDICTOR")

teams=[
        'Chennai Super Kings',
        'Delhi Capitals',
        'Kings XI Punjab',
        'Kolkata Knight Riders',
        'Mumbai Indians',
        'Rajasthan Royals',
        'Royal Challengers Bangalore',
        'Sunrisers Hyderabad'
    ]

cities=['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Dharamsala', 'Pune', 'Raipur',
       'Ranchi', 'Abu Dhabi', 'Sharjah', 'Cuttack', 'Visakhapatnam',
       'Mohali', 'Bengaluru']
pipe=pickle.load(open('pipe.pkl','rb'))
col1,col2=st.columns(2)

with col1:
    batting_team=st.selectbox("Select Batting Team" ,sorted(teams))
with col2:
    bowling_team=st.selectbox("Select Bowling Team" ,sorted(teams))

selected_city=st.selectbox("Select City" ,sorted(cities))

target=st.number_input("Select Target",step=1)

col3,col4,col5=st.columns(3)
with col3:
    score=st.number_input("Score",step=1)
with col4:
    overs=st.number_input("Overs_completed",step=1)
with col5:
    wickets=st.number_input("Wickets out",step=1)

if st.button("Predict Probability"):
    runs_left=target-score
    balls_left=120-(overs*6)
    wickets=10-wickets
    crr=score/overs
    rrr=(runs_left*6)/balls_left
    #we have to give data in form of dataframe
    input_df=pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],
                           'city':[selected_city],'runs_left':[runs_left],'balls_left':[balls_left],
                           'wickets':[wickets],'total_runs_x':[target],'crr':[crr],'rrr':[rrr]})

    result=pipe.predict_proba(input_df)
    loss=result[0][0]
    win=result[0][1]
    st.header(batting_team +"-"+ str(round(win*100))+"%")
    st.header(bowling_team + "-" + str(round(loss*100)) + "%")

