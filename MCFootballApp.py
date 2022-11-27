import streamlit as st
import seaborn as sns
from MC_Score_Predictor import MonteCarloMatchSim, buildScoreMatrix
import matplotlib.pyplot as plt

header = st.container()
team_selector = st.container()
stats_selector = st.container()
simulation_engine = st.container()

score_probabilities = st.container()
team_win_probabilities = st.container()

top_three_scores = st.container()


def back_grad(df):
    return df.to_frame().style.background_gradient(cmap='viridis').set_properties(**{'font-size': '10px'})


def ML_scores(score_matrix, MC_Score_tracker):
    ML_score_dict = {}
    remove_max_score = lambda x, max_likelihood_score: x.pop(max_likelihood_score)
    max_likelihood_score = lambda x: max(x, key=x.get)
    max_likelihood_percent = lambda x: max(x.values())

    ML_score = max_likelihood_score(MC_score_tracker)
    for i in range(3):

        if i > 0:
            remove_max_score(MC_score_tracker, ML_score)
            ML_score = max_likelihood_score(MC_score_tracker)
        ML_percent = max_likelihood_percent(MC_score_tracker)
        ML_score_dict[ML_score] = ML_percent/10000

    print(ML_score_dict)
    return ML_score_dict


st.markdown(
    """
    <style>
    .main{
    background-color: #F5F5F5;
    }
    <style>
    """,
    unsafe_allow_html=True
)

prem_teams = ['Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton', 'Chelsea', 'Crystal Palace', 'Everton',
              'Fulham', 'Leicester', 'Leeds', 'Liverpool', 'Manchester City', 'Manchester United', 'Newcastle United',
              'Nottingham Forest', 'Southampton', 'Tottenham', 'West Ham', 'Wolverhampton Wanderers']

with header:
    st.title('Welcome to Monte Carlo Bivariate Poisson Match Predictor !')

with team_selector:
    st.markdown('** - NOTE - first simulation is slow, as we have to connect to the API to update the stats, but subsequent runs are faster! **')
    st.markdown('**Choose the Home Team and Away Team from the dropdown:**')
    home_col, away_col = st.columns(2)
    home_team = home_col.selectbox('Home Team:', options=prem_teams, index=0)
    away_team = away_col.selectbox('Away Team:', options=prem_teams, index=0)

with stats_selector:
    st.markdown('**Choose the way how we calculate avg. goal parameters:**')

    lookback_col, goal_type_col = st.columns(2)

    games_lookback = lookback_col.slider('How Many Games Shall We LookBack?', min_value=2, max_value=6)
    goal_type = goal_type_col.selectbox('Calculate Goal Rate Based on G or xG?', options=['G', 'xG'], index=0)

teams = [home_team, away_team]
glb = games_lookback

display_button = False

if goal_type == 'G':
    use_xg = 'False'
    display_button = True
else:
    use_xg = 'True'
    display_button = True

col1, simulation_engine, col3 = st.columns(3)

with simulation_engine:

    run_button = st.empty()
    with run_button.container():

        submit = False
        if st.button("Press To Run Simulation"):
            submit = True

            if submit:
                run_button.empty()

            home_win_prob, away_win_prob, draw_prob, MC_score_tracker, x, y, HT_GR, AT_GR = MonteCarloMatchSim(teams, 1000000, GamesLookback=int(glb), BaseOnxG=use_xg)

    if submit:
        sim_end_msg = '<p style="font-family:Arial; color:Red; font-size: 14px;">Simulation Completed. Select New Teams, or Change Goal Parameter Settings to Run Again!</p>'
        st.markdown(sim_end_msg, unsafe_allow_html=True)

        with score_probabilities:
            score_matrix = buildScoreMatrix(MC_score_tracker, teams, x, y)

            st.markdown('** Calculated Goal Rate Parameters: **')
            ht_param, at_param = st.columns(2)
            ht_param.markdown('**{}** Goal Rate Param: {}'.format(home_team, round(HT_GR, 3)))
            at_param.markdown('**{}** Goal Rate Param: {}'.format(away_team, round(AT_GR, 3)))


            st.subheader('Overall Score Probabilities: ')

            fig = plt.figure()
            sns.heatmap(fig)
            st.pyplot(fig)

            # sns.heatmap(score_matrix, annot=True, linewidth=.5, cmap='OrRd', ax=ax)
            # st.pyplot(fig)


            # st.dataframe(data=score_matrix.style.background_gradient(cmap ='OrRd'))

            #st.dataframe(score_matrix.apply(back_grad))

            home_win_row = st.container()
            away_win_row = st.container()
            draw_row = st.container()

            home_win_row.markdown('**{}** Win Percentage Likelihood = {} %'.format(home_team, round(home_win_prob, 1)))
            away_win_row.markdown('**{}** Win Percentage Likelihood = {} %'.format(away_team, round(away_win_prob, 1)))
            draw_row.markdown('**Draw** Percentage Likelihood = {} %'.format(round(draw_prob, 1)))

            with top_three_scores:

                most_likely_score, second_likely_score, third_likely_score = st.columns(3)
                ML_score_dict = ML_scores(score_matrix, MC_score_tracker)

                most_likely_score.markdown('**Most** Likely Score - {}: {} %'.format(list(ML_score_dict.keys())[0],
                                                                                   round(list(ML_score_dict.values())[0], 2)))
                second_likely_score.markdown('**2nd** Likeliest Score - {}: {} %'.format(list(ML_score_dict.keys())[1],
                                                                                    round(list(ML_score_dict.values())[1], 2)))
                third_likely_score.markdown('**3rd** Likeliest Score - {}: {} %'.format(list(ML_score_dict.keys())[2],
                                                                                    round(list(ML_score_dict.values())[2], 2)))

