import pickle as pkl
from flask import Flask, render_template, request, url_for, redirect
import numpy as np
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict')
def predict():
    team1 = str(request.args.get('team1'))
    team2 = str(request.args.get('team2'))

    with open('TeamsCLpkl.pkl', 'rb') as f:
        home = pkl.load(f)

        with open('TeamsCLaway.pkl', 'rb') as f:
            away = pkl.load(f)

        home_team_history_rating = home.iloc[0]["home_team_history_rating"]
        away_team_history_rating = away.iloc[0]["away_team_history_rating"]
        home_team_history_opponent_rating = home.iloc[0]["home_team_history_opponent_rating"]
        away_team_history_opponent_rating = away.iloc[0]["away_team_history_opponent_rating"]

    if team1 == team2:
        return redirect(url_for('index'))
    print(team1)
    print(team1, team2, home_team_history_rating, away_team_history_rating, home_team_history_opponent_rating,
          away_team_history_opponent_rating)

    with open('model.pkl', 'rb') as f:
        model = pkl.load(f)
    arr = np.array([home_team_history_rating, away_team_history_rating, home_team_history_opponent_rating, away_team_history_opponent_rating]).reshape(1,-1)
    print(type(arr))

    predict=model.predict_proba(arr)
    print(predict[0, 0])
    return render_template('after.html', data=predict, gamehome=team1, gameaway=team2)


if __name__ == "__main__":
    app.run(debug=True)