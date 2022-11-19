import pandas as pd
from path_vars import base 

#  Match scores to a subject for time
# features in df, scores in df2   
def matchScores(score_type, df, df2):
  df['score_'+score_type[0:2]] = 0.0
  for j, row in df.iterrows():
    search = row['subject']
    mov = row['move'].split('_')[0].lower()
    moves = ['Raising the Power', 'Push', 'Grasp the Sparrow\'s Tail', 'Wave Hands Like Clouds', 'Brush Knee and Twist Step', 'Golden Rooster Stands on One Leg']
    ind = None
    for i in range(len(moves)):
      if mov in moves[i].lower():
        ind = i 
        break
    col_name = '0' + search[len(search) - 2 :]
    score = float(df2['Subject ' + col_name].loc[(df2['Movement'] == moves[ind]) & (df2['Scoring Element'] == score_type)].item())
    # iterrows returns copies, so need to change reference
    df.at[j, 'score_'+score_type[0:2]] = score
  return df

features = pd.read_csv(base + 'RTP_FeaturesV3.csv')
scores = pd.read_excel(base + 'Scores.xlsx')
data = matchScores('Gross Competency', features, scores)
data = matchScores('Alignment/Posture', data, scores)
data = matchScores('Flow/Integration', data, scores)
data.to_csv(base + 'RTP_FeaturesV3.csv')