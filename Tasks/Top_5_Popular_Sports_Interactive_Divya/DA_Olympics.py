#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd 
import plotly.offline as plt
import plotly.graph_objs as go
import numpy as np
import dash         
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly         
import plotly.express as px


# In[39]:


athlete_events_df = pd.read_csv("athlete_events.csv")


# In[40]:


print("Athletes and Events data -  rows:", athlete_events_df.shape[0]," columns:", athlete_events_df.shape[1])


# In[41]:


athlete_events_df.head(10)


# In[42]:


data = athlete_events_df


# In[43]:


def highest_participants():
    participants = data.groupby(by=['Year', 'Sport']).size().groupby(level=0).nlargest(5).droplevel(0).to_frame().reset_index()
    years = ['Year ' + str(yr) for yr in participants['Year'].unique()]

    participants = participants.groupby(by='Year')

    colors = ['#004D40', '#00897B', '#4DB6AC', '#B2DFDB', '#E0F2F1']

    fig = go.Figure(
        [go.Barpolar(r=participants.nth(i)[0], name='', text=participants.nth(i)['Sport'], marker_color=colors[i],
                     theta=years)
         for i in range(4, -1, -1)],
        go.Layout(height=1000, title='Top 5 popular sports in Olympic History',
                  polar_bgcolor='#212121', paper_bgcolor='#212121',
                  font_size=15, font_color='#FFFFFF',
                  polar=dict(radialaxis=dict(visible=False)))
    )

    plt.plot(fig, 'div.html')


# In[44]:


highest_participants()


# In[46]:


# modules for generating the word cloud
from os import path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator
fields = ['Sport']
text2 = pd.read_csv('athlete_events.csv', usecols=fields)
text3 = ' '.join(text2['Sport'])
text3=text3.split()
text3=set(text3)
text3=list(text3)
text3=" ".join(text3)
mask2 = np.array(Image.open('newsol.png'))
wc = WordCloud(background_color="aqua", max_words=2000, mask=mask2,
               max_font_size=90, random_state=35)
wc.generate(text3)
# create coloring from image
image_colors = ImageColorGenerator(mask2)
plt.figure(figsize=[70,70])
plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
_=plt.show()


# In[ ]:




