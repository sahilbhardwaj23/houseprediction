import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Plotting Demo")

st.title('Analytics')

new_df = pd.read_csv('datasets/data_viz1.csv')
feature_text = pickle.load(open('datasets/feature_text.pkl','rb'))


# group_df = new_df.groupby('sector').mean()[['price','price_per_sqft','built_up_area','latitude','longitude']]
#
# st.header('Sector Price per Sqft Geomap')
# fig = px.scatter_mapbox(group_df, lat="latitude", lon="longitude", color="price_per_sqft", size='built_up_area',
#                   color_continuous_scale=px.colors.cyclical.IceFire, zoom=10,
#                   mapbox_style="open-street-map",width=1200,height=700,hover_name=group_df.index)
#
# st.plotly_chart(fig,use_container_width=True)
# st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_option('deprecation.showPyplotGlobalUse', False)
st.header('Features Wordcloud')
sector = st.selectbox('Select Sector ',['gwal pahari','manesar','sector 1','sector 10',
 'sector 102',
 'sector 103',
 'sector 104',
 'sector 105',
 'sector 106',
 'sector 107',
 'sector 108',
 'sector 109',
 'sector 11',
 'sector 110',
 'sector 111',
 'sector 112',
 'sector 113',
 'sector 12',
 'sector 13',
 'sector 14',
 'sector 15',
 'sector 17',
 'sector 2',
 'sector 21',
 'sector 22',
 'sector 23',
 'sector 24',
 'sector 25',
 'sector 26',
 'sector 27',
 'sector 28',
 'sector 3',
 'sector 30',
 'sector 31',
 'sector 33',
 'sector 36',
 'sector 37',
 'sector 37d',
 'sector 38',
 'sector 39',
 'sector 4',
 'sector 40',
 'sector 41',
 'sector 43','dwarka expressway', 'sector 70a', 'sohna road'
 'sector 45',
 'sector 46',
 'sector 47',
 'sector 48',
 'sector 49',
 'sector 5',
 'sector 50',
 'sector 51',
 'sector 52',
 'sector 53',
 'sector 54',
 'sector 55',
 'sector 56',
 'sector 57',
 'sector 58',
 'sector 59',
 'sector 6',
 'sector 60',
 'sector 61',
 'sector 62',
 'sector 63',
 'sector 63a',
 'sector 65',
 'sector 66',
 'sector 67',
 'sector 67a',
 'sector 68',
 'sector 69',
 'sector 7',
 'sector 70',
 'sector 71',
 'sector 72',
 'sector 73',
 'sector 74',
 'sector 76',
 'sector 77',
 'sector 78',
 'sector 79',
 'sector 8',
 'sector 80',
 'sector 81',
 'sector 82',
 'sector 82a',
 'sector 83',
 'sector 84',
 'sector 85',
 'sector 86',
 'sector 88',
 'sector 88a',
 'sector 89',
 'sector 9',
 'sector 90',
 'sector 91',
 'sector 92',
 'sector 93',
 'sector 95',
 'sector 99'])
wordcloud = WordCloud(width = 800, height = 800,
                      background_color ='black',
                      stopwords = set(['s']),  # Any stopwords you'd like to exclude
                      min_font_size = 10).generate(feature_text)

plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad = 0)
st.pyplot()


# Load the CSV file containing longitude and latitude coordinates
df = pd.read_csv('gurgaon_sectors_coordinates.csv')

st.header('Gurugram Map')

# Select sector from dropdown
Map = st.selectbox('Select Sector', ['Overall','gwal pahari','manesar','sector 1','sector 10',
                                     'sector 102', 'sector 103', 'sector 104', 'sector 105', 'sector 106',
                                     'sector 107', 'sector 108', 'sector 109', 'sector 11', 'sector 110',
                                     'sector 111', 'sector 112', 'sector 113', 'sector 12', 'sector 13',
                                     'sector 14', 'sector 15', 'sector 17', 'sector 2', 'sector 21',
                                     'sector 22', 'sector 23', 'sector 24', 'sector 25', 'sector 26',
                                     'sector 27', 'sector 28', 'sector 3', 'sector 30', 'sector 31',
                                     'sector 33', 'sector 36', 'sector 37', 'sector 37d', 'sector 38',
                                     'sector 39', 'sector 4', 'sector 40', 'sector 41', 'sector 43',
                                     'dwarka expressway', 'sector 70a', 'sohna road', 'sector 45', 'sector 46',
                                     'sector 47', 'sector 48', 'sector 49', 'sector 5', 'sector 50',
                                     'sector 51', 'sector 52', 'sector 53', 'sector 54', 'sector 55',
                                     'sector 56', 'sector 57', 'sector 58', 'sector 59', 'sector 6',
                                     'sector 60', 'sector 61', 'sector 62', 'sector 63', 'sector 63a',
                                     'sector 65', 'sector 66', 'sector 67', 'sector 67a', 'sector 68',
                                     'sector 69', 'sector 7', 'sector 70', 'sector 71', 'sector 72',
                                     'sector 73', 'sector 74', 'sector 76', 'sector 77', 'sector 78',
                                     'sector 79', 'sector 8', 'sector 80', 'sector 81', 'sector 82',
                                     'sector 82a', 'sector 83', 'sector 84', 'sector 85', 'sector 86',
                                     'sector 88', 'sector 88a', 'sector 89', 'sector 9', 'sector 90',
                                     'sector 91', 'sector 92', 'sector 93', 'sector 95', 'sector 99'])

if Map == 'Overall':
    # Create a scatter plot of all coordinates on a map
    fig = px.scatter_mapbox(df, lat='Lat', lon='Long',
                            mapbox_style='open-street-map', zoom=11,
                            text='Sector')
    fig.update_layout(mapbox_center={"lat": 28.44204, "lon": 77.02065},
                      height=850,  # Set the height of the graph
                      width=850)
    st.plotly_chart(fig)
else:
    filtered_df = df[df['Sector'] == Map]
    if not filtered_df.empty:
        # Get the coordinates of the selected sector
        sector_lat = filtered_df['Lat'].iloc[0]
        sector_lon = filtered_df['Long'].iloc[0]

        # Create a scatter plot of the coordinates on a map for the specified sector
        fig = px.scatter_mapbox(filtered_df, lat='Lat', lon='Long',
                                mapbox_style='open-street-map', zoom=10,
                                text='Sector')  # Add 'text' parameter to show sector name

        # Update layout to focus on the selected sector
        fig.update_layout(mapbox_center={"lat": sector_lat, "lon": sector_lon},
                          height=400,  # Set the height of the graph
                          width=400)
        st.plotly_chart(fig)
    else:
        st.write(f"No data found for sector '{Map}'. Please select a valid sector.")





st.header('Area Vs Price')

property_type = st.selectbox('Select Property Type', ['flat','house'])

if property_type == 'house':
    fig1 = px.scatter(new_df[new_df['property_type'] == 'house'], x="built_up_area", y="price", color="bedRoom", title="Area Vs Price")

    st.plotly_chart(fig1, use_container_width=True)
else:
    fig1 = px.scatter(new_df[new_df['property_type'] == 'flat'], x="built_up_area", y="price", color="bedRoom",
                      title="Area Vs Price")

    st.plotly_chart(fig1, use_container_width=True)

st.header('BHK Pie Chart')

sector_options = new_df['sector'].unique().tolist()
sector_options.insert(0,'overall')

selected_sector = st.selectbox('Select Sector', sector_options)

if selected_sector == 'overall':

    fig2 = px.pie(new_df, names='bedRoom')

    st.plotly_chart(fig2, use_container_width=True)
else:

    fig2 = px.pie(new_df[new_df['sector'] == selected_sector], names='bedRoom')

    st.plotly_chart(fig2, use_container_width=True)

st.header('Side by Side BHK price comparison')

fig3 = px.box(new_df[new_df['bedRoom'] <= 4], x='bedRoom', y='price', title='BHK Price Range')

st.plotly_chart(fig3, use_container_width=True)


st.header('Side by Side Distplot for property type')

fig3 = plt.figure(figsize=(10, 4))
sns.distplot(new_df[new_df['property_type'] == 'house']['price'],label='house')
sns.distplot(new_df[new_df['property_type'] == 'flat']['price'], label='flat')
plt.legend()
st.pyplot(fig3)









