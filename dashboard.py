
import os
import time
import numpy as np
import pandas as pd
import altair as alt
import networkx as nx
import nx_altair as nxa
import streamlit as st

from utils import cal_distance, node_positions, bike_network


st.write("# New York’s Citi Bikes Flow Analysis")

st.write("## Yearly data")


def load_month(year=2021, month_index=1):
    month_index = str(month_index)
    month_index = '0' + month_index if len(month_index) == 1 else month_index
    filename = f'./data/JC-{year}{month_index}-citibike-tripdata.csv'
    if os.path.isfile(filename):
        month_df = pd.read_csv(filename)
        if 'started_at' in month_df.columns:
            month_df = month_df[['started_at', 'ended_at',
                                 'start_station_name', 'start_station_id', 'end_station_name',
                                 'end_station_id', 'start_lat', 'start_lng', 'end_lat', 'end_lng']]
        elif '###starttime' in month_df.columns:
            month_df = pd.DataFrame({
                'started_at': month_df["starttime"],
                'ended_at': month_df["stoptime"],
                'start_station_name': month_df["start station name"],
                'start_station_id': month_df["start station id"],
                'end_station_name': month_df["end station name"],
                'end_station_id': month_df["end station id"],
                'start_lat': month_df["start station latitude"],
                'start_lng': month_df["start station longitude"],
                'end_lat': month_df["end station latitude"],
                'end_lng': month_df["end station longitude"],
            })
        return month_df
    # else:
        # print(f"Month {year}-{month} missing.")


def load_year(year=2021):
    dataframes = []
    for i in range(1, 13):
        month_df = load_month(year, i)
        if month_df is not None and 'started_at' in month_df.columns:
            dataframes.append(month_df)
    return pd.concat(dataframes)


@st.cache(allow_output_mutation=True)
def load_data(year=2021, month_index=None):
    if month_index is None:
        df = load_year(year)
    else:
        df = load_month(year, month_index)

    # Preprocessing
    df['started_at'] = pd.to_datetime(df['started_at'])
    df['ended_at'] = pd.to_datetime(df['ended_at'])
    df.dropna(axis=0, inplace=True)

    # Remove invalid stations
    df = df.loc[df.start_station_id.str.contains('[a-zA-Z]')]
    df = df.loc[df.end_station_id.str.contains('[a-zA-Z]')]
    df = df.loc[~df.end_station_id.str.startswith('SYS')]

    # Compute distance and travel time
    df['distance'] = cal_distance(df.start_lng, df.end_lng, df.start_lat, df.end_lat)
    df['diff_time'] = (df['ended_at'] - df['started_at']).values / np.timedelta64(1, 'h') * 60

    return df, node_positions(df)


year_sel = st.slider("Year", 2021, 2022, 2021)

data, node_position_df = load_data(year_sel, None)
possible_months = set(data.started_at.dt.month.to_list())

# st.write(data)
# st.write(node_position_df)


@st.cache(allow_output_mutation=True)
def plot_traffic_graph(g):

    chart = nxa.draw_networkx(
        G=g,
        pos=g.nodes.data('pos'),
        node_color='area',
        edge_color='black',
    )

    edges = chart.layer[0]
    nodes = chart.layer[1]

    brush = alt.selection_interval()
    color = alt.Color('area:N', legend=None)

    edges = edges.encode(
        opacity=alt.value(100 / len(g.edges)),
    )

    nodes = nodes.encode(
        opacity=alt.value(0.8),
        fill=alt.condition(brush, color, alt.value('gray')),
        size='traffic:Q',
        tooltip=[
            alt.Tooltip('name', title='Node name'),
            alt.Tooltip('lng', title='Longitude'),
            alt.Tooltip('lat', title='Latitude'),
            alt.Tooltip('traffic', title='Traffic')
        ]
    ).add_selection(
        brush
    )

    en_chart = (edges + nodes).properties(width=600)

    # Create a bar graph to show highlighted nodes.
    bars = alt.Chart(nodes.data).mark_bar().encode(
        x=alt.X('sum(traffic)', title='Total traffic'),
        y='area:N',
        color='area:N',
        tooltip=[
            alt.Tooltip('area', title='Area'),
            alt.Tooltip('count()', title='Number of nodes'),
            alt.Tooltip('sum(traffic)', title='Total traffic'),
        ]
    ).transform_filter(
        brush
    ).properties(width=600)

    return alt.vconcat(en_chart, bars)


year_graph = bike_network(data, node_position_df=node_position_df)
st.altair_chart(plot_traffic_graph(year_graph))


st.write("## Monthly data (dynamic graph)")


@st.cache(allow_output_mutation=True)
def plot_graph(g, color_strategy="PageRank"):

    if color_strategy == "PageRank":
        node_colors = nx.algorithms.link_analysis.pagerank_alg.pagerank(g, weight="diff_time")
    elif color_strategy == "Betweenness Centrality":
        node_colors = nx.betweenness_centrality(g)
    elif color_strategy == "Closeness Centrality":
        node_colors = nx.closeness_centrality(g)
    elif color_strategy == "Eigenvector":
        node_colors = nx.eigenvector_centrality(g, max_iter=6000, weight="duration")
    else:
        node_colors = {n: 1.0 for n in g.nodes}
    for node in node_colors.keys():
        g.nodes[node]['color'] = node_colors[node]

    chart = nxa.draw_networkx(
        G=g,
        pos=g.nodes.data('pos'),
        node_color='color',
        edge_color='black',
        cmap='viridis',
    )

    edges = chart.layer[0]
    nodes = chart.layer[1]

    nodes = nodes.encode(
        x=alt.X('lng_scaled:Q', scale=alt.Scale(domain=(-2.7, 1.5))),
        y=alt.Y('lat_scaled:Q', scale=alt.Scale(domain=(-2.0, 2.0))),
        opacity=alt.value(0.8),
        size='traffic:Q',
        tooltip=[
            alt.Tooltip('name', title='Node name'),
            alt.Tooltip('lng', title='Longitude'),
            alt.Tooltip('lat', title='Latitude'),
            alt.Tooltip('traffic', title='Traffic'),
            alt.Tooltip('color', title=color_strategy)
        ]
    )

    edges = edges.encode(
        opacity=alt.value(100 / len(g.edges)),
    )

    return (edges + nodes).properties(width=800, height=500)


graphs = {}
for month in possible_months:
    graphs[month] = bike_network(data[data.started_at.dt.month == month],
                                 node_position_df=node_position_df)

possible_color_strategies = ["PageRank", "Betweenness Centrality", "Closeness Centrality", "Eigenvector"]
color_strategy_sel = st.sidebar.selectbox("Coloring strategy", possible_color_strategies, 0)

if len(possible_months) > 1:
    month_sel = st.select_slider("Month", possible_months)
else:
    month_sel = min(possible_months)

month_names = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
               7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}
text = st.empty()
text.markdown("Month: {}".format(month_names[month_sel]))

# new_chart = st.altair_chart(charts[month_sel])
new_chart = st.altair_chart(plot_graph(graphs[month_sel], color_strategy=color_strategy_sel))

# Add start and stop button and animate the chart over the months
col1, col2 = st.columns(2)

with col1:
    start = st.button("Start")
with col2:
    stop = st.button("Stop")

if start:
    for month in possible_months:
        text.markdown("Month: {}".format(month_names[month]))
        st.session_state['month'] = month
        new_chart.altair_chart(plot_graph(graphs[month], color_strategy=color_strategy_sel))
        time.sleep(1.0)

if stop and 'month' in st.session_state:
    new_chart.altair_chart(plot_graph(graphs[st.session_state['month']], color_strategy=color_strategy_sel))
