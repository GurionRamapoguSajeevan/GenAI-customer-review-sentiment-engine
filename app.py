import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# --- Configuration and Data Loading ---

st.set_page_config(layout="wide")

try:
    # Load the processed data (must be in the same directory as app.py)
    df = pd.read_csv('processed_reviews.csv')
except FileNotFoundError:
    st.error("Error: 'processed_reviews.csv' not found. Please ensure the processed data is in the same folder as app.py.")
    st.stop()

# Define meaningful theme labels based on LDA top words (You must update these from your notebook output)
theme_labels = {
    0: "Battery and Charging Life", 
    1: "Performance and Speed",
    2: "Design and Build Quality",
    3: "Price and Value",
    4: "Customer Service Issues"
}
df['theme_label'] = df['theme'].map(theme_labels) 
total_reviews = len(df)

# --- Dashboard Header ---

st.title('🤖 GenAI-Powered Customer Review Insights Dashboard - by Gurion.')
st.subheader("""
This dashboard analyzes customer feedback using **AI and GenAI models** (Sentiment, Zero-Shot Classification) and **Classic ML** (Topic Modeling/LDA). It shows Sentiment (Positive/Neutral/Negative), Common Themes (grouped topic from reviews), Pain Points (common complaints) and Suggestions for improvements. 
""")

st.markdown('**-> Use the filters from the Sidebar to explore key business insights derived from the data.**')

st.markdown("---")

# --- Key Metrics Overview ---

st.header('Key Performance Indicators (KPIs)')

# Calculate overall sentiment stats
sentiment_counts = df['sentiment'].value_counts(normalize=True) * 100
positive_perc = sentiment_counts.get('POSITIVE', 0)
negative_perc = sentiment_counts.get('NEGATIVE', 0)
neutral_perc = sentiment_counts.get('NEUTRAL', 0)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Total Reviews Analyzed", value=f"{total_reviews:,}")
with col2:
    st.metric(label="Overall Positive Sentiment", value=f"{positive_perc:.1f}%", delta=f"{neutral_perc:.1f}% Neutral")
with col3:
    # Calculate the percentage of reviews containing an identified Pain Point
    pain_point_count = df[df['pain_point'] != 'no pain'].shape[0]
    pain_point_perc = (pain_point_count / total_reviews) * 100
    st.metric(label="Reviews with Pain Points", value=f"{pain_point_count:,}", delta=f"{pain_point_perc:.1f}%")
with col4:
    # Calculate the percentage of reviews containing an identified Suggestion
    suggestion_count = df[df['suggestion'] != 'no suggestion'].shape[0]
    suggestion_perc = (suggestion_count / total_reviews) * 100
    st.metric(label="Reviews with Suggestions", value=f"{suggestion_count:,}", delta=f"{suggestion_perc:.1f}%")

st.markdown("---")

# --- Sidebar for Interactivity ---

st.sidebar.header('⚙️ Filter Reviews')

# Filters
selected_sentiments = st.sidebar.multiselect(
    'Filter by Sentiment', 
    options=df['sentiment'].unique(), 
    default=df['sentiment'].unique()
)
selected_pains = st.sidebar.multiselect(
    'Filter by Pain Points', 
    options=df['pain_point'].unique(), 
    default=df['pain_point'].unique()
)
selected_themes = st.sidebar.multiselect(
    'Filter by Themes',
    options=df['theme_label'].unique(),
    default=df['theme_label'].unique()
)

# Apply filters
filtered_df = df[
    df['sentiment'].isin(selected_sentiments) & 
    df['pain_point'].isin(selected_pains) &
    df['theme_label'].isin(selected_themes)
]

st.sidebar.info(f"Showing {len(filtered_df)} reviews after applying filters.")

# --- Main Visualizations ---

col_viz_1, col_viz_2 = st.columns([1, 2])

with col_viz_1:
    st.subheader('Sentiment Distribution')
    st.markdown("*(AI-driven classification: How happy are customers?)*")
    
    # Donut Chart using Plotly for better visual appeal
    sentiment_data = filtered_df['sentiment'].value_counts().reset_index()
    sentiment_data.columns = ['Sentiment', 'Count']
    
    # Define colors
    color_map = {'POSITIVE': '#4CAF50', 'NEGATIVE': '#FF5733', 'NEUTRAL': '#FFC300'}
    
    fig1 = px.pie(
        sentiment_data, 
        values='Count', 
        names='Sentiment', 
        title='Sentiment Breakdown (Donut)',
        color='Sentiment',
        color_discrete_map=color_map,
        hole=0.6 # Creates the donut shape
    )
    
    # Customizing the layout for clarity
    fig1.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#000000', width=1)))
    fig1.update_layout(showlegend=True, height=350, margin=dict(t=30, b=0, l=0, r=0))
    st.plotly_chart(fig1, use_container_width=True)

with col_viz_2:
    st.subheader('Top Themes by Review Count')
    st.markdown("*(LDA Topic Model: What are the main topics discussed?)*")
    
    # Bar Chart for Themes
    fig2 = px.bar(
        filtered_df['theme_label'].value_counts().sort_values(ascending=True),
        orientation='h',
        labels={'value': 'Review Count', 'index': 'Theme'},
        title='Volume of Discussion per Theme',
        color_discrete_sequence=['#3498DB'] # Blue
    )
    fig2.update_layout(yaxis={'categoryorder': 'total ascending'}, height=350, margin=dict(t=30, b=0, l=0, r=0))
    st.plotly_chart(fig2, use_container_width=True)


st.markdown("---")

# --- Sentiment Breakdown by Theme ---

st.header('🔬 Sentiment Breakdown by Theme')
st.markdown("This stacked bar chart shows the distribution of **Positive, Negative, and Neutral** sentiments within each topic/theme. Identify themes with disproportionately high negative feedback.")

# Prepare data for stacked bar chart
theme_sentiment_df = filtered_df.groupby(['theme_label', 'sentiment']).size().reset_index(name='Count')

fig3 = px.bar(
    theme_sentiment_df, 
    x='theme_label', 
    y='Count', 
    color='sentiment',
    title='Sentiment Distribution Across Themes',
    color_discrete_map=color_map,
    labels={'theme_label': 'Product Theme', 'Count': 'Number of Reviews'}
)
fig3.update_layout(xaxis={'categoryorder': 'total descending'})
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# --- Pain Points and Suggestions ---

col_pain, col_sug = st.columns(2)

with col_pain:
    st.subheader('Common Pain Points (Zero-Shot)')
    st.markdown("*(AI-extracted complaints ready for action.)*")
    pain_counts = filtered_df[filtered_df['pain_point'] != 'no pain']['pain_point'].value_counts()
    
    fig4 = px.bar(
        pain_counts,
        labels={'value': 'Count', 'index': 'Pain Point'},
        color_discrete_sequence=['#FFC300'] # Yellow/Orange
    )
    st.plotly_chart(fig4, use_container_width=True)

with col_sug:
    st.subheader('Improvement Suggestions (Zero-Shot)')
    st.markdown("*(AI-extracted ideas for product enhancement.)*")
    sug_counts = filtered_df[filtered_df['suggestion'] != 'no suggestion']['suggestion'].value_counts()
    
    fig5 = px.bar(
        sug_counts,
        labels={'value': 'Count', 'index': 'Suggestion'},
        color_discrete_sequence=['#2ECC71'] # Green
    )
    st.plotly_chart(fig5, use_container_width=True)

st.markdown("---")

# --- Review Details (Data Table) ---

st.header('📝 Review Details')
st.markdown("Browse individual reviews, their star rating, and the AI-extracted insights.")

# Handle ASIN dropdown (hide if only one unique)
unique_asins = filtered_df['asin'].unique()
if len(unique_asins) > 1:
    product = st.selectbox('Select Product ID', unique_asins)
    details_df = filtered_df[filtered_df['asin'] == product]
else:
    details_df = filtered_df

display_cols = ['reviewText', 'overall', 'sentiment', 'theme_label', 'pain_point', 'suggestion']

# Use st.expander for a cleaner layout
with st.expander(f"View {len(details_df)} Sample Reviews (Click to expand)"):
    st.dataframe(details_df[display_cols], height=300)

st.markdown("---")
st.markdown("Built with **Streamlit, Python, and HuggingFace AI models** for portfolio demonstration.")

