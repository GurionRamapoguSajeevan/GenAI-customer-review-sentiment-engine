import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the processed data
df = pd.read_csv('processed_reviews.csv')

# Define meaningful theme labels based on LDA top words (edit these from your notebook output)
theme_labels = {
    0: "Battery and Charging",  # Example: Replace with your actual top words summary
    1: "Performance and Speed",
    2: "Design and Build Quality",
    3: "Price and Value",
    4: "Customer Service Issues"
}
df['theme_label'] = df['theme'].map(theme_labels)  # Add labeled column

st.title('Amazon Review Insights Dashboard')
st.markdown("""
This dashboard analyzes customer reviews using AI and analytics. It shows sentiment (positive/negative), common themes (grouped topics from reviews), pain points (common complaints), and suggestions for improvements. Use the filters on the left to explore.
""")

# Sidebar for interactivity
st.sidebar.header('Filters')
selected_sentiments = st.sidebar.multiselect('Filter by Sentiment', options=df['sentiment'].unique(), default=df['sentiment'].unique(), help='Select sentiments to include in visualizations.')
selected_pains = st.sidebar.multiselect('Filter by Pain Points', options=df['pain_point'].unique(), default=df['pain_point'].unique(), help='Select pain points to focus on.')

# Apply filters
filtered_df = df[df['sentiment'].isin(selected_sentiments) & df['pain_point'].isin(selected_pains)]

# Sentiment Overview
st.header('Sentiment Distribution')
st.markdown("This chart shows how many reviews are positive vs. negative (based on AI analysis of text). Higher positive means happier customers.")
fig1, ax1 = plt.subplots(figsize=(8, 6))
sns.countplot(x='sentiment', data=filtered_df, ax=ax1)
ax1.set_ylabel('Number of Reviews')  # Y-axis label
ax1.set_xlabel('Sentiment')  # X-axis label for clarity
for container in ax1.containers:
    ax1.bar_label(container)  # Add data values on bars
st.pyplot(fig1)

# Themes Overview
st.header('Top Themes')
st.markdown("Themes are AI-grouped topics from reviews (e.g., 'Battery and Charging' discusses power-related feedback). This helps spot common discussion areas.")
fig2, ax2 = plt.subplots(figsize=(10, 6))
theme_counts = filtered_df['theme_label'].value_counts()
theme_counts.plot(kind='bar', ax=ax2)
ax2.set_ylabel('Number of Reviews')  # Y-axis label
ax2.set_xlabel('Theme')  # X-axis label
ax2.tick_params(axis='x', rotation=45)  # Rotate x labels for readability
for container in ax2.containers:
    ax2.bar_label(container)  # Add data values on bars
st.pyplot(fig2)

# Pain Points
st.header('Pain Points Distribution')
st.markdown("Pain points highlight common complaints extracted by AI, like 'quality issue' or 'delivery problem'. Use this to identify areas for improvement.")
fig3, ax3 = plt.subplots(figsize=(10, 6))
pain_counts = filtered_df['pain_point'].value_counts()
pain_counts.plot(kind='bar', ax=ax3)
ax3.set_ylabel('Number of Reviews')  # Y-axis label
ax3.set_xlabel('Pain Point')  # X-axis label
ax3.tick_params(axis='x', rotation=45)
for container in ax3.containers:
    ax3.bar_label(container)  # Add data values on bars
st.pyplot(fig3)

# Suggestions
st.header('Product Suggestions Distribution')
st.markdown("These are AI-detected ideas from reviews for product enhancements, such as 'improve durability'.")
fig4, ax4 = plt.subplots(figsize=(10, 6))
sug_counts = filtered_df['suggestion'].value_counts()
sug_counts.plot(kind='bar', ax=ax4)
ax4.set_ylabel('Number of Reviews')  # Y-axis label
ax4.set_xlabel('Suggestion')  # X-axis label
ax4.tick_params(axis='x', rotation=45)
for container in ax4.containers:
    ax4.bar_label(container)  # Add data values on bars
st.pyplot(fig4)

# Review Details
st.header('Review Details')
st.markdown("Browse individual reviews. Filter by product if multiple available.")

# Handle ASIN dropdown (hide if only one unique)
unique_asins = filtered_df['asin'].unique()
if len(unique_asins) > 1:
    product = st.selectbox('Select Product ID', unique_asins, help='Choose a product to see its specific reviews.')
    details_df = filtered_df[filtered_df['asin'] == product]
else:
    st.info("Only one product in dataset—showing all reviews below.")
    details_df = filtered_df

with st.expander("View Sample Reviews (click to expand)"):
    st.dataframe(details_df[['reviewText', 'overall', 'sentiment', 'pain_point', 'suggestion', 'theme_label']])

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, Python, and HuggingFace AI models. For portfolio demo purposes.")