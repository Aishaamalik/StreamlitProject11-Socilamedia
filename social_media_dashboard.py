import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.linear_model import (
    LogisticRegression, LinearRegression,
    Ridge, Lasso
)
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy import stats
import pickle
import io
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Social Media Engagement Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
    }
    
    .info-box {
        background-color: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .stSelectbox, .stTextInput, .stNumberInput {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Social Media Engagement Dataset.csv")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Data preprocessing function
def preprocess_data(df):
    if df is None:
        return None
    
    # Create a copy to avoid modifying original data
    df_processed = df.copy()
    
    # Convert timestamp to datetime
    df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'])
    df_processed['date'] = df_processed['timestamp'].dt.date
    df_processed['hour'] = df_processed['timestamp'].dt.hour
    df_processed['month'] = df_processed['timestamp'].dt.month
    df_processed['year'] = df_processed['timestamp'].dt.year
    
    # Handle missing values
    numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
    df_processed[numeric_columns] = df_processed[numeric_columns].fillna(df_processed[numeric_columns].mean())
    
    # Fill categorical missing values
    categorical_columns = df_processed.select_dtypes(include=['object']).columns
    df_processed[categorical_columns] = df_processed[categorical_columns].fillna('Unknown')
    
    return df_processed

# Main function
def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Social Media Engagement Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Dashboard Controls")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    if df is None:
        st.error("Failed to load data. Please check if the CSV file is in the correct location.")
        return
    
    # Preprocess data
    with st.spinner("Preprocessing data..."):
        df_processed = preprocess_data(df)
    
    if df_processed is None:
        st.error("Failed to preprocess data.")
        return
    
    # Sidebar filters
    st.sidebar.subheader("📋 Filters")
    
    # Platform filter
    platforms = ['All'] + list(df_processed['platform'].unique())
    selected_platform = st.sidebar.selectbox("Select Platform", platforms)
    
    # Brand filter
    brands = ['All'] + list(df_processed['brand_name'].unique())
    selected_brand = st.sidebar.selectbox("Select Brand", brands)
    
    # Sentiment filter
    sentiments = ['All'] + list(df_processed['sentiment_label'].unique())
    selected_sentiment = st.sidebar.selectbox("Select Sentiment", sentiments)
    
    # Date range filter
    min_date = df_processed['date'].min()
    max_date = df_processed['date'].max()
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Apply filters
    filtered_df = df_processed.copy()
    if selected_platform != 'All':
        filtered_df = filtered_df[filtered_df['platform'] == selected_platform]
    if selected_brand != 'All':
        filtered_df = filtered_df[filtered_df['brand_name'] == selected_brand]
    if selected_sentiment != 'All':
        filtered_df = filtered_df[filtered_df['sentiment_label'] == selected_sentiment]
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['date'] >= date_range[0]) & 
            (filtered_df['date'] <= date_range[1])
        ]
    
    # Main content with dropdowns
    st.markdown('<h2 class="section-header">📊 Dashboard Sections</h2>', unsafe_allow_html=True)
    
    # Data Exploration Section
    with st.expander("📊 Data Exploration", expanded=True):
        st.markdown('<h2 class="section-header">📊 Data Exploration</h2>', unsafe_allow_html=True)
        
        # Dataset overview
        st.subheader("📋 Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"**Total Records:** {len(filtered_df):,}")
            st.info(f"**Total Features:** {len(filtered_df.columns)}")
            st.info(f"**Date Range:** {filtered_df['date'].min()} to {filtered_df['date'].max()}")
        
        with col2:
            st.info(f"**Platforms:** {len(filtered_df['platform'].unique())}")
            st.info(f"**Brands:** {len(filtered_df['brand_name'].unique())}")
            st.info(f"**Languages:** {len(filtered_df['language'].unique())}")
        
        with col3:
            st.info(f"**Missing Values:** {filtered_df.isnull().sum().sum()}")
            st.info(f"**Duplicate Rows:** {filtered_df.duplicated().sum()}")
            st.info(f"**Memory Usage:** {filtered_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Data structure
        st.subheader("🏗️ Data Structure")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Types:**")
            dtype_df = pd.DataFrame({
                'Column': filtered_df.dtypes.index,
                'Data Type': filtered_df.dtypes.values.astype(str)
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        with col2:
            st.write("**Sample Data:**")
            st.dataframe(filtered_df.head(10), use_container_width=True)
        
        # Column descriptions
        st.subheader("📝 Column Descriptions")
        
        column_descriptions = {
            'post_id': 'Unique identifier for each social media post',
            'timestamp': 'Date and time when the post was created',
            'day_of_week': 'Day of the week when the post was made',
            'platform': 'Social media platform (Instagram, Twitter, Reddit, YouTube)',
            'user_id': 'Unique identifier for the user who made the post',
            'location': 'Geographic location of the user',
            'language': 'Language of the post content',
            'text_content': 'The actual text content of the post',
            'hashtags': 'Hashtags used in the post',
            'mentions': 'User mentions in the post',
            'keywords': 'Key words extracted from the post',
            'topic_category': 'Category of the post topic',
            'sentiment_score': 'Numeric sentiment score (-1 to 1)',
            'sentiment_label': 'Categorical sentiment (Positive, Negative, Neutral)',
            'emotion_type': 'Type of emotion detected',
            'toxicity_score': 'Toxicity level score (0 to 1)',
            'likes_count': 'Number of likes received',
            'shares_count': 'Number of shares received',
            'comments_count': 'Number of comments received',
            'impressions': 'Number of impressions/views',
            'engagement_rate': 'Calculated engagement rate',
            'brand_name': 'Brand mentioned in the post',
            'product_name': 'Product mentioned in the post',
            'campaign_name': 'Marketing campaign name',
            'campaign_phase': 'Phase of the marketing campaign',
            'user_past_sentiment_avg': 'Average past sentiment of the user',
            'user_engagement_growth': 'User engagement growth rate',
            'buzz_change_rate': 'Change in buzz/trending rate'
        }
        
        desc_df = pd.DataFrame(list(column_descriptions.items()), columns=['Column', 'Description'])
        st.dataframe(desc_df, use_container_width=True)
        
        # Missing values analysis
        st.subheader("🔍 Missing Values Analysis")
        
        missing_data = filtered_df.isnull().sum()
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing %': (missing_data.values / len(filtered_df)) * 100
        }).sort_values('Missing Count', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_missing = px.bar(
                missing_df.head(15),
                x='Missing Count',
                y='Column',
                orientation='h',
                title="Missing Values by Column",
                color='Missing Count',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_missing, use_container_width=True)
        
        with col2:
            st.dataframe(missing_df, use_container_width=True)
    
    # Overview Dashboard Section
    with st.expander("📈 Overview Dashboard", expanded=False):
        st.markdown('<h2 class="section-header">📈 Overview Dashboard</h2>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Posts</h3>
                <h2>{len(filtered_df):,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_engagement = filtered_df['engagement_rate'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3>Avg Engagement Rate</h3>
                <h2>{avg_engagement:.2%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_sentiment = filtered_df['sentiment_score'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3>Avg Sentiment Score</h3>
                <h2>{avg_sentiment:.3f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            total_impressions = filtered_df['impressions'].sum()
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Impressions</h3>
                <h2>{total_impressions:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_likes = filtered_df['likes_count'].sum()
            st.metric("Total Likes", f"{total_likes:,.0f}")
        
        with col2:
            total_shares = filtered_df['shares_count'].sum()
            st.metric("Total Shares", f"{total_shares:,.0f}")
        
        with col3:
            total_comments = filtered_df['comments_count'].sum()
            st.metric("Total Comments", f"{total_comments:,.0f}")
        
        with col4:
            avg_toxicity = filtered_df['toxicity_score'].mean()
            st.metric("Avg Toxicity Score", f"{avg_toxicity:.3f}")
        
        # Data summary
        st.markdown('<h3 class="section-header">📋 Dataset Summary</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Platform Distribution")
            platform_counts = filtered_df['platform'].value_counts()
            fig_platform_pie = px.pie(
                values=platform_counts.values,
                names=platform_counts.index,
                title="Posts by Platform",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_platform_pie, use_container_width=True)
        
        with col2:
            st.subheader("Sentiment Distribution")
            sentiment_counts = filtered_df['sentiment_label'].value_counts()
            fig_sentiment_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Posts by Sentiment",
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            st.plotly_chart(fig_sentiment_pie, use_container_width=True)
    
    # EDA Section
    with st.expander("🔍 Exploratory Data Analysis", expanded=False):
        st.markdown('<h2 class="section-header">🔍 Exploratory Data Analysis</h2>', unsafe_allow_html=True)
        
        # Distribution analysis
        st.subheader("📊 Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution
            fig_sentiment = px.histogram(
                filtered_df, 
                x='sentiment_score', 
                nbins=30,
                title="Sentiment Score Distribution",
                color_discrete_sequence=['#667eea'],
                marginal='box'
            )
            fig_sentiment.update_layout(showlegend=False)
            st.plotly_chart(fig_sentiment, use_container_width=True)
        
        with col2:
            # Engagement rate distribution
            fig_engagement = px.histogram(
                filtered_df, 
                x='engagement_rate', 
                nbins=30,
                title="Engagement Rate Distribution",
                color_discrete_sequence=['#764ba2'],
                marginal='box'
            )
            fig_engagement.update_layout(showlegend=False)
            st.plotly_chart(fig_engagement, use_container_width=True)
        
        # Correlation analysis
        st.subheader("🔗 Correlation Analysis")
        
        # Select numeric columns for correlation
        numeric_cols = ['sentiment_score', 'toxicity_score', 'likes_count', 'shares_count', 
                       'comments_count', 'impressions', 'engagement_rate', 
                       'user_past_sentiment_avg', 'user_engagement_growth', 'buzz_change_rate']
        
        # Filter columns that exist in the dataset
        available_cols = [col for col in numeric_cols if col in filtered_df.columns]
        correlation_matrix = filtered_df[available_cols].corr()
        
        # Create a more realistic correlation heatmap
        fig_corr = px.imshow(
            correlation_matrix,
            title="Correlation Matrix Heatmap",
            color_continuous_scale='RdBu_r',
            aspect='auto',
            text_auto=True,
            width=800,
            height=600
        )
        
        # Update layout for better appearance
        fig_corr.update_layout(
            title_x=0.5,
            title_font_size=20,
            xaxis_title="Features",
            yaxis_title="Features",
            font=dict(size=12)
        )
        
        # Update colorbar
        fig_corr.update_coloraxes(
            colorbar_title="Correlation Coefficient",
            colorbar_len=0.8
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Top correlations
        st.subheader("🔝 Top Correlations")
        
        # Get top correlations
        corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_pairs.append((
                    correlation_matrix.columns[i],
                    correlation_matrix.columns[j],
                    correlation_matrix.iloc[i, j]
                ))
        
        corr_df = pd.DataFrame(corr_pairs, columns=['Variable 1', 'Variable 2', 'Correlation'])
        corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
        
        # Display top correlations with better formatting
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Strongest Positive Correlations:**")
            positive_corr = corr_df[corr_df['Correlation'] > 0.3].head(5)
            st.dataframe(positive_corr, use_container_width=True)
        
        with col2:
            st.write("**Strongest Negative Correlations:**")
            negative_corr = corr_df[corr_df['Correlation'] < -0.3].head(5)
            st.dataframe(negative_corr, use_container_width=True)
    
    # Visualizations Section
    with st.expander("📊 Data Visualizations", expanded=False):
        st.markdown('<h2 class="section-header">📊 Data Visualizations</h2>', unsafe_allow_html=True)
        
        # Time series analysis
        st.subheader("⏰ Time Series Analysis")
        
        # Daily engagement over time
        daily_engagement = filtered_df.groupby('date').agg({
            'engagement_rate': 'mean',
            'sentiment_score': 'mean',
            'impressions': 'sum',
            'likes_count': 'sum'
        }).reset_index()
        
        fig_time = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Daily Average Engagement Rate', 'Daily Average Sentiment Score', 
                          'Daily Total Impressions', 'Daily Total Likes'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Engagement Rate
        fig_time.add_trace(
            go.Scatter(x=daily_engagement['date'], y=daily_engagement['engagement_rate'], 
                      mode='lines+markers', name='Engagement Rate', 
                      line=dict(color='#667eea', width=2),
                      marker=dict(size=4)),
            row=1, col=1
        )
        
        # Sentiment Score
        fig_time.add_trace(
            go.Scatter(x=daily_engagement['date'], y=daily_engagement['sentiment_score'], 
                      mode='lines+markers', name='Sentiment Score', 
                      line=dict(color='#764ba2', width=2),
                      marker=dict(size=4)),
            row=1, col=2
        )
        
        # Impressions
        fig_time.add_trace(
            go.Scatter(x=daily_engagement['date'], y=daily_engagement['impressions'], 
                      mode='lines+markers', name='Impressions', 
                      line=dict(color='#f093fb', width=2),
                      marker=dict(size=4)),
            row=2, col=1
        )
        
        # Likes
        fig_time.add_trace(
            go.Scatter(x=daily_engagement['date'], y=daily_engagement['likes_count'], 
                      mode='lines+markers', name='Likes', 
                      line=dict(color='#4facfe', width=2),
                      marker=dict(size=4)),
            row=2, col=2
        )
        
        fig_time.update_layout(height=600, showlegend=False, title_text="Time Series Trends")
        fig_time.update_xaxes(title_text="Date")
        fig_time.update_yaxes(title_text="Value")
        
        st.plotly_chart(fig_time, use_container_width=True)
        
        # Platform and brand analysis
        st.subheader("🏢 Platform & Brand Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Platform performance
            platform_stats = filtered_df.groupby('platform').agg({
                'engagement_rate': 'mean',
                'sentiment_score': 'mean',
                'impressions': 'sum'
            }).reset_index()
            
            fig_platform = px.bar(
                platform_stats,
                x='platform',
                y='engagement_rate',
                title="Average Engagement Rate by Platform",
                color='engagement_rate',
                color_continuous_scale='Viridis',
                text='engagement_rate'
            )
            fig_platform.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(fig_platform, use_container_width=True)
        
        with col2:
            # Top brands by engagement
            brand_stats = filtered_df.groupby('brand_name').agg({
                'engagement_rate': 'mean',
                'impressions': 'sum'
            }).reset_index().sort_values('engagement_rate', ascending=False)
            
            fig_brand = px.bar(
                brand_stats.head(10),
                x='brand_name',
                y='engagement_rate',
                title="Top 10 Brands by Engagement Rate",
                color='engagement_rate',
                color_continuous_scale='Plasma',
                text='engagement_rate'
            )
            fig_brand.update_xaxes(tickangle=45)
            fig_brand.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(fig_brand, use_container_width=True)
        
        # Sentiment analysis
        st.subheader("😊 Sentiment Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment label distribution
            sentiment_counts = filtered_df['sentiment_label'].value_counts()
            fig_sentiment_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Sentiment Label Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_sentiment_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_sentiment_pie, use_container_width=True)
        
        with col2:
            # Emotion type distribution
            emotion_counts = filtered_df['emotion_type'].value_counts()
            fig_emotion = px.bar(
                x=emotion_counts.index,
                y=emotion_counts.values,
                title="Emotion Type Distribution",
                color=emotion_counts.values,
                color_continuous_scale='Viridis',
                text=emotion_counts.values
            )
            fig_emotion.update_traces(textposition='outside')
            st.plotly_chart(fig_emotion, use_container_width=True)
        
        # Interactive scatter plot
        st.subheader("🎯 Interactive Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_axis = st.selectbox("Select X-axis", ['sentiment_score', 'engagement_rate', 'impressions', 'likes_count', 'shares_count'])
        
        with col2:
            y_axis = st.selectbox("Select Y-axis", ['engagement_rate', 'sentiment_score', 'impressions', 'likes_count', 'shares_count'])
        
        with col3:
            color_by = st.selectbox("Color by", ['platform', 'sentiment_label', 'emotion_type', 'brand_name'])
        
        fig_scatter = px.scatter(
            filtered_df,
            x=x_axis,
            y=y_axis,
            color=color_by,
            hover_data=['text_content', 'brand_name', 'platform'],
            title=f"{x_axis.replace('_', ' ').title()} vs {y_axis.replace('_', ' ').title()}",
            color_discrete_sequence=px.colors.qualitative.Set3,
            size='impressions',
            size_max=20
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # ML Models Section
    with st.expander("🤖 Machine Learning Models", expanded=False):
        st.markdown('<h2 class="section-header">🤖 Machine Learning Models</h2>', unsafe_allow_html=True)
        
        # Model selection
        st.subheader("🎯 Model Configuration")
        
        # Add helpful information
        st.info("💡 **Tip**: For best results, use 'All' filters or select filters that include diverse data. Very specific filters may result in insufficient data for training.")
        
        # Show current data info
        st.write(f"**Current filtered dataset size:** {len(filtered_df)} records")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Select Model Type",
                ["Classification", "Regression"],
                help="Classification: Predict sentiment labels, Regression: Predict engagement rate"
            )
        
        with col2:
            if model_type == "Classification":
                target_col = st.selectbox("Select Target Variable", ['sentiment_label', 'emotion_type'])
            else:
                target_col = st.selectbox("Select Target Variable", ['engagement_rate', 'impressions', 'likes_count'])
        
        # Show target variable information after selection
        st.write(f"**Selected target variable:** {target_col}")
        
        if model_type == "Classification":
            target_counts = filtered_df[target_col].value_counts()
            st.write(f"**Target variable '{target_col}' distribution:**")
            st.dataframe(target_counts, use_container_width=True)
        else:
            st.write(f"**Target variable '{target_col}' statistics:**")
            st.write(f"Mean: {filtered_df[target_col].mean():.4f}, Std: {filtered_df[target_col].std():.4f}")
            st.write(f"Min: {filtered_df[target_col].min():.4f}, Max: {filtered_df[target_col].max():.4f}")
        
        # Feature engineering
        st.subheader("🔧 Feature Engineering")
        
        # Prepare features
        feature_cols = ['sentiment_score', 'toxicity_score', 'likes_count', 'shares_count', 
                       'comments_count', 'impressions', 'user_past_sentiment_avg', 
                       'user_engagement_growth', 'buzz_change_rate']
        
        # Add encoded categorical features
        le_platform = LabelEncoder()
        le_brand = LabelEncoder()
        
        X = filtered_df[feature_cols].copy()
        X['platform_encoded'] = le_platform.fit_transform(filtered_df['platform'])
        X['brand_encoded'] = le_brand.fit_transform(filtered_df['brand_name'])
        
        # Prepare target
        if model_type == "Classification":
            le_target = LabelEncoder()
            y = le_target.fit_transform(filtered_df[target_col])
        else:
            y = filtered_df[target_col]
        
        # Remove rows with missing values
        mask = ~(X.isnull().any(axis=1) | pd.isnull(y))
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            st.error("No valid data for modeling after removing missing values.")
            return
        
        # Check for classification issues
        if model_type == "Classification":
            unique_classes = np.unique(y)
            if len(unique_classes) < 2:
                st.error(f"❌ Classification requires at least 2 classes. Found only {len(unique_classes)} class(es) in {target_col}.")
                st.info("💡 Try selecting different filters or a different target variable.")
                return
            
            # Check if any class has too few samples
            class_counts = np.bincount(y)
            min_samples = min(class_counts)
            if min_samples < 10:
                st.warning(f"⚠️ Some classes have very few samples (minimum: {min_samples}). This may affect model performance.")
        
        # Check for regression issues
        else:
            if len(y) < 10:
                st.error("❌ Regression requires at least 10 samples. Found only {len(y)} samples.")
                st.info("💡 Try selecting different filters or a different target variable.")
                return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Model training
        st.subheader("🚀 Model Training")
        
        if model_type == "Classification":
            # Classification models
            models = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'Support Vector Machine': SVC(random_state=42, probability=True),
                'Gaussian Naive Bayes': GaussianNB(),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42)
            }
            
            results = {}
            
            for name, model in models.items():
                with st.spinner(f"Training {name}..."):
                    try:
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        accuracy = accuracy_score(y_test, y_pred)
                        results[name] = {
                            'model': model,
                            'accuracy': accuracy,
                            'predictions': y_pred
                        }
                        
                        # Save to session state for export
                        if 'trained_models' not in st.session_state:
                            st.session_state.trained_models = {}
                        st.session_state.trained_models[name] = results[name]
                    except Exception as e:
                        st.error(f"❌ Error training {name}: {str(e)}")
                        if "SVC" in name or "SVR" in name:
                            st.info(f"💡 SVM models might fail with large datasets or certain data types. Try using fewer features or different data.")
                        elif "Naive Bayes" in name:
                            st.info(f"💡 Naive Bayes might fail with negative values or certain data distributions.")
                        elif "Gradient Boosting" in name:
                            st.info(f"💡 Gradient Boosting might fail with insufficient data or extreme values.")
                        else:
                            st.info(f"💡 This might be due to insufficient data, class imbalance, or data type issues. Try adjusting your filters.")
                        continue
            
            # Display results
            if not results:
                st.error("❌ No models were successfully trained. Please check your data and filters.")
                return
                
            st.subheader("📊 Classification Results")
            
            # Model comparison table
            st.subheader("🏆 Model Comparison")
            comparison_data = []
            for name, result in results.items():
                comparison_data.append({
                    'Model': name,
                    'Accuracy': f"{result['accuracy']:.4f}",
                    'Status': '✅ Trained Successfully'
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Accuracy comparison
                accuracies = {name: result['accuracy'] for name, result in results.items()}
                fig_acc = px.bar(
                    x=list(accuracies.keys()),
                    y=list(accuracies.values()),
                    title="Model Accuracy Comparison",
                    color=list(accuracies.values()),
                    color_continuous_scale='Viridis',
                    text=list(accuracies.values())
                )
                fig_acc.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                fig_acc.update_layout(yaxis_title="Accuracy")
                st.plotly_chart(fig_acc, use_container_width=True)
            
            with col2:
                # Best model details
                best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
                best_model = results[best_model_name]
                
                st.markdown(f"**Best Model:** {best_model_name}")
                st.markdown(f"**Accuracy:** {best_model['accuracy']:.4f}")
                
                # Feature importance for Random Forest
                if best_model_name == 'Random Forest':
                    feature_importance = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': best_model['model'].feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig_importance = px.bar(
                        feature_importance.head(10),
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Top 10 Feature Importance",
                        color='Importance',
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
            
            # Confusion matrix for best model
            st.subheader("📈 Confusion Matrix")
            
            cm = confusion_matrix(y_test, best_model['predictions'])
            
            # Create confusion matrix heatmap
            fig_cm = ff.create_annotated_heatmap(
                cm,
                x=[f'Predicted {i}' for i in range(len(cm))],
                y=[f'Actual {i}' for i in range(len(cm))],
                colorscale='Blues',
                showscale=True
            )
            fig_cm.update_layout(title="Confusion Matrix")
            st.plotly_chart(fig_cm, use_container_width=True)
        
        else:
            # Regression models
            models = {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Linear Regression': LinearRegression(),
                'Support Vector Regression': SVR(),
                'Gradient Boosting': GradientBoostingRegressor(random_state=42),
                'Decision Tree': DecisionTreeRegressor(random_state=42),
                'Ridge Regression': Ridge(random_state=42),
                'Lasso Regression': Lasso(random_state=42)
            }
            
            results = {}
            
            for name, model in models.items():
                with st.spinner(f"Training {name}..."):
                    try:
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        results[name] = {
                            'model': model,
                            'mse': mse,
                            'r2': r2,
                            'predictions': y_pred
                        }
                        
                        # Save to session state for export
                        if 'trained_models' not in st.session_state:
                            st.session_state.trained_models = {}
                        st.session_state.trained_models[name] = results[name]
                    except Exception as e:
                        st.error(f"❌ Error training {name}: {str(e)}")
                        if "SVR" in name:
                            st.info(f"💡 SVR might fail with large datasets or certain data types. Try using fewer features or different data.")
                        elif "Gradient Boosting" in name:
                            st.info(f"💡 Gradient Boosting might fail with insufficient data or extreme values.")
                        elif "Ridge" in name or "Lasso" in name:
                            st.info(f"💡 Regularized regression might fail with certain data distributions or scaling issues.")
                        else:
                            st.info(f"💡 This might be due to insufficient data, feature issues, or data type problems. Try adjusting your filters.")
                        continue
            
            # Display results
            if not results:
                st.error("❌ No models were successfully trained. Please check your data and filters.")
                return
                
            st.subheader("📊 Regression Results")
            
            # Model comparison table
            st.subheader("🏆 Model Comparison")
            comparison_data = []
            for name, result in results.items():
                comparison_data.append({
                    'Model': name,
                    'R² Score': f"{result['r2']:.4f}",
                    'MSE': f"{result['mse']:.4f}",
                    'Status': '✅ Trained Successfully'
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # R² comparison
                r2_scores = {name: result['r2'] for name, result in results.items()}
                fig_r2 = px.bar(
                    x=list(r2_scores.keys()),
                    y=list(r2_scores.values()),
                    title="Model R² Score Comparison",
                    color=list(r2_scores.values()),
                    color_continuous_scale='Viridis',
                    text=list(r2_scores.values())
                )
                fig_r2.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                fig_r2.update_layout(yaxis_title="R² Score")
                st.plotly_chart(fig_r2, use_container_width=True)
            
            with col2:
                # Best model details
                best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
                best_model = results[best_model_name]
                
                st.markdown(f"**Best Model:** {best_model_name}")
                st.markdown(f"**R² Score:** {best_model['r2']:.4f}")
                st.markdown(f"**Mean Squared Error:** {best_model['mse']:.4f}")
            
            # Actual vs Predicted plot
            st.subheader("📈 Actual vs Predicted")
            
            fig_pred = px.scatter(
                x=y_test,
                y=best_model['predictions'],
                title="Actual vs Predicted Values",
                labels={'x': 'Actual', 'y': 'Predicted'}
            )
            
            # Add perfect prediction line
            min_val = min(y_test.min(), best_model['predictions'].min())
            max_val = max(y_test.max(), best_model['predictions'].max())
            fig_pred.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(dash='dash', color='red')
            ))
            
            st.plotly_chart(fig_pred, use_container_width=True)
    
    # Data Quality Section
    with st.expander("📋 Data Quality Report", expanded=False):
        st.markdown('<h2 class="section-header">📋 Data Quality Report</h2>', unsafe_allow_html=True)
        
        # Data quality metrics
        st.subheader("🔍 Data Quality Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Completeness
            completeness = (1 - filtered_df.isnull().sum() / len(filtered_df)) * 100
            fig_completeness = px.bar(
                x=completeness.index,
                y=completeness.values,
                title="Data Completeness (%)",
                color=completeness.values,
                color_continuous_scale='RdYlGn'
            )
            fig_completeness.update_layout(yaxis_title="Completeness %")
            fig_completeness.update_xaxes(tickangle=45)
            st.plotly_chart(fig_completeness, use_container_width=True)
        
        with col2:
            # Data types
            dtype_counts = filtered_df.dtypes.value_counts()
            # Create a bar chart instead of pie chart to avoid JSON serialization issues
            fig_dtypes = px.bar(
                x=dtype_counts.index.astype(str),
                y=dtype_counts.values,
                title="Data Types Distribution",
                color=dtype_counts.values,
                color_continuous_scale='Viridis',
                text=dtype_counts.values
            )
            fig_dtypes.update_traces(textposition='outside')
            fig_dtypes.update_layout(xaxis_title="Data Type", yaxis_title="Count")
            st.plotly_chart(fig_dtypes, use_container_width=True)
            
            # Add a simple data types summary table
            st.write("**Data Types Summary:**")
            dtype_summary = pd.DataFrame({
                'Data Type': dtype_counts.index.astype(str),
                'Count': dtype_counts.values
            })
            st.dataframe(dtype_summary, use_container_width=True)
        
        # Data quality summary
        st.subheader("📊 Quality Summary")
        
        quality_metrics = {
            'Total Rows': len(filtered_df),
            'Total Columns': len(filtered_df.columns),
            'Missing Values': filtered_df.isnull().sum().sum(),
            'Duplicate Rows': filtered_df.duplicated().sum(),
            'Memory Usage': f"{filtered_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        }
        
        for metric, value in quality_metrics.items():
            st.metric(metric, value)
        
        # Data validation
        st.subheader("✅ Data Validation")
        
        validation_results = []
        
        # Check for negative values in positive-only columns
        positive_cols = ['engagement_rate', 'impressions', 'likes_count', 'shares_count', 'comments_count']
        for col in positive_cols:
            if col in filtered_df.columns:
                negative_count = (filtered_df[col] < 0).sum()
                validation_results.append({
                    'Check': f'Negative values in {col}',
                    'Status': '❌ Failed' if negative_count > 0 else '✅ Passed',
                    'Details': f'{negative_count} negative values found' if negative_count > 0 else 'All values positive'
                })
        
        # Check for extreme values
        for col in ['engagement_rate', 'sentiment_score']:
            if col in filtered_df.columns:
                extreme_count = ((filtered_df[col] > filtered_df[col].quantile(0.99)) | 
                               (filtered_df[col] < filtered_df[col].quantile(0.01))).sum()
                validation_results.append({
                    'Check': f'Extreme values in {col}',
                    'Status': '⚠️ Warning' if extreme_count > 0 else '✅ Passed',
                    'Details': f'{extreme_count} extreme values found' if extreme_count > 0 else 'No extreme values'
                })
        
        validation_df = pd.DataFrame(validation_results)
        st.dataframe(validation_df, use_container_width=True)
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        recommendations = [
            "✅ Data quality is generally good with high completeness",
            "🔧 Consider handling outliers in engagement_rate and sentiment_score",
            "📊 Monitor for data drift in sentiment scores over time",
            "🎯 Focus on improving engagement prediction models",
            "📈 Consider adding more features for better model performance"
        ]
        
        for rec in recommendations:
            st.write(rec)
    
    # Export Section
    with st.expander("📤 Export Data & Models", expanded=False):
        st.markdown('<h2 class="section-header">📤 Export Data & Models</h2>', unsafe_allow_html=True)
        
        # Export cleaned data
        st.subheader("📊 Export Cleaned Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Current Filtered Dataset:**")
            st.write(f"• **Rows:** {len(filtered_df):,}")
            st.write(f"• **Columns:** {len(filtered_df.columns)}")
            st.write(f"• **File Size:** {filtered_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Export CSV
            csv_buffer = io.StringIO()
            filtered_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📥 Download Cleaned CSV",
                data=csv_data,
                file_name=f"social_media_cleaned_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Download the cleaned and filtered dataset as CSV"
            )
        
        with col2:
            st.write("**Data Summary:**")
            st.write("• Preprocessed and cleaned data")
            st.write("• Applied filters from sidebar")
            st.write("• Handled missing values")
            st.write("• Converted data types")
            
            # Show sample of data to be exported
            st.write("**Sample of data to be exported:**")
            st.dataframe(filtered_df.head(5), use_container_width=True)
        
        # Export trained models
        st.subheader("🤖 Export Trained Models")
        
        # Check if models were trained in this session
        if 'trained_models' not in st.session_state:
            st.session_state.trained_models = {}
        
        if st.session_state.trained_models:
            st.success("✅ Models available for export!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Available Models:**")
                for model_name, model_info in st.session_state.trained_models.items():
                    st.write(f"• **{model_name}**")
                    if 'accuracy' in model_info:
                        st.write(f"  - Accuracy: {model_info['accuracy']:.4f}")
                    if 'r2' in model_info:
                        st.write(f"  - R² Score: {model_info['r2']:.4f}")
                    if 'mse' in model_info:
                        st.write(f"  - MSE: {model_info['mse']:.4f}")
            
            with col2:
                # Export all models as pickle
                model_data = pickle.dumps(st.session_state.trained_models)
                
                st.download_button(
                    label="📥 Download All Models (PKL)",
                    data=model_data,
                    file_name=f"trained_models_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                    mime="application/octet-stream",
                    help="Download all trained models as pickle file"
                )
                
                # Export best model separately
                if st.session_state.trained_models:
                    best_model_name = max(st.session_state.trained_models.keys(), 
                                        key=lambda k: st.session_state.trained_models[k].get('accuracy', st.session_state.trained_models[k].get('r2', 0)))
                    best_model = st.session_state.trained_models[best_model_name]
                    
                    st.write(f"**Best Model:** {best_model_name}")
                    
                    best_model_data = pickle.dumps(best_model)
                    
                    st.download_button(
                        label=f"📥 Download Best Model ({best_model_name})",
                        data=best_model_data,
                        file_name=f"best_model_{best_model_name.replace(' ', '_')}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                        mime="application/octet-stream",
                        help=f"Download the best performing model: {best_model_name}"
                    )
        else:
            st.info("ℹ️ **No models trained yet.** Train models in the 'Machine Learning Models' section to export them.")
            st.write("**To export models:**")
            st.write("1. Go to '🤖 Machine Learning Models' section")
            st.write("2. Select model type and target variable")
            st.write("3. Train the models")
            st.write("4. Return here to export the trained models")
        
        # Export configuration and metadata
        st.subheader("⚙️ Export Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export configuration as JSON
            config_data = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'filters_applied': {
                    'platform': selected_platform,
                    'brand': selected_brand,
                    'sentiment': selected_sentiment,
                    'date_range': [str(date_range[0]), str(date_range[1])] if len(date_range) == 2 else None
                },
                'dataset_info': {
                    'total_rows': int(len(filtered_df)),
                    'total_columns': int(len(filtered_df.columns)),
                    'missing_values': int(filtered_df.isnull().sum().sum()),
                    'memory_usage_mb': float(filtered_df.memory_usage(deep=True).sum() / 1024**2)
                },
                'preprocessing_info': {
                    'timestamp_converted': True,
                    'missing_values_handled': True,
                    'data_types_optimized': True
                }
            }
            
            import json
            config_json = json.dumps(config_data, indent=2)
            
            st.download_button(
                label="📥 Download Configuration (JSON)",
                data=config_json,
                file_name=f"dashboard_config_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                help="Download the current dashboard configuration and metadata"
            )
        
        with col2:
            st.write("**Configuration includes:**")
            st.write("• Applied filters")
            st.write("• Dataset statistics")
            st.write("• Preprocessing steps")
            st.write("• Timestamp information")
        
        # Usage instructions
        st.subheader("📖 Usage Instructions")
        
        st.markdown("""
        **How to use exported files:**
        
        **📊 CSV File:**
        ```python
        import pandas as pd
        df = pd.read_csv('social_media_cleaned_data.csv')
        ```
        
        **🤖 Model Files:**
        ```python
        import pickle
        
        # Load all models
        with open('trained_models.pkl', 'rb') as f:
            models = pickle.load(f)
        
        # Load best model
        with open('best_model.pkl', 'rb') as f:
            best_model = pickle.load(f)
        
        # Use the model
        predictions = best_model['model'].predict(X_new)
        ```
        
        **⚙️ Configuration:**
        ```python
        import json
        with open('dashboard_config.json', 'r') as f:
            config = json.load(f)
        ```
        """)

if __name__ == "__main__":
    main() 