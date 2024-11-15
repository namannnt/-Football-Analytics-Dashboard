import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# --- Streamlit App Sections ---
st.title("Football Analytics Dashboard")

# File uploader for player data
uploaded_file = st.file_uploader("Upload Player Data CSV", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # --- Filter Data for Radar Chart ---
    st.subheader("Player Comparison Radar Chart")
    country = st.selectbox("Select Country", data['nationality'].unique())
    position = st.selectbox("Select Position", ['Forward', 'Midfielder', 'Defender', 'Goalkeeper'])

    position_keywords = {
        'Forward': ['ST', 'CF', 'LW', 'RW'],
        'Midfielder': ['CM', 'CAM', 'RM', 'LM', 'CDM'],
        'Defender': ['CB', 'LB', 'RB', 'LWB', 'RWB'],
        'Goalkeeper': ['GK']
    }

    attributes_dict = {
        'Forward': ['overall_rating', 'crossing', 'finishing', 'short_passing', 'volleys', 'dribbling', 'ball_control', 'positioning'],
        'Midfielder': ['overall_rating', 'short_passing', 'long_passing', 'ball_control', 'vision', 'positioning', 'stamina', 'agility'],
        'Defender': ['overall_rating', 'standing_tackle', 'sliding_tackle', 'interceptions', 'strength', 'jumping', 'heading_accuracy', 'aggression'],
        'Goalkeeper': ['overall_rating', 'GK_diving', 'GK_handling', 'GK_kicking', 'GK_positioning', 'GK_reflexes']
    }

    filtered_data = data[(data['nationality'] == country) & 
                         (data['positions'].str.contains('|'.join(position_keywords[position])))]

    top_players = filtered_data.nlargest(5, 'overall_rating')
    attributes = attributes_dict[position]

    # Normalize attributes
    top_players[attributes] = top_players[attributes].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    # --- Radar Chart Function ---
    def plot_radar(data, attributes, title):
        num_vars = len(attributes)
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles += angles[:1]

        plt.figure(figsize=(8, 8))
        for i, row in data.iterrows():
            values = row[attributes].values.flatten().tolist()
            values += values[:1]

            plt.polar(angles, values, label=row['full_name'])

        plt.xticks(angles[:-1], attributes)
        plt.fill(angles, values, alpha=0.1)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.title(title)
        st.pyplot(plt)

    # Plot radar chart for the selected country
    if not top_players.empty:
        plot_radar(top_players, attributes, f"Top 5 {position}s in {country}")
    else:
        st.warning(f"No top players found for {country} in {position} position.")

    # Plot radar chart for top 5 players globally
    global_data = data[data['positions'].str.contains('|'.join(position_keywords[position]))]
    top_global_players = global_data.nlargest(5, 'overall_rating')
    top_global_players[attributes] = top_global_players[attributes].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    if not top_global_players.empty:
        plot_radar(top_global_players, attributes, f"Top 5 {position}s Globally")
    else:
        st.warning(f"No top global players found for {position} position.")

    # --- Additional Visualizations ---
    st.subheader("Additional Visualizations")

    st.subheader("Histogram of Overall Rating")
    def plot_histogram(data):
        plt.figure(figsize=(10, 6))
        sns.histplot(data=data, x='overall_rating', hue='nationality', multiple="stack", kde=True)
        plt.title('Distribution of Overall Rating by Country')
        plt.xlabel('Overall Rating')
        plt.ylabel('Number of Players')
        st.pyplot(plt)
        
    plot_histogram(data)

    st.subheader("Boxplot of Stamina by Country")
    def plot_boxplot(data):
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=data, x='nationality', y='stamina')
        plt.title('Stamina Distribution by Country')
        plt.xticks(rotation=45)
        st.pyplot(plt)
        
    plot_boxplot(data)

    st.subheader("Correlation Matrix of Player Attributes")
    def plot_correlation_matrix(data, attributes):
        corr_matrix = data[attributes].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        st.pyplot(plt)
        
    plot_correlation_matrix(data, attributes)

# --- Model Training Section ---
st.header("Model Training and Evaluation")
uploaded_model_file = st.file_uploader("Upload Model Training CSV file", type="csv")
if uploaded_model_file:
    model_data = pd.read_csv(uploaded_model_file)
    model_data.columns = ['team1', 'team 2', 'team1_goals', 'team2_goals', 'winning']  # Changed team2 to team 2

    # One-hot encode team names
    data_encoded = pd.get_dummies(model_data, columns=['team1', 'team 2'])  # Changed team2 to team 2

    # Label encode the target column
    label_encoder = LabelEncoder()
    data_encoded['winning'] = label_encoder.fit_transform(data_encoded['winning'])  # Changed to 'winning'

    # Split dataset
    X = data_encoded.drop(columns=['winning'])
    y = data_encoded['winning']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train RandomForest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Display model accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.subheader("Model Accuracy")
    st.write(f"Accuracy: {accuracy:.2f}")

    # Generate and display classification report
    st.subheader("Classification Report")
    unique_classes = sorted(set(y_test) | set(y_pred))
    target_names = [label_encoder.classes_[i] for i in unique_classes]
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

# --- Match Data Analytics ---
st.header("Match Data Analytics")
uploaded_match_file = st.file_uploader("Upload Match Data CSV", type="csv")
if uploaded_match_file:
    match_data = pd.read_csv(uploaded_match_file)
    
    # Clean column names
    match_data.columns = match_data.columns.str.strip()

    # Create 'winner' column based on 'winning' column
    match_data['winner'] = match_data.apply(lambda row: row['team1'] if row['winning'] == 'team1' else row['team 2'], axis=1)  # Changed to 'team 2'

    # Calculate total matches played and wins
    total_matches = match_data['team1'].value_counts() + match_data['team 2'].value_counts()  # Changed to 'team 2'
    total_wins = match_data['winner'].value_counts()

    # Calculate win percentage
    win_percentage = (total_wins / total_matches * 100).fillna(0)

    # --- Visualization: Win Percentage of Teams ---
    st.header("Win Percentage of Teams")
    plt.figure(figsize=(10, 6))
    sns.barplot(x=win_percentage.index, y=win_percentage.values, palette='viridis')
    plt.title('Win Percentage of Teams', fontsize=16)
    plt.xlabel('Teams', fontsize=12)
    plt.ylabel('Win Percentage (%)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)
