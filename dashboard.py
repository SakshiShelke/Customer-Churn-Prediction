import streamlit as st
import shap
import matplotlib.pyplot as plt
shap.initjs()
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import plotly.graph_objs as go
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve

# Add theme selector in sidebar
#theme_mode = st.sidebar.selectbox("Select Theme for SHAP", ["light", "dark"])
theme_mode = "dark"

def apply_custom_theme(theme_mode):
    if theme_mode == 'dark':
        theme_mode = "dark"
        sns.set_theme(style="darkgrid")
        plt.rcParams.update({
            'axes.facecolor': "#DDE5E1",
            'figure.facecolor': "#111111",
            'axes.labelcolor': '#ffffff',
            'xtick.color': '#ffffff',
            'ytick.color': '#ffffff',
            'text.color': '#ffffff',
            'axes.edgecolor': '#ffffff',
            'axes.titlecolor': '#1f77b4',
            'grid.color': "#111111",
            'grid.linestyle': '--',
            'grid.alpha': 0.7
        })
    else:  # light theme
        theme_mode = "light"
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({
            'axes.facecolor': '#ffffff',
            'figure.facecolor': '#ffffff',
            'axes.labelcolor': '#000000',
            'xtick.color': '#000000',
            'ytick.color': '#000000',
            'text.color': '#000000',
            'axes.edgecolor': '#000000',
            'axes.titlecolor': '#1f77b4',
            'grid.color': '#d3d3d3',
            'grid.linestyle': '--',
            'grid.alpha': 0.7
        })

def plot_shap_single(model, X_input, theme_mode):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_input)

    st.subheader("🔎 SHAP Explanation for This Customer")
    st.markdown("Features pushing the prediction toward churn (red) or retention (blue).")

    apply_custom_theme(theme_mode)
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    fig = plt.gcf()
    st.pyplot(fig)
    plt.clf()

# Global variables
model = None
label_encoders = {}
feature_columns = []
df_predictions = None
y_train = None
y_pred_proba_train = None


def preprocess_data(df, is_train=True):
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])

    if is_train:
        if 'churned' in df.columns:
            if df['churned'].isnull().any():
                df['churned'] = df['churned'].fillna((df['tenure'] < 30).astype(int))
                df['churned'] = df['churned'].map({'Yes': 1, 'No': 0}).astype(int)
        else:
            df['churned'] = (df['tenure'] < 30).astype(int)

    replace_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines']
    for col in replace_cols:
        if col in df.columns:
            df[col] = df[col].replace({'No internet service': 'No', 'No phone service': 'No'})

    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'customerID' in cat_cols:
        cat_cols.remove('customerID')

    for col in cat_cols:
        if df[col].isnull().any():
            mode_value = df[col].mode()
            if not mode_value.empty:
                df[col].fillna(mode_value[0], inplace=True)
            else:
                df[col].fillna("Unknown", inplace=True)

    global label_encoders
    if is_train:
        label_encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    else:
        for col in cat_cols:
            if col in label_encoders:
                le = label_encoders[col]
                df[col] = df[col].map(lambda s: s if s in le.classes_ else 'No')
                if 'No' not in le.classes_:
                    le.classes_ = np.append(le.classes_, 'No')
                df[col] = le.transform(df[col])
            else:
                df[col] = 0

    return df

def train_model(df):
    target = 'churned'
    X = df.drop(columns=['customerID', target])
    y = df[target]

    if y.isnull().any():
        st.error("Target column 'churned' contains NaN values.")
        return None

    global feature_columns, y_train, y_pred_proba_train
    feature_columns = X.columns.tolist()

    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    clf.fit(X, y)
    y_pred_proba = clf.predict_proba(X)[:, 1]

    y_train = y
    y_pred_proba_train = y_pred_proba

    auc_score = roc_auc_score(y, y_pred_proba)
    apply_custom_theme(theme_mode)
    st.markdown(
        f'<p style="color:#ffffff; font-size:18px;padding:12px;background-color:#173928;border-radius:10px;">✅ Training AUC-ROC: {auc_score:.4f}</p>',
        unsafe_allow_html=True
    )
    return clf

def plot_shap_summary(model, X_train, theme_mode):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_train)

    st.subheader("🔍 SHAP Summary Plot (Feature Importance)")

    apply_custom_theme(theme_mode)
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)

    fig = plt.gcf()
    
    # ✅ Fix visibility for dark mode
    if theme_mode == "dark":
        fig.patch.set_facecolor("#111111")  # match figure face
        ax = plt.gca()
        ax.set_facecolor("#111111")
        ax.title.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

    st.pyplot(fig)
    plt.clf()


def generate_plots(df):
    prob_trace = go.Histogram(x=df['Probability'], nbinsx=20, marker_color='mediumblue',
                              name='Churn Probability Distribution')
    prob_layout = go.Layout(title='Churn Probability Distribution',
                            xaxis=dict(title='Probability'),
                            yaxis=dict(title='Count'),
                            bargap=0.1)
    prob_fig = go.Figure(data=[prob_trace], layout=prob_layout)

    churn_count = df['churned'].sum()
    retain_count = len(df) - churn_count
    pie_trace = go.Pie(labels=['Churned', 'Retain'],
                       values=[churn_count, retain_count],
                       marker_colors=['red', 'green'],
                       hoverinfo='label+percent',
                       textinfo='label+value')
    pie_layout = go.Layout(title='Churn vs Retain Distribution')
    pie_fig = go.Figure(data=[pie_trace], layout=pie_layout)

    return prob_fig, pie_fig

def plot_roc_curve(df=None):
    if y_train is not None and y_pred_proba_train is not None:
        fpr, tpr, _ = roc_curve(y_train, y_pred_proba_train)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='Overall ROC'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))

        # Add ROC for tenure < 31
        if df is not None and 'tenure' in df.columns:
            mask = df['tenure'] < 31
            if mask.any():
                y_sub = df[mask]['churned']
                y_pred_sub = model.predict_proba(df[mask].drop(columns=['customerID', 'churned']))[:, 1]
                fpr_sub, tpr_sub, _ = roc_curve(y_sub, y_pred_sub)
                fig.add_trace(go.Scatter(x=fpr_sub, y=tpr_sub, mode='lines', name='Tenure <= 30 ROC'))

        fig.update_layout(
            title='ROC Curve Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            legend=dict(x=0.6, y=0.05)
        )
        return fig
    else:
        return go.Figure()
    
# Streamlit UI
st.title("Customer Churn Prediction Dashboard")

st.header("1. Train the Model")
train_file = st.file_uploader("Upload Training CSV", type="csv", key="train")
if train_file is not None:
    df_train = pd.read_csv(train_file)
    df_train_clean = preprocess_data(df_train, is_train=True)
    model = train_model(df_train_clean)

    if model:
        X_train = df_train_clean.drop(columns=['customerID', 'churned'])
        plot_shap_summary(model, X_train, theme_mode)

        st.subheader("ROC Curve (Train Set + AUC-ROC on 30-day churn)")
        roc_fig = plot_roc_curve(df_train_clean)  # <-- pass df_train_clean here
        st.plotly_chart(roc_fig, use_container_width=True)


st.header("2. Predict on New Data")
test_file = st.file_uploader("Upload Test CSV", type="csv", key="test")
if test_file is not None:
    if model is None:
        st.warning("Please train the model first.")
    else:
        input_df = pd.read_csv(test_file)
        df_clean = preprocess_data(input_df, is_train=False)

        X_pred = df_clean[feature_columns]
        probas = model.predict_proba(X_pred)[:, 1]
        df_clean['Probability'] = probas
        df_clean['Churn_Predicted'] = (probas >= 0.5).astype(int)
        df_clean['churned'] = df_clean['Churn_Predicted']
        df_clean['Retain'] = 1 - df_clean['churned']
        df_clean['cost'] = df_clean['MonthlyCharges'] * df_clean['tenure']
        df_clean['contract_raw'] = input_df['Contract']

        df_predictions = df_clean[['customerID', 'Probability', 'churned', 'Retain', 'cost', 'contract_raw']].copy()
        df_predictions.rename(columns={'contract_raw': 'contract'}, inplace=True)

        st.markdown(
            f'<p style="color:#ffffff; font-size:18px;padding:12px;background-color:#173928;border-radius:10px;">✅ Processed {len(df_clean)} records. Model prediction completed.</p>',
            unsafe_allow_html=True
        )

        st.subheader("Top 10 High-Risk Customers")
        top_10 = df_predictions.sort_values('Probability', ascending=False).head(10)
        st.dataframe(top_10)

        st.subheader("Download Predictions")
        csv = df_predictions.to_csv(index=False).encode('utf-8')

        st.markdown("""
            <style>
            .download-button button {
                color: white !important;
                background-color: #ffffff;
            }
            </style>
        """, unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="download-button">', unsafe_allow_html=True)
            st.download_button("Download CSV", csv, "churn_predictions.csv", "text/csv")
            st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("Visualizations")
        hist_fig, pie_fig = generate_plots(df_predictions)
        st.plotly_chart(hist_fig, use_container_width=True)
        st.plotly_chart(pie_fig, use_container_width=True)
        st.subheader("Additional Visual Insights")

        # 1. Box Plot: Monthly Charges vs Churn Prediction
        box1 = go.Box(
            y=df_predictions[df_predictions['churned'] == 1]['cost'],
            name='Churned',
            marker_color='red'
        )
        box2 = go.Box(
            y=df_predictions[df_predictions['churned'] == 0]['cost'],
            name='Retained',
            marker_color='green'
        )
        box_layout = go.Layout(title="Cost Distribution by Churn Status", yaxis_title="Cost")
        st.plotly_chart(go.Figure(data=[box1, box2], layout=box_layout), use_container_width=True)

        # 2. Bar Plot: Churn Rate by Contract Type
        contract_churn = df_predictions.groupby('contract')['churned'].mean().reset_index()
        bar_contract = go.Bar(x=contract_churn['contract'], y=contract_churn['churned'], marker_color='purple')
        st.plotly_chart(go.Figure(data=[bar_contract],
                                  layout=go.Layout(title="Churn Rate by Contract Type",
                                                   xaxis_title="Contract Type",
                                                   yaxis_title="Churn Rate")),
                        use_container_width=True)

        # 3. Scatter Plot: Tenure vs Churn Probability
        scatter_fig = go.Figure()
        scatter_fig.add_trace(go.Scatter(
            x=df_clean['tenure'],
            y=df_clean['Probability'],
            mode='markers',
            marker=dict(size=6, color=df_clean['Probability'], colorscale='Bluered', showscale=True),
            text=df_clean['customerID']
        ))
        scatter_fig.update_layout(title='Tenure vs Churn Probability',
                                  xaxis_title='Tenure',
                                  yaxis_title='Churn Probability')
        st.plotly_chart(scatter_fig, use_container_width=True)

        # 4. Violin Plot: Churn Probability by Contract Type
        violin_data = []
        for contract in df_predictions['contract'].unique():
            violin_data.append(go.Violin(
                y=df_predictions[df_predictions['contract'] == contract]['Probability'],
                name=contract,
                box_visible=True,
                meanline_visible=True
            ))
        violin_layout = go.Layout(title="Churn Probability Distribution by Contract Type", yaxis_title="Probability")
        st.plotly_chart(go.Figure(data=violin_data, layout=violin_layout), use_container_width=True)


#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# Real time Prediction
#Sidebar
st.sidebar.header("🔮 Real-Time Churn Prediction")

if model is not None:
    st.sidebar.subheader("Enter Customer Details")

    # RESET Button Logic
    if "reset_form" not in st.session_state:
        st.session_state.reset_form = False

    # Detect reset request and clear form state
    if st.session_state.reset_form:
        for col in feature_columns:
            key = f"input_{col}"
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.reset_form = False
        st.rerun()

    with st.sidebar.form("customer_form"):
        input_data = {}

        for col in feature_columns:
            key = f"input_{col}"

            if col in label_encoders:
                options = list(label_encoders[col].classes_)
                default = options[0]  # choose default value
                input_data[col] = st.selectbox(
                    f"{col}", options, key=key
                )
            else:
                input_data[col] = st.number_input(
                    f"{col}", key=key, value=0.0
                )

        col1, col2 = st.columns(2)
        predict_clicked = col1.form_submit_button("Predict")
        reset_clicked = col2.form_submit_button("Reset")

    if predict_clicked:
        input_df = pd.DataFrame([input_data])

        for col, le in label_encoders.items():
            if col in input_df.columns:
                input_df[col] = input_df[col].map(lambda s: s if s in le.classes_ else 'No')
                if 'No' not in le.classes_:
                    le.classes_ = np.append(le.classes_, 'No')
                input_df[col] = le.transform(input_df[col])

        probability = model.predict_proba(input_df)[0][1]
        prediction = "Churn" if probability >= 0.5 else "Retain"

        st.sidebar.markdown(f"### Result: *{prediction}*")
        st.sidebar.markdown(f"*Churn Probability:* {probability:.2f}")
        plot_shap_single(model, input_df, theme_mode)

    if reset_clicked:
        st.session_state.reset_form = True
        st.rerun()

else:
    st.sidebar.warning("Train the model first to enable predictions.")

